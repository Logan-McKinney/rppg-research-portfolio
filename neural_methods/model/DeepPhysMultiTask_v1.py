import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class DeepPhysMultiTask_v1(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36, frame_depth=15):
        super(DeepPhysMultiTask_v1, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.frame_depth = frame_depth

        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)

        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        self.appearance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.appearance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.appearance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.appearance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        self.appearance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.appearance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)

        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        if img_size == 36:
            in_features = 3136
        elif img_size == 72:
            in_features = 16384
        elif img_size == 96:
            in_features = 30976
        else:
            raise Exception('Unsupported image size')

        self.final_dense_1 = nn.Linear(in_features, self.nb_dense, bias=True)
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

        self.peak_output = nn.Linear(self.nb_dense, 1, bias=True)
        self.trough_output = nn.Linear(self.nb_dense, 1, bias=True)

        self.lstm = nn.LSTM(input_size=self.nb_dense, hidden_size=self.nb_dense, batch_first=True)

        # Artifacts-removal branch: 9xT (8 auxiliary + 1 rPPG) -> Conv1D -> 2-layer LSTM -> a(t), b(t)
        # Dynamic affine correction: y(t) = a(t)*x(t) + b(t)
        self.noise_conv1 = nn.Conv1d(9, 32, kernel_size=3, padding=1)
        self.noise_conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.noise_conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.noise_lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.noise_fc = nn.Linear(64, 2)  # outputs a(t) and b(t)
        
        # Initialize noise branch as near-identity so it starts as a passthrough
        # and learns small corrections rather than random ones
        nn.init.zeros_(self.noise_fc.weight)
        nn.init.constant_(self.noise_fc.bias, 0)
        self.noise_fc.bias.data[0] = 1.0  # a(t) starts near 1, b(t) starts near 0

    def _extract_auxiliary_8xt(self, diff_input, raw_input):
        """Heuristic fallback auxiliary features from video frames when no external
        motion CSV is provided. Computes motion (3), pose (3), lum (1), and a
        placeholder zero channel (1) in place of the removed distance estimate.
        Returns shape [N, 8, T].
        """
        eps = 1e-6
        dev = diff_input.device
        dtype = diff_input.dtype
        N = diff_input.size(0) // self.frame_depth
        T = self.frame_depth
        H, W = diff_input.size(2), diff_input.size(3)

        # Lum proxy: grayscale spatial mean
        gray = (
            raw_input[:, 0:1] * 0.299
            + raw_input[:, 1:2] * 0.587
            + raw_input[:, 2:3] * 0.114
        )
        lum = gray.mean(dim=(2, 3))  # [N*T, 1]

        # Pose proxy: center of mass of |diff| intensity
        intensity = diff_input.abs().mean(dim=1)
        mass = intensity.sum(dim=(1, 2), keepdim=True) + eps
        w_idx = torch.arange(W, device=dev, dtype=dtype).view(1, 1, W)
        h_idx = torch.arange(H, device=dev, dtype=dtype).view(1, H, 1)
        com_x = (intensity * w_idx).sum(dim=(1, 2), keepdim=True) / mass
        com_y = (intensity * h_idx).sum(dim=(1, 2), keepdim=True) / mass
        com_x = com_x.squeeze(-1).squeeze(-1) / (W - 1 + eps)
        com_y = com_y.squeeze(-1).squeeze(-1) / (H - 1 + eps)
        pose_z = intensity.mean(dim=(1, 2))
        pose_xyz = torch.stack([com_x, com_y, pose_z], dim=1)  # [N*T, 3]

        # Reshape
        lum = lum.view(N, T, 1)
        pose_xyz = pose_xyz.view(N, T, 3)

        # Motion proxy: temporal diff of pose
        motion = torch.zeros_like(pose_xyz)
        motion[:, 1:, :] = pose_xyz[:, 1:, :] - pose_xyz[:, :-1, :]

        # --- ADDED: placeholder zero channel replaces the removed distance estimate ---
        # Previously channel 8 was spatial std of raw_input used as a distance proxy.
        # Distance estimation has been removed per design decision. A zero tensor is
        # used to keep the 8-channel interface intact without introducing false signal.
        placeholder = torch.zeros(N, T, 1, device=dev, dtype=dtype)

        # [Motion(3), Pose(3), Lum(1), Placeholder(1)] -> [N, T, 8] -> [N, 8, T]
        model_input = torch.cat([motion, pose_xyz, lum, placeholder], dim=2)
        return model_input.transpose(1, 2)

    # --- ADDED: helper to build auxiliary tensor from Patrick's MediaPipe CSV features ---
    # Expected CSV columns used (in order):
    #   head_tx_vel  -> channel 0: head translation velocity x (motion)
    #   head_ty_vel  -> channel 1: head translation velocity y (motion)
    #   lips_vel     -> channel 2: lip landmark velocity (motion proxy for facial motion)
    #   yaw          -> channel 3: head yaw angle (pose)
    #   pitch        -> channel 4: head pitch angle (pose)
    #   roll         -> channel 5: head roll angle (pose)
    #   lips_accel   -> channel 6: lip acceleration (replaces lum — stronger motion signal)
    #   eor_l        -> channel 7: left eye openness ratio (blink/expression signal)
    #
    # motion_csv_features: tensor of shape [N, T, 8] loaded and sliced by the dataset class,
    # where N is batch size (number of clips) and T is frame_depth (clip length).
    # Returns [N, 8, T] to match the noise branch input format.
    @staticmethod
    def build_motion_features_from_csv(motion_csv_features):
        """Convert [N, T, 8] CSV feature tensor to [N, 8, T] for the noise branch."""
        return motion_csv_features.transpose(1, 2)

    def forward(self, inputs, motion_csv_features=None, params=None):
        """
        Args:
            inputs: [N*T, 6, H, W] — standard rPPG input (diff + raw frames)
            motion_csv_features: [N, T, 8] tensor from Patrick's MediaPipe CSV, or None.
                If provided, real motion features replace the heuristic auxiliary extraction.
                If None, falls back to _extract_auxiliary_8xt heuristics.
                Expected column order: [head_tx_vel, head_ty_vel, lips_vel,
                                        yaw, pitch, roll, lips_accel, eor_l]
        """
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        # --- ADDED: use Patrick's real MediaPipe features if provided, else fall back
        # to heuristic extraction from the video frames directly ---
        if motion_csv_features is not None:
            # Real motion features from Patrick's CSV — [N, T, 8] -> [N, 8, T]
            model_input_8xt = self.build_motion_features_from_csv(
                motion_csv_features.to(inputs.device)
            )
        else:
            # Fallback: heuristic proxy features computed from the input frames
            model_input_8xt = self._extract_auxiliary_8xt(diff_input, raw_input)

        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.appearance_conv1(raw_input))
        r2 = torch.tanh(self.appearance_conv2(r1))

        g1 = torch.sigmoid(self.appearance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.appearance_conv3(r4))
        r6 = torch.tanh(self.appearance_conv4(r5))

        g2 = torch.sigmoid(self.appearance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)

        batch_size = inputs.size(0)
        d11 = d11.view(batch_size // self.frame_depth, self.frame_depth, -1)

        lstm_out, _ = self.lstm(d11)
        lstm_out = lstm_out.contiguous().view(-1, self.nb_dense)
        pulse_out = self.final_dense_2(lstm_out)  # base rPPG x(t) [N*T, 1]

        # Artifacts branch: [N, 9, T] (8 auxiliary + 1 rPPG) -> Conv1D -> LSTM -> a(t), b(t)
        pulse_out_nt = pulse_out.view(batch_size // self.frame_depth, self.frame_depth, 1).transpose(1, 2)  # [N, 1, T]
        noise_input = torch.cat([model_input_8xt, pulse_out_nt], dim=1)  # [N, 9, T]
        x_conv = torch.tanh(self.noise_conv1(noise_input))
        x_conv = torch.tanh(self.noise_conv2(x_conv))
        x_conv = torch.tanh(self.noise_conv3(x_conv))
        x_conv = x_conv.transpose(1, 2)                    # [N, T, 64]
        noise_lstm_out, _ = self.noise_lstm(x_conv)        # [N, T, 64]
        ab = self.noise_fc(noise_lstm_out)                  # [N, T, 2]
        a_t = ab[:, :, 0:1].contiguous().view(-1, 1)       # [N*T, 1]
        b_t = ab[:, :, 1:2].contiguous().view(-1, 1)       # [N*T, 1]

        out = a_t * pulse_out + b_t  # denoised rPPG: y(t) = a(t)*x(t) + b(t)

        # === DIAGNOSTIC: one-time noise branch check ===
        if not getattr(self, '_noise_logged', False) or not self.training:
            print(f"[v1.forward] motion_csv_features: "
                f"{'provided' if motion_csv_features is not None else 'None (fallback)'}")
            print(f"[v1.forward] a(t): min={a_t.min().item():.4f} max={a_t.max().item():.4f} "
                 f"mean={a_t.mean().item():.4f}")
            print(f"[v1.forward] b(t): min={b_t.min().item():.4f} max={b_t.max().item():.4f} "
                  f"mean={b_t.mean().item():.4f}")
            print(f"[v1.forward] pulse_out (x(t)): min={pulse_out.min().item():.4f} "
                  f"max={pulse_out.max().item():.4f}")
            print(f"[v1.forward] out (y(t)): min={out.min().item():.4f} "
                  f"max={out.max().item():.4f}")
            self._noise_logged = True
# === END DIAGNOSTIC ===

        peak_out = self.peak_output(lstm_out)
        trough_out = self.trough_output(lstm_out)

        return out, peak_out, trough_out
#-----------------------------OLD-------------------------------------------------------
'''import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class DeepPhysMultiTask_v1(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36, frame_depth=15):
        super(DeepPhysMultiTask_v1, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.frame_depth = frame_depth

        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)

        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        self.appearance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.appearance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.appearance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.appearance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        self.appearance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.appearance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)

        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        if img_size == 36:
            in_features = 3136
        elif img_size == 72:
            in_features = 16384
        elif img_size == 96:
            in_features = 30976
        else:
            raise Exception('Unsupported image size')

        self.final_dense_1 = nn.Linear(in_features, self.nb_dense, bias=True)
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

        self.noise_dense_1 = nn.Linear(in_features, self.nb_dense, bias=True)
        self.noise_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

        self.peak_output = nn.Linear(self.nb_dense, 1, bias=True)
        self.trough_output = nn.Linear(self.nb_dense, 1, bias=True)

        self.lstm = nn.LSTM(input_size=self.nb_dense, hidden_size=self.nb_dense, batch_first=True)
        self.noise_lstm = nn.LSTM(input_size=self.nb_dense, hidden_size=self.nb_dense, batch_first=True)

    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.appearance_conv1(raw_input))
        r2 = torch.tanh(self.appearance_conv2(r1))

        g1 = torch.sigmoid(self.appearance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.appearance_conv3(r4))
        r6 = torch.tanh(self.appearance_conv4(r5))

        g2 = torch.sigmoid(self.appearance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)

        n7 = self.avg_pooling_3(d6)
        n8 = self.dropout_3(n7)
        n9 = n8.view(n8.size(0), -1)
        n10 = torch.tanh(self.noise_dense_1(n9))
        n11 = self.dropout_4(n10)

        batch_size = inputs.size(0)
        d11 = d11.view(batch_size // self.frame_depth, self.frame_depth, -1)
        n11 = n11.view(batch_size // self.frame_depth, self.frame_depth, -1)

        lstm_out, _ = self.lstm(d11)
        noise_lstm_out, _ = self.noise_lstm(n11)

        lstm_out = lstm_out.contiguous().view(-1, self.nb_dense)
        noise_lstm_out = noise_lstm_out.contiguous().view(-1, self.nb_dense)

        pulse_out = self.final_dense_2(lstm_out)
        noise_out = self.noise_dense_2(noise_lstm_out)
        out = pulse_out - noise_out

        peak_out = self.peak_output(lstm_out)
        trough_out = self.trough_output(lstm_out)

        return out, peak_out, trough_out
'''