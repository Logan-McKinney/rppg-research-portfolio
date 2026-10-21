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
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class DeepPhysMultiTask(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36, frame_depth=15):
        super(DeepPhysMultiTask, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.frame_depth = frame_depth

        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)

        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Appearance branch convs
        self.appearance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.appearance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.appearance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.appearance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)

        # Attention layers
        self.appearance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.appearance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)

        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')

        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

        # Peak and trough output layers
        self.peak_output = nn.Linear(self.nb_dense, 1, bias=True)
        self.trough_output = nn.Linear(self.nb_dense, 1, bias=True)

        # LSTM for learning dependencies within each heart rate cycle
        self.lstm = nn.LSTM(input_size=self.nb_dense, hidden_size=self.nb_dense, batch_first=True)

    def forward(self, inputs, params=None):
        # === DIAGNOSTIC: one-time shape check ===
        batch_size = inputs.size(0)

        if not getattr(self, '_fwd_logged', False):

            print(f"[Model.forward] inputs shape={tuple(inputs.shape)} "
                  f"dtype={inputs.dtype} device={inputs.device}")

            print(f"[Model.forward] frame_depth={self.frame_depth}")

            self._fwd_logged = True
            
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        # Apply TSM to motion branch inputs
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

        # Apply TSM to motion branch intermediate features
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
        
        # === DIAGNOSTIC: check flatten dim matches final_dense_1 expectation ===

        if not getattr(self, '_flatten_logged', False):

            print(f"[Model.forward] d9 (flattened) shape={tuple(d9.shape)} "
                  f"— expected ({d8.size(0)}, {d9.shape[1]})")

            self._flatten_logged = True
        
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)

        
        # Reshape for LSTM
        d11 = d11.view(batch_size // self.frame_depth, self.frame_depth, -1)

        # === DIAGNOSTIC: LSTM reshape check ===

        if not getattr(self, '_lstm_logged', False):

            print(f"[Model.forward] LSTM input shape={tuple(d11.shape)} "
                  f"(batch={batch_size}, frame_depth={self.frame_depth})")

            self._lstm_logged = True


        # Apply LSTM
        lstm_out, _ = self.lstm(d11)

        # Flatten LSTM output
        lstm_out = lstm_out.contiguous().view(-1, self.nb_dense)

        # Final predictions
        out = self.final_dense_2(lstm_out)
        
         # === DIAGNOSTIC: output sanity ===

        if not getattr(self, '_out_logged', False):

            print(f"[Model.forward] out shape={tuple(out.shape)} "
                  f"min={out.min().item():.4f} max={out.max().item():.4f} "
                  f"nan={out.isnan().any().item()}")

            self._out_logged = True

        # Peak and trough outputs
        peak_out = self.peak_output(lstm_out)
        trough_out = self.trough_output(lstm_out)

        return out, peak_out, trough_out
