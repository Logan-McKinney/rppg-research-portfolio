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


class TS_CAN_Adam(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36, frame_depth=15):
        super(TS_CAN_Adam, self).__init__()
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
