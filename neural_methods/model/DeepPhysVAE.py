"""DeepPhysVAE - Combining DeepPhys with VAE for denoising and improved performance.
"""
import torch
import torch.nn as nn
from neural_methods.model.DeepPhys import DeepPhys

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 36x36
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 18x18
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 9x9
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128 * 9 * 9, 128)
        self.fc_logvar = nn.Linear(128 * 9 * 9, 128)
        self.fc_decode = nn.Linear(128, 128 * 9 * 9)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 18x18
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 36x36
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # 72x72
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x).view(-1, 128 * 9 * 9)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 128, 9, 9)
        return self.decoder(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class DeepPhysVAE(nn.Module):
    def __init__(self, deepphys, vae):
        super(DeepPhysVAE, self).__init__()
        self.deepphys = deepphys
        self.vae = vae
        # Further reduce r2 feature size by additional convolution and pooling layers
        self.reduce_r2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 35x35
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 2x2
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # 1x1
        )
        self.attn_fc = nn.Linear(deepphys.nb_dense + 128 + 128, deepphys.nb_dense)

    def forward(self, inputs):
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        d1 = torch.tanh(self.deepphys.motion_conv1(diff_input))
        d2 = torch.tanh(self.deepphys.motion_conv2(d1))
        # print(f"d2 shape: {d2.shape}")

        r1 = torch.tanh(self.deepphys.apperance_conv1(raw_input))
        r2 = torch.tanh(self.deepphys.apperance_conv2(r1))
        # print(f"r2 shape: {r2.shape}")

        g1 = torch.sigmoid(self.deepphys.apperance_att_conv1(r2))
        g1 = self.deepphys.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.deepphys.avg_pooling_1(gated1)
        d4 = self.deepphys.dropout_1(d3)

        r3 = self.deepphys.avg_pooling_2(r2)
        r4 = self.deepphys.dropout_2(r3)

        d5 = torch.tanh(self.deepphys.motion_conv3(d4))
        d6 = torch.tanh(self.deepphys.motion_conv4(d5))

        r5 = torch.tanh(self.deepphys.apperance_conv3(r4))
        r6 = torch.tanh(self.deepphys.apperance_conv4(r5))

        g2 = torch.sigmoid(self.deepphys.apperance_att_conv2(r6))
        g2 = self.deepphys.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.deepphys.avg_pooling_3(gated2)
        d8 = self.deepphys.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.deepphys.final_dense_1(d9))
        d11 = self.deepphys.dropout_4(d10)
        # print(f"d11 shape: {d11.shape}")

        vae_reconstructed, mu, logvar = self.vae(raw_input)
        vae_feature_reduced = mu
        # print(f"vae_feature_reduced shape: {vae_feature_reduced.shape}")

        r2_reduced = self.reduce_r2(r2).view(r2.size(0), -1)
        # print(f"r2_reduced shape: {r2_reduced.shape}")

        combined_feature = torch.cat((d11, vae_feature_reduced, r2_reduced), dim=1)
        # print(f"combined_feature shape: {combined_feature.shape}")

        attn_weights = torch.sigmoid(self.attn_fc(combined_feature))
        final_feature = attn_weights * d11 + (1 - attn_weights) * vae_feature_reduced
        # print(f"attn_weights shape: {attn_weights.shape}")
        # print(f"final_feature shape: {final_feature.shape}")

        out = self.deepphys.final_dense_2(final_feature)

        return out, vae_reconstructed, mu, logvar










