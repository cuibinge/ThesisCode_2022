import torch
import torch.nn as nn


class GeometryGuidedAttention(nn.Module):
    """维度安全的几何注意力"""

    def __init__(self, hsi_dim, lidar_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(hsi_dim, hsi_dim // 8, 1)
        self.key_conv = nn.Sequential(
            nn.Conv2d(1, hsi_dim // 8, 1),
            nn.ELU())
        self.value_conv = nn.Conv2d(hsi_dim, hsi_dim, 1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, hsi_feat, lidar_elev):
        B, C, H, W = hsi_feat.size()

        # 特征对齐
        Q = self.query_conv(hsi_feat).view(B, -1, H * W)
        K = self.key_conv(lidar_elev).view(B, -1, H * W)
        energy = torch.bmm(Q.permute(0, 2, 1), K) / (C ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        # 特征融合
        V = self.value_conv(hsi_feat).view(B, -1, H * W)
        out = torch.bmm(V, attention.permute(0, 2, 1)).view(B, C, H, W)
        return hsi_feat + self.gamma * out