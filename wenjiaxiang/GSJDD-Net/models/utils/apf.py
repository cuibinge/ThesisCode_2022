import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveProgressiveFusion(nn.Module):
    """自适应渐进式特征融合"""
    def __init__(self, channels, num_scales=2):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * channels, channels // 4, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_scales-1)  # 修正数量匹配
        ])
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * channels, channels, 3, padding=1),
                nn.InstanceNorm2d(channels)
            ) for _ in range(num_scales)
        ])

    def forward(self, features):
        fused = []
        for i, (hsi, lidar) in enumerate(features):
            if i == 0:
                fused_feat = self.fuse_convs[i](torch.cat([hsi, lidar], dim=1))
            else:
                gate = self.gates[i-1](torch.cat([hsi, lidar], dim=1))
                fused_feat = gate * hsi + (1 - gate) * lidar
                fused_feat = self.fuse_convs[i](fused_feat)
            fused.append(fused_feat)

        # 跨层级特征聚合
        final_feat = sum([
            F.interpolate(f, scale_factor=2**i, mode='bilinear', align_corners=False)
            for i, f in enumerate(reversed(fused))
        ])
        return final_feat