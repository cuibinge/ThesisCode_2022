import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformConv3D(nn.Module):
    """适配3D特征的2D可变形卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, groups=4):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size ** 2 * groups,
            kernel_size=3,
            padding=1
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=groups
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # 输入x维度: [B, C, D, H, W]
        B, C, D, H, W = x.size()

        # 合并通道和深度维度
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)

        # 生成偏移量
        offset = self.offset_conv(x_2d)

        # 应用可变形卷积
        out = self.conv(x_2d, offset=offset)
        out = out.view(B, D, -1, H, W).permute(0, 2, 1, 3, 4)
        return self.act(self.norm(out))


class MSDAM(nn.Module):
    def __init__(self, in_channels, scales=[64, 128], deform_groups=4):
        super().__init__()
        self.scale_convs = nn.ModuleList()

        # 修改后的多尺度处理
        for s in scales:
            pool = nn.Sequential(
                nn.AvgPool3d((1, 2, 2)) if s < 128 else nn.Identity(),
                DeformConv3D(in_channels, in_channels * 2, groups=deform_groups)
            )
            self.scale_convs.append(pool)

        self.fusion = nn.Conv2d(len(scales) * in_channels * 2, in_channels * 2, 1)

    def forward(self, hsi_feat, lidar_feat):
        # 处理LiDAR特征维度
        lidar_feat = lidar_feat.unsqueeze(2)  # [B,C,1,H,W]

        aligned_features = []
        for conv in self.scale_convs:
            h = conv(hsi_feat)
            l = conv(lidar_feat)

            # 对齐后的特征处理
            B, C, D, H, W = h.size()
            h = h.view(B, C * D, H, W)
            l = l.view(B, C * D, H, W)

            aligned = F.cosine_similarity(h, l, dim=1).unsqueeze(1)
            aligned_features.append(aligned)

        fused = self.fusion(torch.cat(aligned_features, dim=1))
        return fused