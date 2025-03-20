import torch
import torch.nn as nn
import torch.nn.functional as F


# class SpatialStream(nn.Module):
#     def __init__(self, in_channels=1):
#         super().__init__()
#         self.conv3d_1 = nn.Conv3d(in_channels, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1))
#         self.bn3d_1 = nn.BatchNorm3d(8)
#         self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1))
#         self.bn3d_2 = nn.BatchNorm3d(16)
#         self.pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2))
#
#     def forward(self, x):
#         x = F.gelu(self.bn3d_1(self.conv3d_1(x)))
#         x = self.pool3d(x)
#         x = F.gelu(self.bn3d_2(self.conv3d_2(x)))
#         return x
#
#
# class SpectralStream(nn.Module):
#     def __init__(self, in_channels, reduced_dim=64):
#         super().__init__()
#         self.conv1d_1 = nn.Conv1d(in_channels, 128, kernel_size=7, padding=3)
#         self.bn1d_1 = nn.BatchNorm1d(128)
#         self.conv1d_2 = nn.Conv1d(128, reduced_dim, kernel_size=5, padding=2)
#         self.bn1d_2 = nn.BatchNorm1d(reduced_dim)
#         self.pool1d = nn.AdaptiveMaxPool1d(64)
#
#     def forward(self, x):
#         x = F.gelu(self.bn1d_1(self.conv1d_1(x)))
#         x = self.pool1d(x)
#         x = F.gelu(self.bn1d_2(self.conv1d_2(x)))
#         return x
#
#
# class DualStream(nn.Module):
#     def __init__(self, spectral_bands, spatial_in=1):
#         super().__init__()
#         self.spatial_stream = SpatialStream(spatial_in)
#         self.spectral_stream = SpectralStream(spectral_bands)
#
#     def forward(self, x):
#         # Spatial stream processing
#         spatial = self.spatial_stream(x.unsqueeze(1))  # [B,1,D,H,W]
#         spatial = spatial.squeeze(2)  # 移除光谱维度（假设D维度已压缩）
#
#         # Spectral stream processing
#         B, C, H, W = x.shape
#         spectral_input = x.view(B, C, H * W)  # [B, C, H*W]
#         spectral = self.spectral_stream(spectral_input)
#
#         return spatial, spectral

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class SpatialStream(nn.Module):
#     def __init__(self, in_channels=1):
#         super().__init__()
#         # 调整卷积结构防止尺寸过小
#         self.conv3d_1 = nn.Conv3d(in_channels, 8, kernel_size=(3,3,3), padding=(1,1,1))
#         self.bn3d_1 = nn.BatchNorm3d(8)
#         self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(3,3,3), padding=(1,1,1))
#         self.bn3d_2 = nn.BatchNorm3d(16)
#         self.pool3d = nn.MaxPool3d(kernel_size=(1,2,2))
#         self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 5, 5))  # 确保输出尺寸
#
#     def forward(self, x):
#         x = F.gelu(self.bn3d_1(self.conv3d_1(x)))  # [B,8,30,11,11]
#         x = self.pool3d(x)            # [B,8,30,5,5]
#         x = F.gelu(self.bn3d_2(self.conv3d_2(x)))  # [B,16,30,5,5]
#         x = self.adaptive_pool(x)     # [B,16,1,5,5]
#         return x.squeeze(2)           # [B,16,5,5]
#
#
# class SpectralStream(nn.Module):
#     def __init__(self, in_channels, reduced_dim=64):
#         super().__init__()
#         # 保持原有结构不变
#         self.conv1d_1 = nn.Conv1d(in_channels, 128, kernel_size=7, padding=3)
#         self.bn1d_1 = nn.BatchNorm1d(128)
#         self.conv1d_2 = nn.Conv1d(128, reduced_dim, kernel_size=5, padding=2)
#         self.bn1d_2 = nn.BatchNorm1d(reduced_dim)
#         self.pool1d = nn.AdaptiveMaxPool1d(64)
#
#     def forward(self, x):
#         x = F.gelu(self.bn1d_1(self.conv1d_1(x)))
#         x = self.pool1d(x)
#         x = F.gelu(self.bn1d_2(self.conv1d_2(x)))
#         return x  # [B,64,64]


# class DualStream(nn.Module):
#     def __init__(self, spectral_bands, spatial_in=1):
#         super().__init__()
#         self.spatial_stream = SpatialStream(spatial_in)
#         self.spectral_stream = SpectralStream(spectral_bands)
#
#     def forward(self, x):
#         # 输入验证
#         assert x.dim() == 4, f"输入应为4D张量[B,C,H,W]，实际得到{x.shape}"
#
#         # 空间流处理
#         spatial = self.spatial_stream(x.unsqueeze(1))  # [B,1,C,H,W]->[B,16,5,5]
#
#         # 光谱流处理
#         B, C, H, W = x.shape
#         spectral = self.spectral_stream(x.view(B, C, H * W))  # [B,64,64]
#
#         return spatial, spectral


import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialStream(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入形状：[B,1,30,11,11] (C=30光谱通道)
        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.pool3d_1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # 空间下采样

        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(16)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 5, 5))  # 固定输出形状

    def forward(self, x):
        # 输入x形状：[B,1,30,11,11]
        x = F.gelu(self.bn3d_1(self.conv3d_1(x)))  # [B,8,30,11,11]
        x = self.pool3d_1(x)  # [B,8,30,5,5]
        x = F.gelu(self.bn3d_2(self.conv3d_2(x)))  # [B,16,30,5,5]
        x = self.adaptive_pool(x)  # [B,16,1,5,5]
        return x.squeeze(2)  # [B,16,5,5] 四维输出


class SpectralStream(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 输入形状：[B,C,HW] (C=30光谱通道)
        self.conv1d_1 = nn.Conv1d(in_channels, 128, 7, padding=3)
        self.bn1d_1 = nn.BatchNorm1d(128)
        self.pool1d = nn.MaxPool1d(2)

        self.conv1d_2 = nn.Conv1d(128, 64, 5, padding=2)
        self.bn1d_2 = nn.BatchNorm1d(64)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(64)

    def forward(self, x):
        # 输入x形状：[B,30,121] (假设原始图像11x11)
        x = F.gelu(self.bn1d_1(self.conv1d_1(x)))  # [B,128,121]
        x = self.pool1d(x)  # [B,128,60]
        x = F.gelu(self.bn1d_2(self.conv1d_2(x)))  # [B,64,60]
        x = self.adaptive_pool(x)  # [B,64,64]
        return x  # 三维输出


class DualStream(nn.Module):
    def __init__(self, spectral_bands):
        super().__init__()
        self.spatial_stream = SpatialStream()
        self.spectral_stream = SpectralStream(spectral_bands)

    def forward(self, x):
        # 输入验证
        assert x.dim() == 4, f"输入应为[B,C,H,W]，得到{x.shape}"

        # 空间流处理
        spatial = self.spatial_stream(x.unsqueeze(1))  # [B,1,C,H,W]
        assert spatial.dim() == 4, f"空间特征应为4D，得到{spatial.shape}"

        # 光谱流处理
        B, C, H, W = x.shape
        spectral = self.spectral_stream(x.view(B, C, H * W))
        assert spectral.dim() == 3, f"光谱特征应为3D，得到{spectral.shape}"

        return spatial, spectral
if __name__ == '__main__':
    net = DualStream(30)
    print(net)
    input = torch.randn(12, 30, 11, 11)
    output_a, output_b = net(input)
    print(output_a.shape)  # 应输出 torch.Size([12, 16, 5, 5])
    print(output_b.shape)  # 应输出 torch.Size([12, 64, 64])