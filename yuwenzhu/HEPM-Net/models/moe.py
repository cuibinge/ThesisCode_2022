import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class ResBlock2D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class ResBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.shortcut = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # 输入维度：[B, in_c, L]
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)  # 保持长度不变
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        # 输入x形状：[B, in_c, L]
        residual = self.shortcut(x)  # [B, out_c, L]
        x = F.gelu(self.bn1(self.conv1(x)))  # [B, out_c, L]
        x = self.bn2(self.conv2(x))  # [B, out_c, L]
        return F.gelu(x + residual)  # [B, out_c, L]


class HeterogeneousExpert(nn.Module):
    def __init__(self, expert_type, in_channels):
        super().__init__()
        self.expert_type = expert_type
        layers = []

        # 根据专家类型配置参数
        if expert_type == "spatial":
            # 输入形状：[B, 16, 5, 5]
            block = ResBlock2D
            self.pool = nn.MaxPool2d(2)
            expansion = 2
            final_pool = nn.AdaptiveAvgPool2d(1)
            max_pool_layers = 1  # 减少池化次数
        elif expert_type == "spectral":
            # 输入形状：[B, 64, 64]
            block = ResBlock1D
            self.pool = nn.MaxPool1d(2)
            expansion = 2
            final_pool = nn.AdaptiveAvgPool1d(1)
            max_pool_layers = 3
        else:  # spatio_spectral
            # 输入形状：[B, 281, 16] (修正后的正确维度)
            block = ResBlock1D
            self.pool = nn.MaxPool1d(2)
            expansion = 2
            final_pool = nn.AdaptiveAvgPool1d(1)
            max_pool_layers = 2  # 适配新维度

        current_channels = in_channels
        for i in range(3):
            out_channels = int(current_channels * expansion)
            layers.append(block(current_channels, out_channels))

            # 控制池化层数量
            if i < max_pool_layers:
                layers.append(self.pool)

            current_channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(
            final_pool,
            nn.Flatten(),
            nn.Linear(current_channels, 64)  # 输出统一为64维
        )

    def forward(self, x):
        # 维度验证
        if self.expert_type == "spatio_spectral":
            assert x.dim() == 3, f"需要3D输入，得到{x.shape}"
            assert x.size(1) == 281, f"期望281通道，得到{x.size(1)}"

        x = self.layers(x)  # 经过所有层级处理
        return [], self.final_layer(x)  # 返回空浅层特征