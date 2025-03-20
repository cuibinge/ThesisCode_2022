import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# class FeatureAdjuster(nn.Module):
#     def __init__(self, in_channels, target_channels):
#         super(FeatureAdjuster, self).__init__()
#         # 初始化卷积层，输入通道数为in_channels，输出通道数为target_channels
#         self.adjust_channels = nn.Conv2d(in_channels=in_channels, out_channels=target_channels, kernel_size=1)


#     def forward(self, feature_maps, target_size):
#         """
#         对特征图进行上采样并调整通道数。

#         :param feature_maps: 输入的特征图，形状为 (B, C, H, W)
#         :param target_size: 目标空间尺寸，即 (H, W)
#         :return: 上采样并调整通道数后的特征图
#         """
#         # 上采样特征图
#         upsampled_feature_maps = F.interpolate(feature_maps, size=target_size, mode='bilinear', align_corners=True)

#         # 应用卷积层来调整通道数
#         adjusted_feature_maps = self.adjust_channels(upsampled_feature_maps)

#         return adjusted_feature_maps

# 解码器网络，使用反卷积逐步上采样
class FeatureAdjuster(nn.Module):
    def __init__(self, in_channels):
        super(FeatureAdjuster, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 恢复到3通道RGB图像
            # nn.Sigmoid()  # 将输出值映射到[0, 1]
        )

    def forward(self, x):
        decoded_image = self.decoder(x)
        return decoded_image