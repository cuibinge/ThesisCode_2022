import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# class DualTemporalFusionLayer(nn.Module):
#     def __init__(self):
#         super(DualTemporalFusionLayer, self).__init__()
#         # 使用自定义卷积层来处理 6 通道输入
#         self.conv = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)  # 将 6 通道转换为 3 通道
#         self.resnet = models.resnet18(pretrained=True)
#         # self.resnet.conv1 = self.conv  # 替换 ResNet 的第一层卷积
#
#     def forward(self, img1, img2):
#         # 将影像沿通道维度拼接
#         fused_input = torch.cat((img1, img2), dim=1)  # img1 和 img2 需为 [batch_size, 3, H, W]
#         fused_input = self.conv(fused_input)
#         # 通过 ResNet 处理
#         fused_output = self.resnet(fused_input)
#
#         return fused_output
from models.help_funcs import TwoLayerConv2d


class DualTemporalFusionLayer(nn.Module):
    def __init__(self):
        super(DualTemporalFusionLayer, self).__init__()
        # 使用自定义卷积层来处理 6 通道输入
        self.conv = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)  # 将 6 通道转换为 3 通道
        self.resnet = models.resnet18(pretrained=True)
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        # self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = 5
        expand = 1
        self.if_upsample_2x = True
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        # self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()
        # self.resnet.conv1 = self.conv  # 替换 ResNet 的第一层卷积

    def forward(self, img1, img2):
        # resnet layers
        fused_input = torch.cat((img1, img2), dim=1)  # img1 和 img2 需为 [batch_size, 3, H, W]
        x = self.conv(fused_input)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transforms.ToTensor()(img)
    return img.unsqueeze(0)  # 添加批次维度

# 读取图像
img1 = load_image(r'G:\project\New Folder\A\1_2_3.png')  # 替换为实际路径
img2 = load_image(r'G:\project\New Folder\B\1_2_3.png')  # 替换为实际路径

# 实例化融合层
fusion_layer = DualTemporalFusionLayer()

# 将影像输入融合层
fused_output = fusion_layer(img1, img2)

# 查看输出结果
print(fused_output.shape)  # 应输出 (1, 2) 的形状
# 可视化原始图像和融合后的特征图
def visualize_images(img1, img2, fused_output):
    # 显示原始图像
    img1_np = img1.squeeze(0).permute(1, 2, 0).numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).numpy()
    plt.subplot(1, 3, 1)
    plt.imshow(img1_np)
    plt.title('Image 1')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img2_np)
    plt.title('Image 2')
    plt.axis('off')

    # 显示融合后的特征图
    fused_grid = make_grid(fused_output.squeeze(0), nrow=8, normalize=True)
    fused_grid_np = fused_grid.permute(1, 2, 0).detach().numpy()

    # 如果特征图通道数不为 3，取前三个通道进行可视化
    if fused_grid_np.shape[2]!= 3:
        fused_grid_np = fused_grid_np[:, :, :3]

    plt.subplot(1, 3, 3)
    plt.imshow(fused_grid_np)
    plt.title('Fused Output')
    plt.axis('off')

    plt.show()
visualize_images(img1, img2, fused_output)