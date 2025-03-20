'''
2023.2.26
跳跃链接处增加aspp
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from base.funtion import ASPP
from itertools import chain

# from funtion import ASPP
# from funtion import ASPP
# import utils.ASPP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return down_conv


# def x2conv(in_channels, out_channels, inner_channels=None):
#     inner_channels = out_channels // 2 if inner_channels is None else inner_channels
#     down_conv = nn.Sequential(
#         nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(inner_channels),
#         nn.ReLU(inplace=True),
#         # nn.GELU(),
#         nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         # nn.GELU()
#         nn.ReLU(inplace=True)
#     )
#     return down_conv

# 2023-12-16 添加SE模块 增强通道特征

'''-------------一、SE_Block模块-----------------------------'''


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


'''-------------二、BasicBlock模块-----------------------------'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
        # SE_Block放在BN之后，shortcut之前
        self.SE = SE_Block(outchannel)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, self.expansion * outchannel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * outchannel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        SE_out = self.SE(out)
        out = out * SE_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)
        # self.aspp1 = ASPP(512, [6, 12, 18], 512)
        # self.aspp2 = ASPP(256, [6, 12, 18], 256)
        # self.aspp3 = ASPP(128, [6, 12, 18], 128)
        # self.aspp4 = ASPP(64, [6, 12, 18], 64)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)
        # x_copy = self.aspp1(x_copy)

        if (x.size(2) != x_copy.size(2) or x.size(3) != x_copy.size(3)):
            if interpolate:
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)), mode="bilinear", align_corners=True)
            else:
                diffy = x_copy.size()[2] - x.size()[2]
                diffx = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffx // 2, diffx - diffx // 2),
                          diffy // 2, diffy - diffy // 2)

        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=4, freeze_bn=False):
        super(UNet, self).__init__()

        self.start_conv = x2conv(in_channels, 64)
        self.down1 = Encoder(64, 128)
        self.down2 = Encoder(128, 256)
        self.down3 = Encoder(256, 512)
        self.down4 = Encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)
        # self.BasicBlock = BasicBlock(1024,1024)
        # self.BasicBlock1 = BasicBlock(512, 512)
        # self.BasicBlock2 = BasicBlock(256, 256)
        # self.BasicBlock3 = BasicBlock(128, 128)
        # self.BasicBlock4 = BasicBlock(64, 64)
        self.up1 = Decoder(1024, 512)
        self.up2 = Decoder(512, 256)
        self.up3 = Decoder(256, 128)
        self.up4 = Decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()

        self.dropout = nn.Dropout(p=0.5)

        # ASPP

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        num = 0
        for m in self.modules():
            num += 1
            # print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        print(f"depth: {num}")

    def forward(self, x):
        x1 = self.start_conv(x)  # 64 256 256
        x2 = self.down1(x1)  # 128 128 128
        x3 = self.down2(x2)  # 256 64 64
        x4 = self.down3(x3)  # 512 32 32
        x = self.middle_conv(self.down4(x4))  # 1024 16 16
        # x = self.BasicBlock(self.down4(x4))
        # x = self.dropout(x)
        # x = self.up1(self.BasicBlock1(x4), x)
        # x = self.up2(self.BasicBlock2(x3), x)
        # x = self.up3(self.BasicBlock3(x2), x)
        # x = self.up4(self.BasicBlock4(x1), x)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)
        # x_sea = x[:, 6:7, :, :]
        # x_sea = torch.cat([x_sea,torch.zeros(x_sea.shape).to(device)], dim=1)
        # print(f"x: {x.shape}")
        #         x = self.softmax(x)

        return x

from torchsummary import summary

if __name__ == '__main__':
    model = UNet(2, 4)

    # x = torch.rand((2,4,128,128))
    summary(model, (4, 64, 64), device="cpu")
    # print(model(x).shape)
