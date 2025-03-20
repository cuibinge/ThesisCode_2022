import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CDAM4(nn.Module):
    def __init__(self, k_size=5):
        super(CDAM4, self).__init__()
        self.h = 8
        self.w = 8
        self.avg_pool_x = nn.AdaptiveAvgPool2d((self.h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, self.w))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(64, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(64, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Conv2d(64 * 4 * 5, 64 * 5, kernel_size=3, padding=1, bias=False)
        self.conv111 = nn.Conv2d(in_channels=64 * 5 * 2, out_channels=64 * 5 * 2, kernel_size=1, padding=0, stride=1)
        self.conv222 = nn.Conv2d(in_channels=64 * 5 * 2, out_channels=64 * 5 * 2, kernel_size=1, padding=0, stride=1)

        self.conv1h = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(self.h, 1), padding=(0, 0), stride=1)
        self.conv1s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, self.w), padding=(0, 0), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        n, c, h, w = x.size()

        # 对x进行水平方向的池化
        y1 = self.avg_pool_x(x)  # 输出形状为 (n, c, self.h, 1)
        # 调整y1的形状以匹配Conv1d的输入要求
        y1 = y1.view(n, c, self.h).transpose(1, 2)  # 形状变为 (n, self.h, c)

        # 对x进行垂直方向的池化
        y2 = self.avg_pool_y(x)  # 输出形状为 (n, c, 1, self.w)
        # 调整y2的形状以匹配Conv1d的输入要求
        y2 = y2.view(n, c, self.w).transpose(1, 2)  # 形状变为 (n, self.w, c)

        # 通过Conv1d和激活函数处理y1和y2
        y1 = self.sigmoid(self.conv11(self.relu1(self.conv1(y1))))
        y2 = self.sigmoid(self.conv22(self.relu1(self.conv2(y2))))

        # 将y1和y2的结果上采样到与x相同的空间尺寸
        y1 = F.interpolate(y1.unsqueeze(-1), size=(h, w), mode='nearest').squeeze(-1)
        y2 = F.interpolate(y2.unsqueeze(-1), size=(h, w), mode='nearest').squeeze(-1)

        # 将y1和y2的结果与原始输入x相乘，并合并
        yac = self.conv111(torch.cat([x * y1, x * y2], dim=1))


        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3, x * y4], dim=1))

        out = self.convout(torch.cat([yac, yap], dim=1))
        return out

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FSFF_4(nn.Module):
    def __init__(self, width=64, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(FSFF_4, self).__init__()
        self.up_kwargs = up_kwargs

        # 定义卷积和BN层，现在处理80个输入通道，输出宽度为64
        self.conv = nn.Sequential(
            nn.Conv2d(80, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        # 输出层，从宽度64输出16个通道
        self.conv_out = nn.Sequential(
            nn.Conv2d(width, 16, 1, padding=0, bias=False),
            norm_layer(16)
        )

        # CDAM注意力模块
        self.CDAM = CDAM4()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        # 直接处理单个输入
        feat = self.conv(x)
        feat = self.conv_out(self.CDAM(feat))
        return feat
