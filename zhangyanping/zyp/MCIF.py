import torch
from torchvision import models
import torch.nn as nn
import numpy as np
# from resnet import resnet34
#import resnet
from torch.nn import functional as F
# import torchsummary
from torch.nn import init
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


# 521 对第二层进行双注意力






def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                           has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode='bilinear',
                              align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output


class MCIF(nn.Module):
    def __init__(self, channel):
        super(MCIF, self).__init__()
        self.dilate11 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate22 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate33 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate44 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=1, padding=(1, 0))
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=2, padding=(2, 0))
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=4, padding=(4, 0))
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=8, padding=(8, 0))
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=1, padding=(0, 1))
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=2, padding=(0, 2))
        self.dilate7 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=4, padding=(0, 4))
        self.dilate8 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=8, padding=(0, 8))
        self.dconv = nn.Conv2d(channel * 5, channel, kernel_size=(1, 1), stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.ASPPH = ASPPPoolingH(in_channels=channel, out_channels=channel)
        self.ASPPW = ASPPPoolingW(in_channels=channel, out_channels=channel)

        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate11_out = nonlinearity(self.dilate11(x))
        dilate21_out = nonlinearity(self.dilate22(dilate11_out))
        dilate31_out = nonlinearity(self.dilate33(dilate21_out))
        dilate41_out = nonlinearity(self.dilate44(dilate31_out))

        dilate1_out = self.conv1(dilate11_out + dilate21_out + dilate31_out + dilate41_out)

        dilate12_out = nonlinearity(self.dilate1(x))
        dilate22_out = nonlinearity(self.dilate2(dilate12_out))
        dilate32_out = nonlinearity(self.dilate3(dilate22_out))
        dilate42_out = nonlinearity(self.dilate4(dilate32_out))

        dilate2_out = self.conv2(dilate12_out + dilate22_out + dilate32_out + dilate42_out)

        dilate13_out = nonlinearity(self.dilate5(x))
        dilate23_out = nonlinearity(self.dilate6(dilate13_out))
        dilate33_out = nonlinearity(self.dilate7(dilate23_out))
        dilate43_out = nonlinearity(self.dilate8(dilate33_out))

        dilate3_out = self.conv3(dilate13_out + dilate23_out + dilate33_out + dilate43_out)

        dilateH_out = self.ASPPH(x)
        dilateW_out = self.ASPPW(x)

        outsum = torch.cat([dilate1_out, dilate2_out, dilate3_out, dilateH_out, dilateW_out], dim=1)

        out = self.dconv(outsum)
        out = self.gamma * out + x * (1 - self.gamma)
        return out


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes=512, out_planes=512, ksize=3, stride=1, pad=1, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(DecoderBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last == False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs


class ASPPPoolingH(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingH, self).__init__(
            nn.AdaptiveAvgPool2d((32, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPPoolingW(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingW, self).__init__(
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)





