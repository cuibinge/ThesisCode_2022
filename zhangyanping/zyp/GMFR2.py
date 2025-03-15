import torch
from torch import nn
import torch.nn.functional as F
import math
from model.FRCFNet.SRU import SRU16


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GMFR2(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in//4
        k_size =3
        pad =(k_size -1) // 2

        self.SRU16 = SRU16(c_dim_in)

        self.params_c = nn.Parameter(torch.Tensor(1, c_dim_in, 1, 1), requires_grad=True)
        nn.init.ones_(self.params_c)
        self.conv_c = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_x = nn.Parameter(torch.Tensor(1, 1, x, 1), requires_grad=True)
        nn.init.ones_(self.params_x)
        self.conv_x = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_y = nn.Parameter(torch.Tensor(1, 1, 1, y), requires_grad=True)
        nn.init.ones_(self.params_y)
        self.conv_y = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in))

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1))

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        params_c = self.params_c
        x1 = x1 * self.conv_c(F.interpolate(params_c, size=x1.shape[2:4] ,mode='bilinear', align_corners=True))
        x1 = self.SRU16(x1)

        x2 = x2.permute(0, 3, 1, 2)
        params_x = self.params_x
        x2 = x2 * self.conv_x(F.interpolate(params_x, size=x2.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.SRU16(x2)

        x3 = x3.permute(0, 2, 1, 3)
        params_y = self.params_y
        x3 = x3 * self.conv_y(F.interpolate(params_y, size=x3.shape[2:4] ,mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x3 = self.SRU16(x3)

        x4 = self.dw(x4)
        x4 = self.SRU16(x4)

        x = torch.cat([x1,x2,x3,x4],dim=1)

        x = self.norm2(x)
        x = self.ldw(x)

        return x

