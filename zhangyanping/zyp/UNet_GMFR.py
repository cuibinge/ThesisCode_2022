import torch
from torch import nn
import torch.nn.functional as F
from model.zyp.GMFR2 import GMFR2
from model.zyp.GMFR1 import GMFR1


import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv, self).__init__()
        self.squre = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.ReLU = nn.ReLU(True)

    def forward(self, x):
        x1 = self.squre(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.ReLU(self.bn(x1 + x2 + x3))


class UNet_GMFR(nn.Module):
    def __init__(self, band_num, class_num):
        super(UNet_GMFR, self).__init__()
        self.band_num = band_num
        self.class_num = class_num
        self.name = 'UNet_GMFR'

        channels = [16, 32, 64, 128, 256]

        self.conv1 = nn.Sequential(
            Conv(self.band_num, channels[0]),
            Conv(channels[0], channels[0]))

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv(channels[0], channels[1]),
            Conv(channels[1], channels[1]))

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv(channels[1], channels[2]),
            Conv(channels[2], channels[2]),
            Conv(channels[2], channels[2]))

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv(channels[2], channels[3]),
            Conv(channels[3], channels[3]),
            Conv(channels[3], channels[3]))

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Conv(channels[3], channels[4]),
            Conv(channels[4], channels[4]),
            Conv(channels[4], channels[4]))

        self.skblock4 = GMFR2(channels[3]*3, channels[3]*3)
        self.skblock3 = GMFR2(channels[2]*3, channels[2]*3)
        self.skblock2 = GMFR1(channels[1]*3, channels[1]*3)
        self.skblock1 = GMFR1(channels[0]*3, channels[0]*3)

        # 使用转置卷积层替换 MCIF 模块
        self.deconv5_4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=(2, 2), stride=(2, 2))
        self.deconv4_3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=(2, 2), stride=(2, 2))
        self.deconv3_2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=(2, 2), stride=(2, 2))
        self.deconv2_1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=(2, 2), stride=(2, 2))

        self.conv6 = nn.Sequential(Conv(channels[3]*3, channels[3]), Conv(channels[3], channels[3]))
        self.conv7 = nn.Sequential(Conv(channels[2]*3, channels[2]), Conv(channels[2], channels[2]))
        self.conv8 = nn.Sequential(Conv(channels[1]*3, channels[1]), Conv(channels[1], channels[1]))
        self.conv9 = nn.Sequential(Conv(channels[0]*3, channels[0]), Conv(channels[0], channels[0]))

        self.conv10 = nn.Conv2d(channels[0], self.class_num, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        deconv5 = conv5  # 移除 MCIF 模块，直接使用 conv5
        deconv5 = self.deconv5_4(deconv5)
        conv5 = self.deconv5_4(conv5)

        conv6 = torch.cat((deconv5, conv4, conv5), 1)
        conv6 = self.skblock4(conv6)
        deconv4 = self.conv6(conv6)

        deconv4 = self.deconv4_3(deconv4)
        conv4 = self.deconv4_3(conv4)

        conv7 = torch.cat((deconv4, conv4, conv3), 1)
        conv7 = self.skblock3(conv7)
        deconv3 = self.conv7(conv7)

        deconv3 = self.deconv3_2(deconv3)
        conv3 = self.deconv3_2(conv3)

        conv8 = torch.cat((deconv3, conv3, conv2), 1)
        conv8 = self.skblock2(conv8)
        deconv2 = self.conv8(conv8)

        deconv2 = self.deconv2_1(deconv2)
        conv2 = self.deconv2_1(conv2)

        conv9 = torch.cat((deconv2, conv2, conv1), 1)
        conv9 = self.skblock1(conv9)
        deconv1 = self.conv9(conv9)

        output = F.sigmoid(self.conv10(deconv1))

        return output

if __name__ == '__main__':
    x = torch.rand(2, 4, 128, 128)
    model = UNet_GMFR(band_num=4, class_num=2)
    # 计算模型的总参数量
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)


