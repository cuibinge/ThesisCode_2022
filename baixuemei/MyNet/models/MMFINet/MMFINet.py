import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class GAM_RU(nn.Module):
    def __init__(self, channels, h, w):
        super().__init__()
        self.conv11 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv21 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv12 = nn.Conv2d(2, 1, kernel_size=1)
        self.conv22 = nn.Conv2d(2, 1, kernel_size=1)
        self.max_pool1 = nn.MaxPool2d((1, channels))
        self.avg_pool1 = nn.AvgPool2d((1, channels))
        self.max_pool2 = nn.MaxPool2d((h, w))
        self.avg_pool2 = nn.AvgPool2d((h, w))

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv21(x)
        x1 = x1.permute([0, 2, 3, 1])
        x11 = self.max_pool1(x1)
        x12 = self.avg_pool1(x1)
        x21 = self.max_pool2(x2)
        x22 = self.avg_pool2(x2)
        x13 = torch.concat([x11, x12], dim=3)
        x23 = torch.concat([x21, x22], dim=2)
        x13 = x13.permute([0, 3, 1, 2])
        x23 = x23.permute([0, 2, 3, 1])
        x14 = self.conv12(x13)
        x24 = self.conv22(x23)
        x24 = x24.permute([0, 3, 1, 2])
        x15 = torch.sigmoid_(x14)
        x25 = torch.sigmoid_(x24)
        x3 = x15 * x25 * x
        y = x3 + x
        return y


class Upload(nn.Module):
    def __init__(self, in_channels, skip_num=1):
        super().__init__()
        out_channels = in_channels // 2
        self.skip = SkipBlock(out_channels, skip_num)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBP(out_channels, out_channels)
        )

    def forward(self, x_list):
        x_list[0] = self.block(x_list[0])
        return self.skip(x_list)


class SkipBlock(nn.Module):
    def __init__(self, in_channels, skip_num):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels * skip_num, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x_list):
        x = torch.concat(x_list, dim=1)
        return self.skip(x)


class ResNet50(nn.Module):
    output_size = 2048

    def __init__(self, in_channels, pretrained=False):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

        original_num_channels = self.conv1.in_channels
        if in_channels != original_num_channels:
            self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        b1 = self.layer1(b)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out


class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, x2, x4, last=False):
        # # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
        # x_1_2_fusion = self.fusion_1x2(x1, x2)
        # x_1_4_fusion = self.fusion_1x4(x1, x4)
        # x_fused = x_1_2_fusion + x_1_4_fusion
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused


class MAF_Unet(nn.Module):
    def __init__(self, in_channels, num_classes=1000, include_top=True):
        super(MAF_Unet, self).__init__()

        base_dims = 64
        base_h = 64
        base_w = 64
        hidden_dim = base_dims

        self.include_top = include_top

        self.backbone = ResNet50(in_channels, True)
        self.AFFs = nn.ModuleList([
            MSAA(hidden_dim * 7, base_dims * 4),
            MSAA(hidden_dim * 7, base_dims * 8),
            MSAA(hidden_dim * 7, base_dims * 16),
        ])
        self.transfer = nn.ModuleList(
            [
                nn.Conv2d(base_dims * 4, hidden_dim, 1, bias=False),
                nn.Conv2d(base_dims * 8, hidden_dim * 2, 1, bias=False),
                nn.Conv2d(base_dims * 16, hidden_dim * 4, 1, bias=False),
            ]
        )
        self.U11 = Upload(base_dims * 32)
        self.U12 = Upload(base_dims * 16, 2)
        self.U13 = Upload(base_dims * 8, 2)
        self.U14 = Upload(base_dims * 4, 2)
        self.U15 = Upload(base_dims * 2, 2)

        self.U21 = Upload(base_dims * 16)
        self.GAM_RU21 = GAM_RU(base_dims * 8, base_h // 8, base_w // 8)
        self.U22 = Upload(base_dims * 8, 2)
        self.GAM_RU22 = GAM_RU(base_dims * 4, base_h // 4, base_w // 4)
        self.U23 = Upload(base_dims * 4, 2)
        self.GAM_RU23 = GAM_RU(base_dims * 2, base_h // 2, base_w // 2)
        self.U24 = Upload(base_dims * 2, 2)
        self.GAM_RU24 = GAM_RU(base_dims, base_h, base_w)

        self.U31 = Upload(base_dims * 8)
        self.GAM_RU31 = GAM_RU(base_dims * 4, base_h // 4, base_w // 4)
        self.U32 = Upload(base_dims * 4)
        self.GAM_RU32 = GAM_RU(base_dims * 2, base_h // 2, base_w // 2)
        self.U33 = Upload(base_dims * 2)
        self.GAM_RU33 = GAM_RU(base_dims, base_h, base_w)

        self.super_vision = SkipBlock(base_dims, 3)
        self.MLP = nn.Sequential(
            nn.Linear(base_h * base_w * num_classes, base_h),
            nn.BatchNorm1d(base_h),
            nn.ReLU(),
            nn.Linear(base_h, num_classes),
            nn.Softmax(dim=1)
        )
        if include_top:
            self.conv_top = nn.Sequential(
                nn.Conv2d(base_dims, num_classes, kernel_size=1, stride=1),
                nn.BatchNorm2d(num_classes)
            )
            self.softmax = nn.Softmax(1)

    def forward_resnet(self, x):
        res1, res2, res3, res4 = self.backbone(x)
        x_downsample = [res1, res2, res3]
        x = res4
        return x, x_downsample

    def forward_downfeatures(self, x_downsample):
        x_downsample_2 = x_downsample
        x_downsample = []
        for idx, feat in enumerate(x_downsample_2):
            feat = self.transfer[idx](feat)
            x_downsample.append(feat)

        x_down_4_3 = F.interpolate(x_downsample[2], scale_factor=2.0, mode="bilinear", align_corners=True)
        x_down_2_3 = F.interpolate(x_downsample[0], scale_factor=0.5, mode="bilinear", align_corners=True)

        x_down_2_4 = F.interpolate(x_downsample[0], scale_factor=0.25, mode="bilinear", align_corners=True)
        x_down_3_4 = F.interpolate(x_downsample[1], scale_factor=0.5, mode="bilinear", align_corners=True)

        x_down_3 = self.AFFs[1](x_downsample[1], x_down_2_3, x_down_4_3)
        x_down_4 = self.AFFs[2](x_downsample[2], x_down_3_4, x_down_2_4)

        return [x_down_3, x_down_4]

    def forward(self, x):
        x1, x_downsample = self.forward_resnet(x)
        x3, x2 = self.forward_downfeatures(x_downsample)

        u31 = self.U31([x3])  # [1, 64*expansion, 64, 64]
        u31 = self.GAM_RU31(u31)  # [1, 64*expansion, 64, 64]
        u32 = self.U32([u31])  # [1, 32*expansion, 128, 128]
        u32 = self.GAM_RU32(u32)  # [1, 32*expansion, 128, 128]
        u33 = self.U33([u32])
        u33 = self.GAM_RU33(u33)

        u21 = self.U21([x2])  # [1, 128*expansion, 32, 32]
        u21 = self.GAM_RU21(u21)  # [1, 128*expansion, 32, 32]
        u22 = self.U22([u21, u31])  # [1, 64*expansion, 64, 64]
        u22 = self.GAM_RU22(u22)  # [1, 64*expansion, 64, 64]
        u23 = self.U23([u22, u32])  # [1, 32*expansion, 128, 128]
        u23 = self.GAM_RU23(u23)  # [1, 32*expansion, 128, 128]
        u24 = self.U24([u23, u33])
        u24 = self.GAM_RU24(u24)

        u11 = self.U11([x1])  # [1, 256*expansion, 16, 16]
        u12 = self.U12([u11, u21])  # [1, 128*expansion, 32, 32]
        u13 = self.U13([u12, u22])  # [1, 64*expansion, 64, 64]
        u14 = self.U14([u13, u23])  # [1, 32*expansion, 128, 128]
        u15 = self.U15([u14, u24])

        y = self.super_vision([u15, u24, u33])

        if self.include_top:
            y = self.conv_top(y)  # [1, 17, 256, 256]
            y = self.softmax(y)  # [1, 17, 256, 256]
            u15 = self.conv_top(u15)  # [1, 17, 256, 256]
            u15 = self.softmax(u15)  # [1, 17, 256, 256]
            u24 = self.conv_top(u24)  # [1, 17, 256, 256]
            u24 = self.softmax(u24)  # [1, 17, 256, 256]
            u33 = self.conv_top(u33)  # [1, 17, 256, 256]
            u33 = self.softmax(u33)  # [1, 17, 256, 256]

        b, c, h, w = y.shape
        y = y.view(b, c * h * w)
        y = self.MLP(y)
        if not self.training:
            return y
        y1 = u15.view(b, c * h * w)
        y1 = self.MLP(y1)
        y2 = u24.view(b, c * h * w)
        y2 = self.MLP(y2)
        y3 = u33.view(b, c * h * w)
        y3 = self.MLP(y3)
        return y, y1, y2, y3


class CBR(nn.Module):
    def __init__(self, kernel_size, dilation, in_channels, out_channels):
        super().__init__()
        if dilation == 1:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  dilation=dilation, padding=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CBP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class ASPP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.CBR_1_1_1 = CBR(1, 1, in_channels=channels, out_channels=channels)
        self.CBR_3_6 = CBR(3, 6, in_channels=channels, out_channels=channels)
        self.CBR_3_12 = CBR(3, 12, in_channels=channels, out_channels=channels)
        self.CBR_3_18 = CBR(3, 18, in_channels=channels, out_channels=channels)
        self.attention = nn.Sequential(
            nn.MaxPool2d(kernel_size=1),
            nn.Conv2d(kernel_size=1, in_channels=channels, out_channels=channels)
        )
        self.CBR_1_1_2 = CBR(1, 1, in_channels=channels * 4, out_channels=channels)
        self.CBP = CBP(in_channels=channels, out_channels=channels)

    def forward(self, x):
        x1 = self.CBR_1_1_1(x)
        x2 = self.CBR_3_6(x)
        x3 = self.CBR_3_12(x)
        x4 = self.CBR_3_18(x)
        x5 = self.attention(x)
        x = torch.concat([x1, x2, x3, x4], dim=1)
        x = self.CBR_1_1_2(x)
        x = x * x5
        x = self.CBP(x)
        return x


def get_model(num_classes):
    return MAF_Unet(30, num_classes=num_classes)


if __name__ == '__main__':
    x = torch.rand([1, 30, 64, 64])
    net = get_model(9)
    y = net(x)
    print(y[0].shape)
