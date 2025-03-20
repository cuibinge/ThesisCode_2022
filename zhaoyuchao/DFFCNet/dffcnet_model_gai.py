import torch
import torch.nn as nn
from config import swin_tiny_patch4_224 as swin
import torch.nn.functional as F
import math
from DFConv import DeformConv2d
import timm

# cSE模块：这是一个通道注意力模块，用于对输入特征图的通道进行重新加权。

class cSE(nn.Module):  # noqa: N801
    """
    The channel-wise SE (Squeeze and Excitation) block from the
    `Squeeze-and-Excitation Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x

# sSE:这是一个空间注意力模块，用于对输入特征图的空间维度进行重新加权。

class sSE(nn.Module):  # noqa: N801
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x

# scSE:这是一个结合了通道和空间注意力的模块。

class scSE(nn.Module):  # noqa: N801
    """
    The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
    block from the `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.
    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    Shape:
    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels: The number of channels
                in the feature map of the input.
            r: The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x

# ChannelPool:这个模块将输入特征图在通道维度上进行最大池化和平均池化，并将结果拼接在一起。

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# DWconv:这是一个深度可分离卷积模块，包含一个深度卷积和一个点卷积。

class DWconv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1):
        super(DWconv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

# BiFFM 模块融合了 Transformer 和 CNN 分支的特征，通过自适应池化、深度可分离卷积、可变形卷积和 scSE 块来实现特征融合。

class BiFFM(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFFM, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.residual = ATR(ch_1 + ch_2, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dw1 = DWconv(ch_2, ch_2 // r_2, padding=8, dilation=8)
        self.dw2 = DWconv(ch_2, ch_2 // r_2, padding=8, dilation=8)
        self.dw3 = DWconv(ch_2 // r_2, ch_2, padding=8, dilation=8)
        self.df_conv = DeformConv2d(ch_1, ch_int)
        self.scse = scSE(ch_int)
        self.haam = HAAM(ch_int,ch_int)   #HAAM类（混合自适应注意力模块）
        self.hwab = HWAB(n_feat=ch_int,o_feat=ch_int)  #半小波注意力块（HWAB）
        self.dw4 = DWconv(ch_int, ch_out, padding=8, dilation=8)

    def forward(self, g, x):
        ##Transformer_branch
        # print(g.shape)
        y1 = self.avg_pool(x)
        y1 = self.dw1(y1)
        y2 = self.max_pool(x)
        y2 = self.dw2(y2)
        y = self.relu(y1 + y2)
        y = self.dw3(y)
        y = self.sigmoid(y) * x

        ##CNN_branch
        c1 = self.df_conv(g)
        #         print("c1_1:",c1.shape)   #[2, 256, 16, 16]
        #         添加DF模块，频率动态滤波
        #         block = DynamicFilter(256, size=16) # size==H,W
        _, _, H, W = x.shape  # 获取输入的尺寸
        # 使用动态尺寸初始化 DynamicFilter
        dynamic_filter = DynamicFilter(self.ch_int, size=H)  # 或者 W，取决于具体需求
        c1_df = c1.permute(0, 2, 3, 1)  # Convert from BCHW to BHWC for DynamicFilter
        #         print("c1_2:",c1.shape)   #[2, 16, 16, 256]
        c1_df = dynamic_filter(c1_df)  # Apply DynamicFilter
        c1_df = c1_df.permute(0, 3, 1, 2)  # Convert back from BHWC to BCHW
        #         print("DF:", c1.shape)  #[2, 256, 16, 16]
        #         c1 = self.dynamic(c1)
        #         print("DF:",c1.shape)
        # c1 = self.scse(c1)
        # c1_scse  = self.scse(c1)
        c1_haam = self.haam(c1)

        # 使用半小波注意力，HWAB模块
        c1_hwab = self.hwab(c1)

        # 合并两个分支
        c1 = c1_df + c1_hwab  # 或者使用 c1 = c1_df * c1_haam


        # # 合并两个分支
        # c1 = c1_df + c1_haam  # 或者使用 c1 = c1_df * c1_haam

        # # 合并两个分支
        # c1 = torch.cat((c1_df, c1_scse), dim=1)  # 沿着通道维度拼接

        c1 = self.dw4(c1)
        c2 = self.sigmoid(c1) * g

        fuse = self.residual(torch.cat([y, c2], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

 # 主网络模型，综合了EfficientNet和Swin Transformer，通过一系列上采样和融合操作生成最终输出。

class DFFCNet(nn.Module):
    # 初始化权重,该方法初始化模型的权重，使用 Kaiming 正态分布初始化卷积层权重，使用标准正态分布初始化线性层权重。
    def __init__(self, num_classes=1, drop_rate=0.4, normal_init=True, pretrained=False):
        super(DFFCNet, self).__init__()

        self.efficienet = timm.create_model('efficientnet_b3')
        if pretrained:
            self.efficienet.load_state_dict(torch.load('./pretained/efficientnet_b3_ra2-cf984f9c.pth'))

        self.transformer = swin(pretrained=pretrained)

        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)

        ###加的
        self.up3 = Up(64, 32)

        self.final_x = nn.Sequential(
            Conv(232, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_3 = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )

        self.up_c = BiFFM(ch_1=232, ch_2=768, r_2=2, ch_int=256, ch_out=232, drop_rate=drop_rate / 2)

        self.up_c_1_1 = BiFFM(ch_1=136, ch_2=384, r_2=2, ch_int=128, ch_out=136,
                              drop_rate=drop_rate / 2)
        self.up_c_1_2 = Up(in_ch1=232, out_ch=128, in_ch2=136, attn=True)

        self.up_c_2_1 = BiFFM(ch_1=48, ch_2=192, r_2=1, ch_int=64, ch_out=48, drop_rate=drop_rate / 2)
        self.up_c_2_2 = Up(128, 64, 48, attn=True)

        ###
        self.up_c_3_1 = BiFFM(ch_1=32, ch_2=96, r_2=1, ch_int=32, ch_out=32, drop_rate=drop_rate / 2)
        self.up_c_3_2 = Up(64, 32, 32, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    # 前向传播,描述输入图片如何通过各个子模块处理并生成输出。
    def forward(self, imgs):
        # transformer path Swin Transformer 主干网络提取特征图，逐层处理输入图像。
        x_b = self.transformer(imgs)
        x_b_1 = x_b[0]
        x_b_1 = torch.transpose(x_b_1, 1,
                                2)  ##maybe need to take out the first tensor,i.e., x_b_1[0], if x_b_1 is tuple, due to the torch version issue.
        x_b_1 = x_b_1.view(x_b_1.shape[0], -1, 128, 128)
        x_b_1 = self.drop(x_b_1)

        x_b_2 = x_b[1]
        x_b_2 = torch.transpose(x_b_2, 1,
                                2)  ## ##maybe need to take out the first tensor,i.e., x_b_2[0], if x_b_1 is tuple, due to the torch version issue.
        x_b_2 = x_b_2.view(x_b_2.shape[0], -1, 64, 64)
        x_b_2 = self.drop(x_b_2)

        x_b_3 = x_b[2]
        x_b_3 = torch.transpose(x_b_3, 1,
                                2)  ## ##maybe need to take out the first tensor,i.e., x_b_3[0], if x_b_1 is tuple, due to the torch version issue.
        x_b_3 = x_b_3.view(x_b_3.shape[0], -1, 32, 32)
        x_b_3 = self.drop(x_b_3)

        x_b_4 = x_b[3]
        x_b_4 = torch.transpose(x_b_4, 1,
                                2)  ## ##maybe need to take out the first tensor,i.e., x_b_4[0], if x_b_1 is tuple, due to the torch version issue.
        x_b_4 = x_b_4.view(x_b_4.shape[0], -1, 16, 16)
        x_b_4 = self.drop(x_b_4)

        # CNN path EfficientNet 主干网络提取特征图，逐层处理输入图像。
        ####effinetb3
        x_u128 = self.efficienet.conv_stem(imgs)
        x_u128 = self.efficienet.bn1(x_u128)
        x_u128 = self.efficienet.act1(x_u128)
        x_u128 = self.efficienet.blocks[0](x_u128)
        x_u64 = self.efficienet.blocks[1](x_u128)

        x_u_2 = self.efficienet.blocks[2](x_u64)
        x_u_2 = self.drop(x_u_2)

        x_u_3 = self.efficienet.blocks[3](x_u_2)
        x_u_3 = self.drop(x_u_3)

        x_u_3 = self.efficienet.blocks[4](x_u_3)
        x_u_3 = self.drop(x_u_3)

        x_u = self.efficienet.blocks[5](x_u_3)
        x_u = self.drop(x_u)

        # joint path
        x_c = self.up_c(x_u, x_b_4)

        x_c_1_1 = self.up_c_1_1(x_u_3, x_b_3)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)

        ###
        x_c_3_1 = self.up_c_3_1(x_u64, x_b_1)
        x_c_3 = self.up_c_3_2(x_c_2, x_c_3_1)

        #
        map_x = F.interpolate(self.final_x(x_c), scale_factor=32, mode='bilinear')
        map_1 = F.interpolate(self.final_1(x_c_1), scale_factor=16, mode='bilinear')
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=8, mode='bilinear')
        map_3 = F.interpolate(self.final_3(x_c_3), scale_factor=4, mode='bilinear')
        return map_x, map_1, map_2, map_3

    # 初始化网络模型中的权重参数，确保模型在训练开始前的权重设置合理。
    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.up3.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final_3.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)
        self.up_c_3_1.apply(init_weights)
        self.up_c_3_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# 用于特征的上采样，通常在卷积神经网络中用于放大特征图，常见于分割网络中，如UNet，用于逐步恢复到原始图像的分辨率。
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ATR(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = ATG(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

# 功能：
# channel_shuffle 函数用于通道混洗。这在某些神经网络（尤其是轻量级网络，如 ShuffleNet）中很常见，通过将通道分组并混洗，可以促进跨通道的信息流动，提高模型的表现力。
#
# 步骤：
#
# 获取输入张量 x 的大小（批次大小、通道数、高度、宽度）。
# 计算每组的通道数。
# 重塑张量以便分组。
# 交换组和通道维度。
# 将张量重塑回原来的形状，但通道已混洗。
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


####功能：
# ATG 类是一个简单的卷积块，结合了通道混洗功能。它首先对输入张量进行 1x1 卷积，然后对输出进行通道混洗。
#
# 步骤：
#
# 初始化时，定义一个 1x1 卷积层。
# 在 forward 方法中，将输入张量通过卷积层，然后调用 channel_shuffle 函数进行通道混洗。
class ATG(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ATG, self).__init__()
        self.W_g = nn.Sequential(
            DWconv(F_g, F_int, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            DWconv(F_l, F_int, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            DWconv(F_int, 1, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi1 = self.psi(psi) * x
        return psi1

# 功能：
# DoubleConv 类实现了一个双层卷积块，常用于 U-Net 等结构中。每个卷积层后接 Batch Normalization 和 ReLU 激活函数。
#
# 步骤：
#
# 初始化时，定义一个包含两个 3x3 卷积层的序列，每个卷积层后接 Batch Normalization 和 ReLU。
# 在 forward 方法中，将输入张量依次通过这个卷积块。
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))

        return out
# 功能：
# ATR 类实现了一个带有 Batch Normalization 和 ReLU 激活函数的卷积块。它包含一个 1x1 卷积、一个深度卷积（group 卷积）和另一个 1x1 卷积。
#
# 步骤：
#
# 初始化时，定义三个卷积层：一个 1x1 卷积、一个深度卷积和另一个 1x1 卷积，并添加 Batch Normalization 和 ReLU。
# 在 forward 方法中，依次通过这些卷积层，
class ATR(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(ATR, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        self.se = cSE(output_dim)

    def forward(self, x):
        return self.se(self.conv_block(x) + self.conv_skip(x))

# Conv 类实现了一个可选带有 Batch Normalization 和 ReLU 激活函数的卷积块。这个类提供了创建卷积层的灵活性，可以选择是否添加 BN 和 ReLU。
#
# 步骤：
#
# 初始化时，定义一个卷积层。可以选择性地添加 Batch Normalization 和 ReLU 激活函数。
# 在 forward 方法中，依次通过卷积层、Batch Normalization（如果有）、ReLU（如果有）。
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)




        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




#
#
# 本文网络模型 DFFCNet 具体用到了以下功能：
#
# 卷积神经网络 (CNN) 和变换器 (Transformer) 结合：
# 使用 EfficientNet 作为 CNN 特征提取器。
# 使用 Swin Transformer 作为变换器路径，提取特征。
# 注意力机制：
# 使用了 Squeeze-and-Excitation (SE) 机制，包括:
# cSE (Channel-wise Squeeze and Excitation)
# sSE (Spatial Squeeze and Channel Excitation)
# scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
# 使用了 BiFFM (Bilateral Feature Fusion Module) 模块进行特征融合。
# 分层上采样模块：
# 通过 Up 模块实现特征图的上采样。
# 残差连接和卷积层：
# 使用残差连接和不同类型的卷积层，包括深度可分离卷积 (DWconv) 和可变形卷积 (DeformConv2d)，以增强特征提取能力。
# 池化操作：
# 使用自适应平均池化 (AdaptiveAvgPool2d) 和自适应最大池化 (AdaptiveMaxPool2d) 操作进行特征图的降维。
# 权重初始化：
# 使用 Kaiming Normal 初始化方式初始化网络权重。
# 丢弃层 (Dropout)：
# 通过 Dropout2d 层进行正则化，以防止过拟合。
# 以下是代码中具体实现这些功能的模块：
#
# cSE, sSE, scSE：实现了不同类型的 SE 块，用于不同层级的注意力机制。
# DWconv, DeformConv2d：实现了深度可分离卷积和可变形卷积，用于增强特征提取。
# BiFFM：实现了双向特征融合模块，用于融合 CNN 和 Transformer 的特征。
# Up：实现了上采样模块，用于特征图的分辨率提升。
# AdaptiveAvgPool2d, AdaptiveMaxPool2d：用于特征图的自适应池化。
# 这些功能共同构成了 DFFCNet 网络模型，使其能够在处理图像数据时有效地融合 CNN 和 Transformer 提取的特征，并通过多种注意力机制和卷积操作增强特征提取和融合能力。