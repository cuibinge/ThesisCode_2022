import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 边界注意力门控模块（EAS）
class EAS(nn.Module):
    def __init__(self, in_channels):
        super(EAS, self).__init__()
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.sa = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.abs(edge_x) + torch.abs(edge_y)
        ca_weight = self.ca(x)
        sa_weight = torch.sigmoid(self.sa(x))
        x = x * ca_weight + x * sa_weight
        x = self.conv(x + edge)
        return x



class MixBlock(nn.Module):
    """特征混合模块，实现双时相特征交互"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels*2, in_channels, kernel_size=3,
            padding=1, groups=in_channels  # 分组卷积实现通道交互
        )
        self.norm = nn.InstanceNorm2d(in_channels)
        # self.act = nn.PReLU()

    def forward(self, feat_T1, feat_T2):
        x = torch.cat([feat_T1, feat_T2], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        # x = self.act(x)
        return x


class ChangeAttentionBlock(nn.Module):
    """变化注意力生成模块，基于余弦相似度"""

    def __init__(self):
        super().__init__()

    def forward(self, feat_T1, feat_T2):
        # 展平特征并L2归一化
        B, C, H, W = feat_T1.shape
        Vi1 = F.normalize(feat_T1.flatten(2).permute(0, 2, 1), p=2, dim=-1)  # [B, HW, C]
        Vi2 = F.normalize(feat_T2.flatten(2).permute(0, 2, 1), p=2, dim=-1)

        # 计算余弦相似度并调整到[0,1]
        sim = torch.sum(Vi1 * Vi2, dim=-1)  # [B, HW]
        attn = (sim + 1) / 2  # 映射到0-1范围
        return attn.view(B, 1, H, W)  # [B, 1, H, W]


class MAB(nn.Module):
    """改进版多级特征聚合模块（三输入版本）"""

    def __init__(self, in_channels, skip_channels):
        super().__init__()
        # # 新增上采样组件
        # self.skip_upsample = nn.Sequential(
        #     nn.ConvTranspose2d(skip_channels, skip_channels,
        #                       kernel_size=2, stride=2),  # 尺寸扩大一倍
        #     nn.InstanceNorm2d(skip_channels),
        #     nn.PReLU()
        # )
        # 特征混合分支
        self.mix_block = nn.Sequential(
            MixBlock(in_channels),
            MixBlock(in_channels)  # 增加第二个混合块
        )

        # 跳跃连接处理（接收上层MAB输出）
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)

        # 注意力融合模块
        self.attn_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.PReLU()
        )

        # 上采样组件
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
            nn.InstanceNorm2d(in_channels // 2),
            nn.PReLU()
        )

    def forward(self, feat_T1, feat_T2, upper_feat):
        """
        输入参数：
        - feat_T1: 时相1的特征图 [B, C, H, W]
        - feat_T2: 时相2的特征图 [B, C, H, W]
        - upper_feat: 上层MAB的输出特征 [B, C_skip, H, W]
        """
        # 阶段1：双时相特征混合
        # upper_feat = self.upsample(upper_feat)  # 新增上采样
        mixed = self.mix_block[0](feat_T1, feat_T2)

        # 阶段2：跨层特征融合
        skip = self.skip_conv(upper_feat)
        fused = torch.cat([mixed, skip], dim=1)
        fused = self.attn_fusion(fused)

        # 阶段3：注意力加权输出
        attn_map = ChangeAttentionBlock()(feat_T1, feat_T2)
        weighted = fused * attn_map

        # 阶段4：级联混合与上采样
        refined = self.mix_block[1](weighted, skip)
        return self.upsample(refined)

# class MAB(nn.Module):
#     """多级特征聚合模块"""
#
#     def __init__(self, in_channels, skip_channels=None):
#         super().__init__()
#         self.mix = MixBlock(in_channels)
#         self.attn = ChangeAttentionBlock()
#
#         # 跳跃连接处理
#         self.skip_conv = nn.Conv2d(skip_channels, in_channels, 1) if skip_channels else None
#         self.fuse_conv = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
#
#         # 上采样组件
#         self.upsample = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
#             nn.InstanceNorm2d(in_channels // 2),
#             nn.PReLU()
#         )
#
#     def forward(self, feat_T1, feat_T2, skip=None):
#         # 特征混合与注意力生成
#         mixed = self.mix(feat_T1, feat_T2)
#         attn_map = self.attn(feat_T1, feat_T2)
#
#         # 跳跃连接融合
#         if skip is not None:
#             skip = self.skip_conv(skip)
#             mixed = torch.cat([mixed, skip], dim=1)
#             mixed = self.fuse_conv(mixed)
#
#         # 注意力加权与上采样
#         out = mixed * attn_map
#         return self.upsample(out)

# # 改进 Decoder（多层上采样）
# class Decoder(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(Decoder, self).__init__()
#         self.upsample1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
#         self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64
#         self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 64x64 -> 128x128
#         self.upsample4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256
#         self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)  # 最终通道调整
#
#     def forward(self, x):
#         x = F.relu(self.upsample1(x))
#         x = F.relu(self.upsample2(x))
#         x = F.relu(self.upsample3(x))
#         x = F.relu(self.upsample4(x))
#         x = self.final_conv(x)
#         return x

class EACDNet(nn.Module):
    def __init__(self, num_classes=2,in_channels=3, enc_channels=[64, 128, 256, 512]):
        super(EACDNet, self).__init__()
        # # 阶段1：256x256
        # self.stage1 = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # 新stage1（输出128x128）
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),  # 关键修改：添加stride=2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 阶段2：128x128
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 阶段3：64x64
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # 阶段4：32x32
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.Mix = MixBlock(512)

        self.eas1 = EAS(64)  # 第一阶段特征增强
        self.eas2 = EAS(128)  # 第二阶段特征增强
        self.eas3 = EAS(256)  # 第三阶段特征增强
        self.eas4 = EAS(512)  # 第四阶段特征增强

        channels = enc_channels
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(enc_channels[3], enc_channels[3] // 2, 2, stride=2),
            nn.InstanceNorm2d(in_channels // 2),
            nn.PReLU()
        )

        # 各层跳跃连接通道数配置（根据上采样后的输出维度）

        self.mab2 = MAB(in_channels=channels[2], skip_channels=channels[2]//2)
        self.mab3 = MAB(in_channels=channels[1], skip_channels=channels[1]//2)
        self.mab4 = MAB(in_channels=channels[0], skip_channels=channels[0]//2)  # 最顶层无跳跃输入
        #
        # self.mab2 = MAB(512)
        # self.mab3 = MAB(256)
        # self.mab4 = MAB(128)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)  # 最终通道调整

    def forward(self, t1, t2):
        # 提取不同层级的特征
        feat_t1_1 = self.stage1(t1)
        feat_t1_2 = self.stage2(feat_t1_1)
        feat_t1_3 = self.stage3(feat_t1_2)
        feat_t1_4 = self.stage4(feat_t1_3)

        feat_t2_1 = self.stage1(t2)
        feat_t2_2 = self.stage2(feat_t2_1)
        feat_t2_3 = self.stage3(feat_t2_2)
        feat_t2_4 = self.stage4(feat_t2_3)

        # 经过边界注意力增强
        feat_t1_1 = self.eas1(feat_t1_1)
        feat_t1_2 = self.eas2(feat_t1_2)
        feat_t1_3 = self.eas3(feat_t1_3)
        feat_t1_4 = self.eas4(feat_t1_4)

        feat_t2_1 = self.eas1(feat_t2_1)
        feat_t2_2 = self.eas2(feat_t2_2)
        feat_t2_3 = self.eas3(feat_t2_3)
        feat_t2_4 = self.eas4(feat_t2_4)

        fused1 = self.Mix(feat_t1_4, feat_t2_4)
        fused1 = self.upsample(fused1)
        # 进行特征融合
        fused2 = self.mab2(feat_t1_3, feat_t2_3, fused1)
        fused3 = self.mab3(feat_t1_2, feat_t2_2, fused2)
        fused4 = self.mab4(feat_t1_1, feat_t2_1, fused3)

        out = self.final_conv(fused4)

        return out


# 测试代码
if __name__ == "__main__":
    model = EACDNet()
    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)
    output = model(t1, t2)
    print(output.shape)  # 期望输出: (1, 2, 256, 256)
