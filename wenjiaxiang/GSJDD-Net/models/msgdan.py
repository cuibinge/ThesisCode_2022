import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.ops import DeformConv2d
import numpy as np
from torch_geometric.nn import GCNConv
# 在训练循环中添加
from torch.cuda.amp import autocast

class GraphConvBlock(nn.Module):
    """图卷积交互模块"""
    def __init__(self, channels):
        super().__init__()
        self.gcn1 = GCNConv(channels, channels)
        self.gcn2 = GCNConv(channels, channels)
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        x = self.act(self.gcn1(x, edge_index))
        return self.gcn2(x, edge_index)


class DeformConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, groups=4):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size ** 2 * groups,
            kernel_size=3,
            padding=dilation,  # 根据dilation调整padding
            dilation=dilation
        )
        self.conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,  # 添加dilation参数
            groups=groups
        )
        self.norm = nn.InstanceNorm2d(out_channels)  # 保持2D归一化
        self.act = nn.GELU()

    def forward(self, x):
        B, C, D, H, W = x.size()
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        offset = self.offset_conv(x_2d)

        # 关键修改点：在reshape前进行归一化
        out = self.conv(x_2d, offset)  # [B*D, out_C, H, W]
        out = self.norm(out)  # 正确应用4D输入
        out = self.act(out)

        # 恢复5D形状
        out = out.view(B, D, -1, H, W).permute(0, 2, 1, 3, 4)  # [B, out_C, D, H, W]
        return out


class MSDAM(nn.Module):
    def __init__(self, in_channels, scales=[(64, 1), (128, 2)], deform_groups=4):
        super().__init__()
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool3d((1, 2, 2), padding=(0, 1, 1)) if s == 64 else nn.Identity(),
                DeformConv3D(in_channels, in_channels * 2, dilation=dilation, groups=deform_groups)
            ) for s, dilation in scales
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(len(scales), 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.GELU()
        )

    def forward(self, hsi_feat, lidar_3d):
        B, C, D, H, W = hsi_feat.size()
        aligned_features = []


        for conv in self.scale_convs:
            # 特征提取
            h = conv(hsi_feat)  # [B, 2C, D', H', W']
            l = conv(lidar_3d)

            # 统一空间尺寸
            h = F.adaptive_avg_pool3d(h, (D, H, W))  # 强制对齐到输入尺寸
            l = F.adaptive_avg_pool3d(l, (D, H, W))

            # 计算通道相似度
            h_flat = h.view(B, -1, H * W)  # [B, 2C*D', H*W]
            l_flat = l.view(B, -1, H * W)
            similarity = F.cosine_similarity(h_flat, l_flat, dim=1)  # [B, H*W]
            similarity_map = similarity.view(B, 1, H, W)  # [B, 1, H, W]

            aligned_features.append(similarity_map)

        # 融合多尺度相似度
        fused = self.fusion(torch.cat(aligned_features, dim=1))  # [B, len(scales), H, W] → [B, 128, H, W]
        return fused


class GeometryGuidedAttention(nn.Module):
    """增强版几何引导光谱注意力"""

    def __init__(self, hsi_dim, lidar_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(hsi_dim, hsi_dim // 8, 1)
        self.key_conv = nn.Sequential(
            nn.Conv2d(1, hsi_dim // 8, 1),
            nn.ELU()
        )
        self.spec_mlp = nn.Sequential(
            nn.Linear(hsi_dim, hsi_dim // 2),
            nn.GELU()
        )
        self.energy_conv = nn.Conv2d(hsi_dim // 8 + hsi_dim // 8 + hsi_dim // 2, 1, 1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, hsi_feat, lidar_elev):
        B, C, H, W = hsi_feat.size()
        # 光谱交互
        spec_feat = self.spec_mlp(hsi_feat.mean(dim=(2, 3)))  # [B, C/2]
        spec_feat = spec_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)
        # 注意力计算
        Q = self.query_conv(hsi_feat)  # [B, C/8, H, W]
        K = torch.cat([self.key_conv(lidar_elev), spec_feat], dim=1)

        energy = self.energy_conv(torch.cat([Q, K], dim=1))
        attention = torch.sigmoid(energy)

        return hsi_feat + self.gamma * attention * hsi_feat


class AdaptiveProgressiveFusion(nn.Module):
    """两层自适应渐进式融合"""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(4 * channels, channels // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.fuse_conv_1 = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
        self.fuse_conv_2 = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
        self.gcn = GraphConvBlock(channels)
        if torch.cuda.is_available():
            self.gcn = self.gcn.cuda()
        self.lidar_adjust = nn.Conv2d(channels, 2 * channels, 1)
        # 添加一个卷积层用于将128维特征降到64维
        self.downsample = nn.Conv2d(128, channels, kernel_size=1)

    def build_graph(self, feat):
        B, C, H, W = feat.size()
        # 确保coord在正确的设备上
        coord = torch.stack(torch.meshgrid(
            torch.arange(H, device=feat.device),  # 添加device参数
            torch.arange(W, device=feat.device),  # 添加device参数
            indexing='ij'
        )).float()  # 不需要.to(feat.device)，因为已经指定了device
        nodes = coord.view(2, -1).t()  # [N, 2]
        dist = torch.cdist(nodes, nodes)
        edge_index = dist.topk(k=4, largest=False).indices

        num_nodes = H * W
        # 生成source_nodes时指定设备
        source_nodes = torch.arange(num_nodes, device=feat.device).unsqueeze(1).expand(-1, 4).flatten()
        target_nodes = edge_index.flatten()

        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
        return edge_index

    def forward(self, features):
        assert len(features) == 2, "特征对的数量必须为2"
        (hsi_1, lidar_1), (hsi_2, lidar_2) = features

        # 第一阶段：直接拼接
        fused_feat_1 = self.fuse_conv_1(torch.cat([hsi_1, lidar_1], dim=1))
        target_size = fused_feat_1.shape[-2:]
        # 第二阶段：门控融合与GCN特征融合
        lidar_2 = self.lidar_adjust(lidar_2)
        gate = self.gate(torch.cat([hsi_2, lidar_2], dim=1))
        fused_feat_gate = gate * hsi_2 + (1 - gate) * lidar_2

        B, C, H, W = hsi_2.size()
        edge_index = self.build_graph(hsi_2)
        # 将hsi_2的特征维度从128降到64
        hsi_2 = self.downsample(hsi_2)
        node_feat = hsi_2.view(B, self.gcn.gcn1.in_channels, -1).permute(0, 2, 1)
        # 确保输入到GCN的特征维度正确
        assert node_feat.size(-1) == self.gcn.gcn1.in_channels, f"输入到GCN的特征维度应为 {self.gcn.gcn1.in_channels}，但实际为 {node_feat.size(-1)}"
        node_feat = self.gcn(node_feat, edge_index)
        gcn_feat = node_feat.permute(0, 2, 1).view(B, self.gcn.gcn1.in_channels, H, W)

        fused_feat_2 = self.fuse_conv_2(torch.cat([fused_feat_gate, gcn_feat], dim=1))

        # 尺寸对齐
        if fused_feat_2.shape[-2:] != target_size:
            fused_feat_2 = F.interpolate(fused_feat_2, target_size, mode='bilinear')

        return fused_feat_1 + fused_feat_2


class MSGSAN(nn.Module):
    """修正维度后的完整实现"""
    def __init__(self, num_classes=3, deploy=False):
        super().__init__()
        self.deploy = deploy

        # HSI编码器
        self.hsi_conv = nn.Sequential(
            nn.Conv3d(1, 64, (8, 3, 3), stride=(8, 1, 1), padding=(0, 1, 1)),
            nn.GELU()
        )
        # self.hsi_conv = nn.Conv3d(1, 64, (16, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1))

        self.msdam = MSDAM(64, scales=[(64, 1), (128, 2)])  # 修正 scales 参数

        # LiDAR编码器
        self.lidar_encoder = nn.Sequential(
            self.SobelEdge(),
            nn.Conv2d(1, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.GELU()
        )

        # 特征交互模块
        self.ggsa = GeometryGuidedAttention(hsi_dim=128, lidar_dim=64)
        self.apf = AdaptiveProgressiveFusion(channels=64)

        # 动态路由分类头
        self.router = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 3, 1),
            nn.Softmax(dim=1)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1, groups=64),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128, num_classes, 1)
            ) for _ in range(3)
        ])

    class SobelEdge(nn.Module):
        """维度安全的边缘检测"""

        def __init__(self):
            super().__init__()
            # 调整卷积核为四维 [out_channels, in_channels, H, W]
            self.register_buffer('kernel_x', torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float() / 8.0)
            self.register_buffer('kernel_y', torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float() / 8.0)
            self.gaussian = nn.Conv2d(1, 1, 3, padding=1, bias=False)
            nn.init.constant_(self.gaussian.weight, 1 / 16.0)
            self.gaussian.weight.data[0, 0, 1, 1] = 4 / 16.0  # 确保高斯核中心值正确

        def forward(self, x):
            elev = self.gaussian(x[:, 0:1])
            grad_x = F.conv2d(elev, self.kernel_x, padding=1)
            grad_y = F.conv2d(elev, self.kernel_y, padding=1)
            edge = torch.clamp(grad_x ** 2 + grad_y ** 2, 0, 1)
            return torch.cat([edge, x[:, 1:]], dim=1)

    def forward(self, hsi, lidar):
        # HSI编码
        with autocast():
        # print(lidar.shape)
            hsi = hsi.unsqueeze(1)
            hsi_feat = self.hsi_conv(hsi)
        # LiDAR编码
            lidar_feat = self.lidar_encoder(lidar)
            lidar_feat = F.adaptive_avg_pool2d(lidar_feat, hsi_feat.shape[-2:])
            # 维度对齐
            lidar_3d = lidar_feat.unsqueeze(2)
            lidar_3d = lidar_3d.expand(-1, -1, hsi_feat.size(2), -1, -1)
            # 多尺度对齐
            aligned_feat = self.msdam(hsi_feat, lidar_3d)
        # 几何引导注意力
            attended_hsi = self.ggsa(aligned_feat, lidar[:, 0:1])
        # 渐进式融合
            features = [
                (hsi_feat.mean(dim=2), lidar_feat),
                (attended_hsi, lidar_feat)
            ]
            fused_feat = self.apf(features)
        # 动态路由分类
            weights = self.router(fused_feat)  # [B, 3, 1, 1]
            expert_outputs = [e(fused_feat) for e in self.experts]  # 每个 expert 的输出是 [B, num_classes, 1, 1]
            expert_outputs = [e.squeeze(-1).squeeze(-1) for e in expert_outputs]  # [B, num_classes]
            weights = weights.squeeze(-1).squeeze(-1)  # [B, 3]
            output = torch.sum(weights.unsqueeze(-1) * torch.stack(expert_outputs, dim=1), dim=1)  # [B, num_classes]
            print(time.time())
        return output



if __name__ == "__main__":
    model = MSGSAN(num_classes=3, deploy=True).cuda()
    lidar = torch.randn([2,1,13,13]).cuda()
    hsi = torch.randn([2,32,13,13]).cuda()
    # oup = model(hsi)
    oup = model(hsi, lidar)
    print(oup.shape)