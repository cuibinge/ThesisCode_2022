import torch
import torch.nn as nn
import torch.nn.functional as F

class RotationInvariantModule(nn.Module):
    def __init__(self, in_channels=128, num_angles=8):
        super(RotationInvariantModule, self).__init__()
        self.in_channels = in_channels
        self.num_angles = num_angles  # 旋转角度数量（如 0°, 45°, 90° 等）

        # 空间注意力模块，输入通道数应为 1（因为均值压缩后通道数为 1）
        self.spatial_conv = nn.Conv2d(1, 1, 1)
        self.spatial_bn = nn.BatchNorm2d(1)

        # 输出调整
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)

    def rotate_feature(self, x, angle):
        """对特征图进行旋转操作"""
        # x: [batch_size, channels, num_classes, H, W]
        rotated_x = torch.rot90(x, k=angle, dims=[-2, -1])  # 在 H, W 维度上旋转
        return rotated_x

    def forward(self, x):
        # 输入形状: [batch_size, channels, num_classes, H, W]
        batch_size, channels, num_classes, H, W = x.size()

        # 初始化旋转等变特征列表
        rotated_features = []

        # 对特征图进行多角度旋转
        for angle_idx in range(self.num_angles):
            angle = angle_idx * (360 // self.num_angles)  # 均匀分布的角度
            rotated_x = self.rotate_feature(x, angle_idx)  # 旋转特征图
            rotated_features.append(rotated_x)

        # 将旋转特征堆叠，形状为 [batch_size, channels, num_classes, num_angles, H, W]
        rotated_features = torch.stack(rotated_features, dim=3)

        # 对旋转维度进行池化，提取旋转不变特征
        # 使用最大池化，形状变为 [batch_size, channels, num_classes, H, W]
        invariant_features = torch.max(rotated_features, dim=3)[0]

        # 对旋转不变特征进行空间注意力加权
        # 首先沿通道维度压缩，得到空间权重
        spatial_weight = invariant_features.mean(dim=1, keepdim=True)  # [batch_size, 1, num_classes, H, W]

        # 重塑张量以适配 2D 卷积：将 [batch_size, 1, num_classes, H, W] 转为 [batch_size * num_classes, 1, H, W]
        spatial_weight = spatial_weight.view(batch_size * num_classes, 1, H, W)
        spatial_weight = torch.sigmoid(self.spatial_bn(self.spatial_conv(spatial_weight)))  # [batch_size * num_classes, 1, H, W]

        # 恢复形状为 [batch_size, 1, num_classes, H, W]
        spatial_weight = spatial_weight.view(batch_size, 1, num_classes, H, W)

        # 加权
        spatial_weighted_features = invariant_features * spatial_weight  # [batch_size, channels, num_classes, H, W]

        # 输出调整：将 [batch_size, channels, num_classes, H, W] 转为 [batch_size * num_classes, channels, H, W] 以适配 2D 卷积
        spatial_weighted_features = spatial_weighted_features.view(batch_size * num_classes, channels, H, W)
        out = self.out_conv(spatial_weighted_features)  # [batch_size * num_classes, channels, H, W]

        # 恢复形状为 [batch_size, channels, num_classes, H, W]
        out = out.view(batch_size, channels, num_classes, H, W)

        # 通过残差连接融合
        out = out + x

        return out

class ClassRelationModule(nn.Module):
    def __init__(self, in_channels=128, reduction=16):
        super(ClassRelationModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 类别特征嵌入
        self.class_embed = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU()
        )

        # 类别关系建模（自注意力）
        self.query_conv = nn.Conv2d(in_channels // reduction, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels // reduction, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels // reduction, in_channels // reduction, 1)

        # 输出调整
        self.out_conv = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        # 输入形状: [batch_size, channels, num_classes, H, W]
        batch_size, channels, num_classes, H, W = x.size()

        # 对每个类别的特征图进行嵌入
        # 形状: [batch_size * num_classes, channels, H, W]
        x_reshaped = x.view(batch_size * num_classes, channels, H, W)
        class_embed = self.class_embed(x_reshaped)  # [batch_size * num_classes, channels // reduction, H, W]

        # 全局池化，得到类别描述符
        class_desc = F.adaptive_avg_pool2d(class_embed, 1).view(batch_size, num_classes, channels // self.reduction, 1, 1)
        class_desc = class_desc.squeeze(-1).squeeze(-1)  # [batch_size, num_classes, channels // reduction]

        # 自注意力机制，建模类别关系
        # Query, Key, Value
        query = self.query_conv(class_embed).view(batch_size, num_classes, channels // self.reduction, H * W)
        key = self.key_conv(class_embed).view(batch_size, num_classes, channels // self.reduction, H * W)
        value = self.value_conv(class_embed).view(batch_size, num_classes, channels // self.reduction, H * W)

        # 计算类别关系矩阵
        # [batch_size, num_classes, num_classes]
        relation_matrix = torch.matmul(query.transpose(-2, -1), key) / (H * W) ** 0.5
        relation_matrix = F.softmax(relation_matrix, dim=-1)

        # 加权类别特征
        # [batch_size, num_classes, channels // reduction, H * W]
        weighted_class = torch.matmul(relation_matrix, value.transpose(-2, -1))
        weighted_class = weighted_class.transpose(-2, -1).contiguous()
        weighted_class = weighted_class.view(batch_size, num_classes, channels // self.reduction, H, W)

        # 调整维度顺序为 [batch_size, channels // reduction, num_classes, H, W]
        weighted_class = weighted_class.permute(0, 2, 1, 3, 4).contiguous()

        # 输出调整：将 [batch_size, channels // reduction, num_classes, H, W] 转为 [batch_size * num_classes, channels // reduction, H, W] 以适配 2D 卷积
        weighted_class = weighted_class.view(batch_size * num_classes, channels // self.reduction, H, W)
        out = self.out_conv(weighted_class)  # [batch_size * num_classes, channels, H, W]

        # 恢复形状为 [batch_size, channels, num_classes, H, W]
        out = out.view(batch_size, channels, num_classes, H, W)

        # 通过残差连接融合
        out = out + x

        return out

class FeatureOptimizer(nn.Module):
    def __init__(self, in_channels=128, num_angles=8, reduction=16):
        super(FeatureOptimizer, self).__init__()
        self.rotation_module = RotationInvariantModule(in_channels, num_angles)
        self.class_module = ClassRelationModule(in_channels, reduction)

    def forward(self, x):
        # 先进行空间维度的旋转不变性优化
        x = self.rotation_module(x)
        # 再进行类别维度的类别关系建模
        x = self.class_module(x)
        return x

# 测试代码
if __name__ == "__main__":
    batch_size, channels, num_classes, H, W = 5, 128, 6, 24, 24
    x = torch.randn(batch_size, channels, num_classes, H, W)
    model = FeatureOptimizer(channels)
    out = model(x)
    print(out.shape)  # 应输出 [5, 128, 6, 24, 24]