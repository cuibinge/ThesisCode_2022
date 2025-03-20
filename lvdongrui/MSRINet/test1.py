import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassRelationModule(nn.Module):
    def __init__(self, in_channels=128, num_classes=6, reduction=16):
        super(ClassRelationModule, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
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

# 测试代码
if __name__ == "__main__":
    batch_size, channels, num_classes, H, W = 5, 128, 6, 24, 24
    x = torch.randn(batch_size, channels, num_classes, H, W)
    model = ClassRelationModule(channels, num_classes)
    out = model(x)
    print(out.shape)  # 应输出 [5, 128, 6, 24, 24]