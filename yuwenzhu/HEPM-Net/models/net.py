import torch
import torch.nn as nn
from .dual_stream import DualStream
from .moe import HeterogeneousExpert
from .prototype import Router, PrototypeMemory


class MoEProtoNet(nn.Module):
    def __init__(self, spectral_bands, num_classes):
        super().__init__()
        # 双流特征提取
        self.dual_stream = DualStream(spectral_bands)

        # 异构专家
        self.experts = nn.ModuleDict({
            "spatial": HeterogeneousExpert("spatial", in_channels=16),
            "spectral": HeterogeneousExpert("spectral", in_channels=64),
            "spatio_spectral": HeterogeneousExpert("spatio_spectral", in_channels=281)
        })

        # 路由门
        self.router = Router(num_experts=3, feat_dim=64)

        # 原型记忆库
        self.prototype_memory = PrototypeMemory(num_classes, feat_dim=64)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, labels=None):
        # 双流特征提取
        spatial_feat, spectral_feat = self.dual_stream(x)

        # 准备各专家输入
        expert_inputs = {
            "spatial": spatial_feat,
            "spectral": spectral_feat,
            "spatio_spectral": self._prepare_spatio_spectral(spatial_feat, spectral_feat)
        }

        # 专家特征提取
        expert_outputs = []
        for name, expert in self.experts.items():
            input_feat = expert_inputs[name]
            _, deep_feature = expert(input_feat)
            expert_outputs.append(deep_feature)

        # 路由融合
        fused_feature, gate_weights = self.router(expert_outputs)

        # 原型更新（训练阶段）
        if self.training and labels is not None:
            self.prototype_memory.update(fused_feature.detach(), labels)

        # 分类
        logits = self.classifier(fused_feature)

        return {
            "logits": logits,
            "gate_weights": gate_weights,
            "prototypes": self.prototype_memory.get_prototypes()
        }

    def _prepare_spatio_spectral(self, spatial, spectral):
        """ 维度转换说明：
        输入：
            spatial: [B,16,5,5] → 展平为[B,16*5*5=400]
            spectral: [B,64,64] → 展平为[B,64*64=4096]
        合并：400+4096=4496 → 重塑为[B,281,16]（不是[B,16,281]！）
        """
        # 维度验证
        assert spatial.dim() == 4, f"空间特征应为4D，实际得到{spatial.shape}"
        assert spectral.dim() == 3, f"光谱特征应为3D，实际得到{spectral.shape}"

        # 展平处理
        spatial_flat = spatial.flatten(start_dim=1)  # [B,400]
        spectral_flat = spectral.flatten(start_dim=1)  # [B,4096]

        # 合并并重塑
        combined = torch.cat([spatial_flat, spectral_flat], dim=1)  # [B,4496]
        assert combined.shape[1] == 4496, f"合并特征维度错误，应为4496，实际得到{combined.shape[1]}"

        # 重塑为[B,281,16] 而非 [B,16,281]
        return combined.view(combined.size(0), 281, 16)  # 关键修复点！


if __name__ == '__main__':
    model = MoEProtoNet(spectral_bands=30, num_classes=10)
    dummy_input = torch.randn(12, 30, 13, 13)

    # 显示各专家输入维度
    spatial, spectral = model.dual_stream(dummy_input)
    spatio_spectral = model._prepare_spatio_spectral(spatial, spectral)
    print("\n=== 专家输入维度 ===")
    print(f"Spatial专家输入: {spatial.shape}")  # [12,16,5,5]
    print(f"Spectral专家输入: {spectral.shape}")  # [12,64,64]
    print(f"Spatio-spectral专家输入: {spatio_spectral.shape}")  # [12,281,16]

    # 前向传播验证
    outputs = model(dummy_input)
    print("\n=== 输出维度 ===")
    print(f"Logits shape: {outputs['logits'].shape}")  # [12,10]