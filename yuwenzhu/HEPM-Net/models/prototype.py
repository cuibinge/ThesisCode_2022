import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, num_experts, feat_dim):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(num_experts * feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, expert_features):
        # 输入: list of [B,64]
        concated = torch.cat(expert_features, dim=1)  # [B, 3*64]
        weights = self.gating(concated)  # [B,3]
        fused = torch.stack(expert_features, dim=1)  # [B,3,64]
        return torch.bmm(weights.unsqueeze(1), fused).squeeze(1), weights


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.99):
        super().__init__()
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.alpha = alpha
        self.feat_dim = feat_dim

    def update(self, features, labels):
        # 过滤无效标签
        valid_mask = (labels >= 0) & (labels < self.prototypes.size(0))
        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]

        # 逐类更新
        for cls_id in torch.unique(valid_labels):
            cls_mask = (valid_labels == cls_id)
            cls_feats = valid_features[cls_mask]
            if cls_feats.size(0) == 0:
                continue

            new_proto = cls_feats.mean(dim=0)
            self.prototypes[cls_id] = (
                    self.alpha * self.prototypes[cls_id] +
                    (1 - self.alpha) * new_proto
            )

    def get_prototypes(self):
        return self.prototypes