# -*- coding: UTF-8 -*-
"""
@File    ：SemanticSimilarity.py
@Author  ：ChenRuipeng of SDUST
@Date    ：2024/3/31 14:33
@Tloml   : "永远都像初次见你那样，使我心荡漾"
"""
import torch
import torch.nn as nn


class SemanticSimilarity(nn.Module):
    def __init__(self):
        super(SemanticSimilarity, self).__init__()

    def forward(self, features, superpixel_indices):
        b, c, h, w = features.shape
        N = 32

        # 将所有张量移动到相同的设备上
        device = features.device
        superpixel_indices = superpixel_indices.to(device)

        # 初始化超像素特征张量
        superpixel_features = torch.zeros(b, N, c, device=device)

        # 计算每个超像素的特征均值
        for i in range(b):
            for sp in range(N):
                mask = (superpixel_indices[i] == sp)
                if mask.any():
                    selected_features = features[i, :, mask]
                    if selected_features.numel() > 0:
                        superpixel_features[i, sp] = selected_features.mean(dim=1)

        # 使用向量化操作计算相似性矩阵
        superpixel_features = superpixel_features.view(b, N, 1, c)
        diff = superpixel_features - superpixel_features.transpose(1, 2)
        similarity_matrices = 1 - 0.5 * torch.norm(diff, dim=3) ** 2

        return superpixel_features.view(b, N, c), similarity_matrices
