import torch
import torch.nn.functional as F


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def compute_losses(preds_A, preds_B, labels_A, labels_B, changemap, changemap_gt):
    """
    计算语义分割损失和变化检测损失。

    参数：
    preds_A (torch.Tensor): 模型对A图的语义分割预测结果，形状为 (B, C, H, W)。
    preds_B (torch.Tensor): 模型对B图的语义分割预测结果，形状为 (B, C, H, W)。
    labels_A (torch.Tensor): A图的语义分割标签，形状为 (B, H, W)。
    labels_B (torch.Tensor): B图的语义分割标签，形状为 (B, H, W)。
    change_map_A (torch.Tensor): A图的变化检测标签，形状为 (B, H, W)。
    change_map_B (torch.Tensor): B图的变化检测标签，形状为 (B, H, W)。

    返回：
    tuple: (语义分割损失, 变化检测损失)
    """

    # 计算语义分割损失
    # 交叉熵损失需要标签是长整型
    labels_A = labels_A.long()
    labels_B = labels_B.long()
    labels_A = torch.squeeze(labels_A, dim=1)
    labels_B = torch.squeeze(labels_B, dim=1)


    # 计算A图的语义分割损失
    loss_seg_A = F.cross_entropy(preds_A, labels_A)

    # 计算B图的语义分割损失
    loss_seg_B = F.cross_entropy(preds_B, labels_B)

    # 总的语义分割损失
    loss_seg = (loss_seg_A + loss_seg_B) / 2

    # 计算变化检测损失
    # 变化检测通常是二进制交叉熵损失
    changemap_gt = changemap_gt.long()
    changemap_gt = torch.squeeze(changemap_gt, dim=1)

    # 计算变化检测损失
    loss_change = F.cross_entropy(changemap, changemap_gt)
    # print(loss_seg)
    # print(loss_change)
    total_loss = loss_change + loss_seg

    return total_loss
