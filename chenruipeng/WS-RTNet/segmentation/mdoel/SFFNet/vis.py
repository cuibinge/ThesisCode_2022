import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import timm
import imageio

from MDAF import MDAF
from FMS import FMS

# ------------------ 基础模块定义 ------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

# ------------------ WF 模块 ------------------
class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

# ------------------ SFFNet 模型定义 ------------------
class SFFNet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.5,
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=False,
                 window_size=8,
                 num_classes=2,
                 use_aux_loss=True):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.backbone = timm.create_model(
            model_name=backbone_name,
            features_only=True,
            pretrained=pretrained,
            output_stride=32,
            out_indices=(0, 1, 2, 3)
        )

        self.conv2 = ConvBN(192, decode_channels, kernel_size=1)
        self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(768, decode_channels, kernel_size=1)

        self.MDAF_L = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.MDAF_H = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.fuseFeature = FMS(in_ch=3 * decode_channels, out_ch=decode_channels, num_heads=8, window_size=window_size)
        self.WF1 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.WF2 = WF(in_channels=decode_channels, decode_channels=decode_channels)

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1)  # 此层用于 Grad-CAM hook
        )
        self.down = Conv(in_channels=3 * decode_channels, out_channels=decode_channels, kernel_size=1)

    def forward(self, x, imagename=None):
        b = x.size()[0]
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        res1h, res1w = res1.size()[-2:]

        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        # 这里我们直接使用原始连接后的特征作为中间层（用于 Grad-CAM 计算）
        middleres = torch.cat([res2, res3, res4], dim=1)
        fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(middleres, imagename)
        global_feature = self.MDAF_L(fusefeature_L, glb)
        global_feature = F.interpolate(global_feature, size=(res1h, res1w), mode='bicubic', align_corners=False)
        local_feature = self.MDAF_H(fusefeature_H, local)
        local_feature = F.interpolate(local_feature, size=(res1h, res1w), mode='bicubic', align_corners=False)
        deep_feat = self.WF1(global_feature, local_feature)
        middleres_down = self.down(middleres)
        deep_feature = F.interpolate(deep_feat, size=(res1h, res1w), mode='bicubic', align_corners=False)
        deep_feature = middleres_down + deep_feature
        deep_feature_for_seg = self.WF2(deep_feature, res1)
        seg_out = self.segmentation_head(deep_feature_for_seg)
        out = F.interpolate(seg_out, size=(h, w), mode='bilinear', align_corners=False)
        return out

def forward_with_features(model, x, imagename=None):
    """
    修改后的 forward，返回四个中间特征（用于 Grad-CAM）以及最终输出：
      1. middle_feature: 中间层特征（直接返回计算后未插值的 middleres）
      2. local_feature: 高频与局部融合特征
      3. global_feature: 低频与全局融合特征
      4. deep_feature: 深层特征（在 WF1 输出之后、WF2 之前）
      5. out: 最终分割结果
    """
    b = x.size()[0]
    h, w = x.size()[-2:]
    res1, res2, res3, res4 = model.backbone(x)
    res1h, res1w = res1.size()[-2:]

    res2 = model.conv2(res2)
    res3 = model.conv3(res3)
    res4 = model.conv4(res4)
    res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
    res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
    res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
    # middleres 用于 Grad-CAM 计算中间层特征
    middleres = torch.cat([res2, res3, res4], dim=1)
    # 这里直接返回 middleres 作为 middle_feature
    middle_feature = middleres  
    fusefeature_L, fusefeature_H, glb, local = model.fuseFeature(middleres, imagename)
    global_feature = model.MDAF_L(fusefeature_L, glb)
    global_feature = F.interpolate(global_feature, size=(res1h, res1w), mode='bicubic', align_corners=False)
    local_feature = model.MDAF_H(fusefeature_H, local)
    local_feature = F.interpolate(local_feature, size=(res1h, res1w), mode='bicubic', align_corners=False)
    deep_feat = model.WF1(global_feature, local_feature)
    middleres_down = model.down(middleres)
    deep_feature = F.interpolate(deep_feat, size=(res1h, res1w), mode='bicubic', align_corners=False)
    deep_feature = middleres_down + deep_feature
    deep_feature_for_seg = model.WF2(deep_feature, res1)
    seg_out = model.segmentation_head(deep_feature_for_seg)
    out = F.interpolate(seg_out, size=(h, w), mode='bilinear', align_corners=False)
    return middle_feature, local_feature, global_feature, deep_feature, out

# ------------------ 修改后的 Grad-CAM 计算函数 ------------------
def compute_gradcam(feature, score, input_size):
    """
    根据中间特征 feature 及目标 score 计算 Grad-CAM 热力图
      feature: 形状 [1, C, H, W]
      score: 标量目标输出（例如，对某类别输出所有像素均值）
      input_size: 上采样目标尺寸 (H_in, W_in)
    返回归一化热力图，形状为 [1, 1, H_in, W_in]
    """
    # 使用 torch.autograd.grad 显式计算梯度
    grads = torch.autograd.grad(score, feature, retain_graph=True)[0]  # [1, C, H, W]
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * feature).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
    cam_min = cam.view(cam.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
    cam_max = cam.view(cam.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam

# ------------------ 叠加热力图函数 ------------------
def overlay_heatmap_on_image(input_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    input_image: 原始图像，numpy 数组，形状为 (H, W, 3)，值范围 [0, 255]
    heatmap: 归一化热力图，numpy 数组，形状为 (H, W)，值范围 [0, 1]
    返回叠加后的图像
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    overlayed = cv2.addWeighted(input_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed

# ------------------ 图像加载函数 ------------------
def load_image(image_path):
    """
    使用 imageio 读取图像，并模仿训练预处理：
      1. 读取图像
      2. 转换为 numpy 数组并转换为 float
      3. 转换为 torch tensor，并调整通道顺序为 (C, H, W)
    """
    image = imageio.imread(image_path)
    image = np.array(image).astype(float)
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image.float()

# ------------------ 主函数 ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./vis_res", exist_ok=True)

    # 1. 初始化模型并加载权重
    model = SFFNet(
        decode_channels=96,
        dropout=0.5,
        backbone_name="convnext_tiny.in12k_ft_in1k_384",
        pretrained=False,
        window_size=8,
        num_classes=2,
        use_aux_loss=True
    ).to(device)
    checkpoint_path = "../../save_model_sffnetdnmask/epcho_99_loss_0.01107914.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. 加载测试图像（请修改为实际路径）
    test_image_path = "../../data/test/img/4.png"
    input_image = load_image(test_image_path)  # [C, H, W]
    input_tensor = input_image.unsqueeze(0).to(device)  # [1, C, H, W]

    # 3. 前向推理，获取四个中间特征及最终输出
    middle_feat, local_feat, global_feat, deep_feat, out = forward_with_features(model, input_tensor)

    # 4. 定义目标类别（例如二分类中取类别 1），计算目标分数（对该类别所有像素求均值）
    target_class = 1
    score = out[:, target_class, :, :].mean()
    model.zero_grad()
    score.backward(retain_graph=True)

    input_size = (input_tensor.shape[2], input_tensor.shape[3])
    cam_middle = compute_gradcam(middle_feat, score, input_size)
    cam_local  = compute_gradcam(local_feat,  score, input_size)
    cam_global = compute_gradcam(global_feat, score, input_size)
    cam_deep   = compute_gradcam(deep_feat,   score, input_size)

    # 转换为 numpy 数组
    cam_middle_np = cam_middle[0, 0].detach().cpu().numpy()
    cam_local_np  = cam_local[0, 0].detach().cpu().numpy()
    cam_global_np = cam_global[0, 0].detach().cpu().numpy()
    cam_deep_np   = cam_deep[0, 0].detach().cpu().numpy()

    # 5. 读取原始图像用于叠加（转换为 BGR 格式）
    orig_image = imageio.imread(test_image_path)
    if orig_image.ndim == 2:
        orig_image = cv2.cvtColor(orig_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        orig_image = cv2.cvtColor(orig_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 6. 对四个 Grad-CAM 热力图叠加到原图并保存
    overlay_middle = overlay_heatmap_on_image(orig_image, cam_middle_np, alpha=0.5)
    overlay_local  = overlay_heatmap_on_image(orig_image, cam_local_np, alpha=0.5)
    overlay_global = overlay_heatmap_on_image(orig_image, cam_global_np, alpha=0.5)
    overlay_deep   = overlay_heatmap_on_image(orig_image, cam_deep_np, alpha=0.5)

    cv2.imwrite(os.path.join("./vis_res", "GradCAM_Middle_Feature.png"), overlay_middle)
    cv2.imwrite(os.path.join("./vis_res", "GradCAM_Local_Feature.png"), overlay_local)
    cv2.imwrite(os.path.join("./vis_res", "GradCAM_Global_Feature.png"), overlay_global)
    cv2.imwrite(os.path.join("./vis_res", "GradCAM_Deep_Feature.png"), overlay_deep)
    print("四个中间特征的 Grad-CAM 可视化结果已保存到 ./vis_res 文件夹。")

if __name__ == "__main__":
    main()
