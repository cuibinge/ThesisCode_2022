import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import timm
import imageio

# -------------------- 请确保下面这些模块在你的项目中可以正确导入 --------------------
# 例如，你可能有 custom_modules.py 文件包含了以下模块：
# ConvBN, MDAF, FMS, WF, ConvBNReLU, Conv
from custom_modules import ConvBN, MDAF, FMS, WF, ConvBNReLU, Conv
# -------------------------------------------------------------------------------------------

# ==================== 定义 SFFNet 模型 ====================
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
        # 使用 timm 创建 backbone
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
            Conv(decode_channels, num_classes, kernel_size=1)
        )
        self.down = Conv(in_channels=3 * decode_channels, out_channels=decode_channels, kernel_size=1)

    def forward(self, x, imagename=None):
        """
        原始 forward，仅返回最终分割结果
        """
        b = x.size()[0]
        h, w = x.size()[-2:]

        # 提取 backbone 特征
        res1, res2, res3, res4 = self.backbone(x)
        res1h, res1w = res1.size()[-2:]

        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        # 中间层特征
        middleres = torch.cat([res2, res3, res4], dim=1)

        fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(middleres, imagename)
        glb = self.MDAF_L(fusefeature_L, glb)
        local = self.MDAF_H(fusefeature_H, local)
        res = self.WF1(glb, local)

        middleres_down = self.down(middleres)
        res = F.interpolate(res, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res = middleres_down + res
        res = self.WF2(res, res1)
        res = self.segmentation_head(res)

        out = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
        return out

def forward_with_features(model, x, imagename=None):
    """
    与 forward 几乎相同的函数，但返回中间关键特征：
    1. middleres：中间层特征
    2. glb：低频与全局融合层特征
    3. local：高频与局部融合层特征
    4. deep_feat：深层特征（WF1 的输出）
    5. out：最终分割输出
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

    # 1. 中间层特征
    middleres = torch.cat([res2, res3, res4], dim=1)

    # 分解得到低频/高频及全局/局部信息
    fusefeature_L, fusefeature_H, glb, local = model.fuseFeature(middleres, imagename)

    # 2. 低频与全局融合层特征
    glb = model.MDAF_L(fusefeature_L, glb)
    # 3. 高频与局部融合层特征
    local = model.MDAF_H(fusefeature_H, local)
    # 4. 深层特征
    deep_feat = model.WF1(glb, local)

    middleres_down = model.down(middleres)
    deep_feat_up = F.interpolate(deep_feat, size=(res1h, res1w), mode='bicubic', align_corners=False)
    deep_feat_up = middleres_down + deep_feat_up
    deep_feat_up = model.WF2(deep_feat_up, res1)
    seg_out = model.segmentation_head(deep_feat_up)

    out = F.interpolate(seg_out, size=(h, w), mode='bilinear', align_corners=False)

    return middleres, glb, local, deep_feat, out

# ------------------- 定义可视化函数，将结果保存至指定文件夹 -------------------
def visualize_feature_maps(feature_tensor, title="", max_channels=4):
    """
    可视化给定特征张量的前 max_channels 个通道，并保存图片到 ./vis_res 文件夹
    feature_tensor: [B, C, H, W]
    """
    feat = feature_tensor[0].detach().cpu().numpy()  # 取 batch 中第一个样本
    num_channels = feat.shape[0]
    num_plots = min(num_channels, max_channels)
    
    plt.figure(figsize=(4 * num_plots, 4))
    for i in range(num_plots):
        plt.subplot(1, num_plots, i + 1)
        channel_map = feat[i, :, :]
        channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min() + 1e-8)
        plt.imshow(channel_map, cmap='jet')
        plt.axis('off')
    plt.suptitle(title)
    # 构造安全的文件名
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    save_path = os.path.join("./vis_res", f"{safe_title}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_segmentation(seg_tensor, title="Final_Segmentation_Output"):
    """
    可视化最终分割结果（若为二分类，则取 argmax），并保存图片到 ./vis_res 文件夹
    seg_tensor: [B, num_classes, H, W]
    """
    # 对分割结果取 argmax
    pred = torch.argmax(seg_tensor, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    plt.figure(figsize=(6, 6))
    plt.imshow(pred, cmap='gray')
    plt.title(title)
    plt.axis('off')
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    save_path = os.path.join("./vis_res", f"{safe_title}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ------------------- 根据训练预处理代码写个加载图像的函数 -------------------
def load_image(image_path):
    """
    使用 imageio 读取图像，并模仿训练预处理：
      1. 读取图像
      2. 转换为 numpy 数组并转换为 float
      3. 转换为 torch tensor，并调整通道顺序为 (C, H, W)
    """
    image = imageio.imread(image_path)
    image = np.array(image)
    image = image.astype(float)
    # 如果图像没有归一化处理，根据需要可以在此处处理
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image.float()

# ------------------- 主函数 -------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("./vis_res", exist_ok=True)  # 创建保存结果的文件夹

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

    checkpoint_path = "/neSeg/save_model_sffnetdnmask/epcho_93_loss_0.01154287.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 根据实际情况调整加载方式
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. 加载测试图像（请修改为你实际的测试图像路径）
    test_image_path = "/neSeg/data/test/img/6.png"
    input_image = load_image(test_image_path)
    input_tensor = input_image.unsqueeze(0).to(device)  # 扩展 batch 维度

    # 3. 前向推理，获取中间层特征及最终分割输出
    with torch.no_grad():
        middleres, glb, local, deep_feat, final_out = forward_with_features(model, input_tensor)

    # 4. 保存各阶段特征的可视化结果到 ./vis_res 文件夹
    visualize_feature_maps(middleres, title="1_Middle_Features", max_channels=4)
    visualize_feature_maps(glb, title="2_Global_Low_Frequency_Features", max_channels=4)
    visualize_feature_maps(local, title="3_Local_High_Frequency_Features", max_channels=4)
    visualize_feature_maps(deep_feat, title="4_Deep_Features", max_channels=4)
    visualize_segmentation(final_out, title="Final_Segmentation_Output")

if __name__ == "__main__":
    main()
