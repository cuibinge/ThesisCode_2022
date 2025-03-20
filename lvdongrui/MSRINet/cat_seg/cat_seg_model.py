# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import FeaturePyramidNetwork
from einops import rearrange

@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024

        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer_indexes = [6, 3, 1] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
        
        # 1. 加载预训练的 ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = resnet
        self.resnet_layers_to_extract = ['layer1', 'layer2', 'layer3']  # 提取 C2、C3、C4 层
        self.resnet_features = {}

        # 注册钩子以提取 ResNet 的中间层特征
        def hook(name):
            def hook_fn(module, input, output):
                self.resnet_features[name] = output
            return hook_fn
        for name, module in resnet.named_modules():
            if name in self.resnet_layers_to_extract:
                module.register_forward_hook(hook(name))

        # 2. 定义 FPN 模块，用于优化 clip_features
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 768, 768, 768],  # ResNet 的 C2、C3、C4 和 CLIP 的 res4、res5、res6
            out_channels=512,  # 直接输出 512 通道，与 clip_features 匹配
        )

        # 3. 定义投影层，将 FPN 输出调整为与 clip_features 兼容
        self.fpn_projector = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # 4. 添加可学习权重参数 alpha
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始值为 0.1，表示 FPN 特征的初始贡献较小

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)  # [B, 577, 512]

        # 1. 生成原始 features 字典（保持不变）
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)  # [B, 512, 24, 24]
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)  # [B, 768, 24, 24]
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)  # [B, 768, 24, 24]
        res6 = rearrange(self.layers[2][1:, :, :], "(H W) B C -> B C H W", H=24)  # [B, 768, 24, 24]

        features = {
            'res6': self.upsample3(res6),  # [B, 64, 192, 192]
            'res5': self.upsample2(res5),  # [B, 128, 96, 96]
            'res4': self.upsample1(res4),  # [B, 256, 48, 48]
            'res3': res3,
        }

        # 调试：打印 features 形状
        # print("features shapes:")
        # for key, feat in features.items():
        #     print(f"{key}: {feat.shape}")

        # 2. 提取 ResNet 的多尺度特征
        self.resnet_features = {}
        _ = self.resnet(clip_images_resized)
        resnet_features = [self.resnet_features[layer] for layer in self.resnet_layers_to_extract]

        # 3. 特征对齐（调整分辨率以适配 FPN）
        target_resolutions = [24, 48, 96, 192]


        # print("Before interpolation:")
        # for i, feat in enumerate(resnet_features):
        #     print(f"resnet_features[{i}].shape: {feat.shape}")
        # for i, feat in enumerate(clip_features):
        #     print(f"clip_features[{i}].shape: {feat.shape}")

        resnet_aligned = []
        for i, feat in enumerate(resnet_features):
            feat = F.interpolate(feat, size=(target_resolutions[i+1], target_resolutions[i+1]), mode='bilinear', align_corners=False)
            resnet_aligned.append(feat)

        clip_aligned = []
        for feat, target_res in zip([res4, res5, res6], target_resolutions[1:]):
            feat = F.interpolate(feat, size=(target_res, target_res), mode='bilinear', align_corners=False)
            clip_aligned.append(feat)

        # print("\nAfter interpolation:")
        # for i, feat in enumerate(resnet_aligned):
        #     print(f"resnet_aligned[{i}].shape: {feat.shape}")
        # for i, feat in enumerate(clip_aligned):
        #     print(f"clip_aligned[{i}].shape: {feat.shape}")

        # 4. FPN 融合（用于优化 clip_features）
        all_features = resnet_aligned + clip_aligned  # [C2, C3, C4, res4, res5, res6]
        fpn_input = {f'feat{i}': feat for i, feat in enumerate(all_features)}
        fpn_features = self.fpn(fpn_input)

        # 调试：打印 FPN 输出形状
        # print("FPN features shapes:")
        # for key, feat in fpn_features.items():
        #     print(f"{key}: {feat.shape}")

        # 5. 使用 FPN 输出优化 clip_features（加权融合）
        fpn_enhance = fpn_features['feat0']  # [B, 512, 48, 48]
        fpn_enhance = F.interpolate(fpn_enhance, size=(24, 24), mode='bilinear', align_corners=False)  # [B, 512, 24, 24]
        fpn_enhance = self.fpn_projector(fpn_enhance)  # [B, 512, 24, 24]
        fpn_enhance = rearrange(fpn_enhance, "B C H W -> B (H W) C")  # [B, 576, 512]

        cls_token = clip_features[:, :1, :]  # [B, 1, 512]
        spatial_features = clip_features[:, 1:, :]  # [B, 576, 512]
        enhanced_spatial_features = spatial_features + self.alpha * fpn_enhance  # 加权融合
        optimized_clip_features = torch.cat([cls_token, enhanced_spatial_features], dim=1)  # [B, 577, 512]
        # print(f"optimized_clip_features shape: {optimized_clip_features.shape}")
        # print(f"alpha value: {self.alpha.item()}")  # 调试：打印 alpha 的值

        # 6. 输入到 sem_seg_head
        outputs = self.sem_seg_head(optimized_clip_features, features)
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results

    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images_resized = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False)
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)  # [B, 577, 512]
        
        # 1. 生成原始 features 字典
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res6 = rearrange(self.layers[2][1:, :, :], "(H W) B C -> B C H W", H=24)

        features = {
            'res6': self.upsample3(res6),  # [B, 64, 192, 192]
            'res5': self.upsample2(res5),  # [B, 128, 96, 96]
            'res4': self.upsample1(res4),  # [B, 256, 48, 48]
            'res3': res3,
        }

        # 2. 提取 ResNet 的多尺度特征
        self.resnet_features = {}
        _ = self.resnet(clip_images_resized)
        resnet_features = [self.resnet_features[layer] for layer in self.resnet_layers_to_extract]

        # 3. 特征对齐
        target_resolutions = [24, 48, 96, 192]

        resnet_aligned = []
        for i, feat in enumerate(resnet_features):
            feat = F.interpolate(feat, size=(target_resolutions[i+1], target_resolutions[i+1]), mode='bilinear', align_corners=False)
            resnet_aligned.append(feat)

        clip_aligned = []
        for feat, target_res in zip([res4, res5, res6], target_resolutions[1:]):
            feat = F.interpolate(feat, size=(target_res, target_res), mode='bilinear', align_corners=False)
            clip_aligned.append(feat)

        # 4. FPN 融合
        all_features = resnet_aligned + clip_aligned
        fpn_input = {f'feat{i}': feat for i, feat in enumerate(all_features)}
        fpn_features = self.fpn(fpn_input)

        # 5. 使用 FPN 输出优化 clip_features（加权融合）
        fpn_enhance = fpn_features['feat0']  # [B, 512, 48, 48]
        fpn_enhance = F.interpolate(fpn_enhance, size=(24, 24), mode='bilinear', align_corners=False)  # [B, 512, 24, 24]
        fpn_enhance = self.fpn_projector(fpn_enhance)  # [B, 512, 24, 24]
        fpn_enhance = rearrange(fpn_enhance, "B C H W -> B (H W) C")  # [B, 576, 512]

        cls_token = clip_features[:, :1, :]
        spatial_features = clip_features[:, 1:, :]
        enhanced_spatial_features = spatial_features + self.alpha * fpn_enhance  # 加权融合
        optimized_clip_features = torch.cat([cls_token, enhanced_spatial_features], dim=1)  # [B, 577, 512]

        # 6. 输入到 sem_seg_head
        outputs = self.sem_seg_head(optimized_clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]