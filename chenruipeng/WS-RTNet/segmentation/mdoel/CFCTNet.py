import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

from .ViT import truncated_normal_, BasicBlock, PathchEmbed
from .modules import Bi_DirectionalDecoder, FeatureAggregationModule, DepthwiseConv2d


class CNN_Block123(nn.Module):
    def __init__(self, inchans, outchans):
        super().__init__()
        self.stage = nn.Sequential(
            DepthwiseConv2d(in_chans=inchans, out_chans=outchans, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
            DepthwiseConv2d(in_chans=outchans, out_chans=outchans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
        )
        self.conv1x1 = DepthwiseConv2d(in_chans=inchans, out_chans=outchans, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        stage = self.stage(x)
        max = self.maxpool(x)
        max = self.conv1x1(max)
        stage = stage + max
        return stage


class CNN_Block45(nn.Module):
    def __init__(self, inchans, outchans):
        super().__init__()

        self.stage = nn.Sequential(
            DepthwiseConv2d(in_chans=inchans, out_chans=outchans, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
            DepthwiseConv2d(in_chans=outchans, out_chans=outchans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
            DepthwiseConv2d(in_chans=outchans, out_chans=outchans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchans),
            nn.ReLU(),
        )
        self.conv1x1 = DepthwiseConv2d(in_chans=inchans, out_chans=outchans, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        stage = self.stage(x)
        max = self.maxpool(x)
        max = self.conv1x1(max)
        stage = stage + max
        return stage


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q.matmul(k.permute(0, 1, 3, 2))) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape(-1, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(BasicBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, epsilon=1e-6, sr_ratio=1):
        super().__init__(dim, num_heads)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(PathchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                         embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).permute(0, 2, 1)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class CTCFNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 epsilon=1e-6, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], class_dim=2):
        super().__init__()
        self.class_dim = class_dim
        self.depths = depths
        self.img_size = img_size
        self.interpolate_size = img_size // 4

        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2,
                                       in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2,
                                       in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2,
                                       in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(drop_rate)

        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(drop_rate)

        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(drop_rate)

        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, embed_dims[3]))
        self.pos_drop4 = nn.Dropout(drop_rate)

        dpr = np.linspace(0, drop_path_rate, sum(depths))
        cur = 0

        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, epsilon=epsilon,
                sr_ratio=sr_ratios[0]
            ) for i in range(depths[0])
        ])

        cur = cur + depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, epsilon=epsilon,
                sr_ratio=sr_ratios[1]
            ) for i in range(depths[1])
        ])

        cur = cur + depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, epsilon=epsilon,
                sr_ratio=sr_ratios[2]
            ) for i in range(depths[2])
        ])

        cur = cur + depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i], norm_layer=norm_layer, epsilon=epsilon,
                sr_ratio=sr_ratios[3]
            ) for i in range(depths[3])
        ])
        self.norm = norm_layer(embed_dims[3])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        if class_dim > 0:
            self.head = nn.Linear(embed_dims[3], class_dim)

        self.cnn_branch1 = CNN_Block123(inchans=in_chans, outchans=64)
        self.cnn_branch2 = CNN_Block123(inchans=64, outchans=128)
        self.cnn_branch3 = CNN_Block123(inchans=64, outchans=128)
        self.cnn_branch4 = CNN_Block45(inchans=128, outchans=256)
        self.cnn_branch5 = CNN_Block45(inchans=320, outchans=640)
        self.conv1x1_4 = nn.Conv2d(in_channels=512, out_channels=320, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)

        self.CTmerge2 = FeatureAggregationModule(cnn_chans=128, trans_chans=64)
        self.CTmerge3 = FeatureAggregationModule(cnn_chans=128, trans_chans=128)
        self.CTmerge4 = FeatureAggregationModule(cnn_chans=256, trans_chans=320)
        self.CTmerge5 = FeatureAggregationModule(cnn_chans=640, trans_chans=512)

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=640, out_channels=320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.detail_head = nn.Conv2d(in_channels=128, out_channels=self.class_dim, kernel_size=1)

        self.bi_directional_decoder = Bi_DirectionalDecoder(self.interpolate_size)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.class_dim, kernel_size=1)
        )

        truncated_normal_(self.pos_embed1)
        truncated_normal_(self.pos_embed2)
        truncated_normal_(self.pos_embed3)
        truncated_normal_(self.pos_embed4)
        truncated_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            truncated_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1)

    def reset_drop_path(self, drop_path_rate):
        dpr = np.linspace(0, drop_path_rate, sum(self.depths))
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur = cur + self.depths[0]
        for i in range(self.depth[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur = cur + self.depths[1]
        for i in range(self.depth[2]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur = cur + self.depths[2]
        for i in range(self.depth[3]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

    def forward(self, x):
        B = x.shape[0]
        c_s1 = self.cnn_branch1(x)
        c_s2 = self.cnn_branch2(c_s1)
        # Stage 1
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.CTmerge2(c_s2, x)
        m_s1 = x
        c_s3 = self.cnn_branch3(m_s1)
        # Stage 2
        x, (H, W) = self.patch_embed2(x)
        x = x + self.pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.CTmerge3(c_s3, x)
        m_s2 = x
        c_s4 = self.cnn_branch4(m_s2)
        # Stage 3
        x, (H, W) = self.patch_embed3(x)
        x = x + self.pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.CTmerge4(c_s4, x)
        m_s3 = x
        c_s5 = self.cnn_branch5(m_s3)
        # Stage 4
        x, (H, W) = self.patch_embed4(x)
        x = x + self.pos_embed4
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.CTmerge5(c_s5, x)
        m_s4 = x

        output = self.bi_directional_decoder(m_s1, m_s2, m_s3, m_s4)
        output = F.interpolate(output, size=self.img_size // 2, mode='bilinear', align_corners=True)
        output = self.head(output)
        return F.interpolate(output, size=self.img_size, mode='bilinear', align_corners=True), self.detail_head(m_s2)

        # if self.training:
        #     return F.interpolate(output, size=self.img_size, mode='bilinear', align_corners=True), \
        #         self.detail_head(m_s2)
        # else:
        #     return F.interpolate(output, size=self.img_size, mode='bilinear', align_corners=True)


if __name__ == '__main__':
    model = CTCFNet(img_size=256, in_chans=3, class_dim=2,
                    patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm, depths=[3, 3, 6, 3], sr_ratios=[8, 4, 2, 1])
    summary(model, (3, 256, 256), device="cpu")
