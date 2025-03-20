# --------------------------------------------------------
# CAT-Seg: Cost Aggregation for Open-vocabulary Semantic Segmentation
# Licensed under The MIT License [see LICENSE for details]
# Written by Seokju Cho and Heeseong Shin
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pandas as pd

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert

# Modified Swin Transformer blocks for guidance implementetion
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, appearance_guidance_dim, num_heads, head_dim=None, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim + appearance_guidance_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        q = self.q(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x[:, :, :self.dim]).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, dim, appearance_guidance_dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, appearance_guidance_dim=appearance_guidance_dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                for w in (
                        slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, appearance_guidance):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if appearance_guidance is not None:
            appearance_guidance = appearance_guidance.view(B, H, W, -1)
            x = torch.cat([x, appearance_guidance], dim=-1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerBlockWrapper(nn.Module):
    def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=5, pad_len=0):
        super().__init__()
        self.block_1 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=0)
        self.block_2 = SwinTransformerBlock(dim, appearance_guidance_dim, input_resolution, num_heads=nheads, head_dim=None, window_size=window_size, shift_size=window_size // 2)
        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, appearance_guidance_dim)) if pad_len > 0 and appearance_guidance_dim > 0 else None
    
    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
            appearance_guidance: B C H W
        """
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'B C T H W -> (B T) (H W) C')
        if appearance_guidance is not None:
            appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
        x = self.block_1(x, appearance_guidance)
        x = self.block_2(x, appearance_guidance)
        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError
    
    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
        k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, 'B L H D -> B L (H D)')
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type='linear', pooling_size=(4, 4), pad_len=256) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads, attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, guidance_dim)) if pad_len > 0 and guidance_dim > 0 else None
    
    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, C, T, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()
        
        if self.padding_tokens is not None:
            orig_len = x.size(2)
            if orig_len < self.pad_len:
                # pad to pad_len
                padding_tokens = repeat(self.padding_tokens, '1 1 C -> B C T H W', B=B, T=self.pad_len - orig_len, H=H_pool, W=W_pool)
                x_pool = torch.cat([x_pool, padding_tokens], dim=2)

        x_pool = rearrange(x_pool, 'B C T H W -> (B H W) T C')
        if guidance is not None:
            if self.padding_guidance is not None:
                if orig_len < self.pad_len:
                    padding_guidance = repeat(self.padding_guidance, '1 1 C -> B T C', B=B, T=self.pad_len - orig_len)
                    guidance = torch.cat([guidance, padding_guidance], dim=1)
            guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance) # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool)) # MLP

        x_pool = rearrange(x_pool, '(B H W) T C -> (B T) C H W', H=H_pool, W=W_pool)
        x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
        x_pool = rearrange(x_pool, '(B T) C H W -> B C T H W', B=B)

        if self.padding_tokens is not None:
            if orig_len < self.pad_len:
                x_pool = x_pool[:, :, :orig_len]

        x = x + x_pool # Residual
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256) -> None:
        super().__init__()
        self.swin_block = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.attention = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)

    def forward(self, x, appearance_guidance, text_guidance):
        """
        Arguments:
            x: B C T H W
        """
        x = self.swin_block(x, appearance_guidance)
        x = self.attention(x, text_guidance)
        return x


class AggregatorResNetLayer(nn.Module):
    def __init__(self, hidden_dim=64, appearance_guidance=512) -> None:
        super().__init__()
        self.conv_linear = nn.Conv2d(hidden_dim + appearance_guidance, hidden_dim, kernel_size=1, stride=1)
        self.conv_layer = Bottleneck(hidden_dim, hidden_dim // 4)


    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
        """
        B, T = x.size(0), x.size(2)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        appearance_guidance = repeat(appearance_guidance, 'B C H W -> (B T) C H W', T=T)

        x = self.conv_linear(torch.cat([x, appearance_guidance], dim=1))
        x = self.conv_layer(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, guidance=None):
        x = self.up(x)
        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)

#-------------------------------------------------------------------------------------------------------#

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channel, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return x * self.sigmoid(out.view(x.size(0), x.size(1), 1, 1))

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.conv(torch.cat([avg_pool, max_pool], dim=1))

class GuidedUp_v2(nn.Module):
    """改进的引导上采样模块，使用可学习的卷积方式"""
    def __init__(self, in_ch, out_ch, guide_ch=None):
        super().__init__()
        # 可学习的上采样（反卷积）
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),  # 反卷积上采样
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        # 引导特征处理（加入通道注意力）
        self.guide_proj = nn.Sequential(
            nn.Conv2d(guide_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            ChannelAttention(out_ch)
        ) if guide_ch else None

        # 特征融合（加入空间注意力）
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),  # 普通卷积
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            SpatialAttention()
        ) if guide_ch else None

    def forward(self, x, guide=None):
        x = self.up(x)
        if guide is not None and self.guide_proj:
            B, C, H, W = x.shape
            original_guide = guide  # [3, 32, 48, 48]
            
            # 重复引导特征以匹配批次维度
            T = B // original_guide.shape[0]  # 计算重复次数 12/3=4
            guide = repeat(original_guide, "B C H W -> (B T) C H W", T=T)
            
            # 调整空间分辨率
            guide = F.interpolate(guide, size=x.shape[-2:], mode='bilinear', align_corners=True)
            guide = self.guide_proj(guide)
            x = torch.cat([x, guide], dim=1)
            x = self.fusion(x)
        return x


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块，使用普通卷积"""
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # 普通卷积
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ) for i in range(3)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),  # 普通卷积
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        features = [F.interpolate(f, size=x.shape[-2:], mode='bilinear', align_corners=True)
                    for f in features]
        return self.fusion(torch.cat(features, dim=1))

class HighResDecoder(nn.Module):
    """高分辨率解码器"""
    def __init__(self, hidden_dim=128, guidance_dims=[32, 16]):
        super().__init__()
        # 四级上采样 (24->48->96->192->384)
        self.decoder1 = GuidedUp_v2(hidden_dim, 64, guidance_dims[0])  # 24->48
        self.decoder2 = GuidedUp_v2(64, 32, guidance_dims[1])          # 48->96
        # self.decoder3 = GuidedUp_v2(32, 16, None)                      # 96->192
        # self.decoder4 = GuidedUp_v2(16, 8, None)                       # 192->384

        # 多尺度特征融合
        # self.msf = MultiScaleFusion(8)

        # 最终预测层
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 使用普通卷积
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)   # 最终输出
        )

    def forward(self, x, guidance):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'B C T H W -> (B T) C H W')

        # 上采样流程
        x = self.decoder1(x, guidance[0] if guidance else None)  # 24->48
        x = self.decoder2(x, guidance[1] if guidance else None)  # 48->96
        # x = self.decoder3(x)  # 96->192
        # x = self.decoder4(x)  # 192->384

        # 多尺度融合
        # x = self.msf(x)

        # 最终预测
        x = self.head(x)  # [B*T, 1, 384, 384]
        x = rearrange(x, '(B T) () H W -> B T H W', B=B)
        return x


# --------------------------------------------------GIB-----------------------------------------------------#
from einops import rearrange


def infonce_loss(x, y, temperature=0.1, neg_sampling_ratio=0.1):
    """
    计算 InfoNCE 损失，增加负样本采样策略。
    x: 视觉特征，形状为 [B, T, S, D]，其中 S 是空间样本数（例如 H*W）
    y: 视觉特征，形状为 [B, T, H*W, C]，需要与 x 对齐
    temperature: 温度参数，用于控制分布平滑程度
    neg_sampling_ratio: 负样本采样的比例，用于增强负样本的多样性
    """
    B, T, S, D = x.shape
    _, _, H_W, C = y.shape
    
    # 展平 x 的空间维度
    x = x.reshape(B * T * S, D)  # [B*T*S, D]
    
    # 展平 y 的空间维度
    y = y.reshape(B * T * H_W, C)  # [B*T*H*W, C]
    
    # 归一化特征以提高稳定性
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    # 计算相似度矩阵（点积）
    similarity = torch.matmul(x, y.T) / temperature  # [B*T*S, B*T*H*W]
    
    # 构造正样本标签：假设 x 和 y 的空间维度是对齐的（S = H*W）
    labels = torch.arange(B * T * S, device=x.device)  # 正样本标签，对应于 x 的每个样本
    
    # 如果 S != H*W，则需要调整相似度和标签
    if S != H_W:
        # 通过插值或池化对齐 x 和 y 的空间维度
        raise ValueError("S and H*W must be equal for InfoNCE loss. Consider aligning spatial dimensions.")
    
    # 增强负样本采样：随机选择一部分负样本，增加挑战性
    num_neg_samples = int(B * T * H_W * neg_sampling_ratio)
    neg_indices = torch.randperm(B * T * H_W, device=x.device)[:num_neg_samples]
    neg_similarity = similarity[:, neg_indices]  # [B*T*S, num_neg_samples]
    
    # 拼接正样本和负样本的相似度
    full_similarity = torch.cat([similarity, neg_similarity], dim=1)  # [B*T*S, B*T*H*W + num_neg_samples]
    
    # 构造正确的标签：正样本标签为对应的索引，负样本标签为 -1（将被忽略）
    full_labels = labels  # [B*T*S]
    # 注意：负样本的标签不需要显式拼接，因为 F.cross_entropy 会忽略负样本的贡献
    
    # 计算交叉熵损失
    loss = F.cross_entropy(full_similarity, full_labels, ignore_index=-1)
    
    return loss

class IBLayerSpatial(nn.Module):
    def __init__(self, in_channels, out_channels, feature_channels, beta=1.0, temperature=0.1):
        super(IBLayerSpatial, self).__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.beta = beta
        self.temperature = temperature
        # 添加映射层，将 z_spatial 和 x 映射到相同的特征维度
        self.map_z = nn.Linear(out_channels, feature_channels)
        self.map_x = nn.Linear(in_channels, feature_channels)
    
    def setup_graph(self, edge_index, batch_size, device):
        u, v = edge_index
        g_list = []
        for _ in range(batch_size):
            g = dgl.graph((u, v), device=device)
            g = dgl.add_self_loop(g)
            g_list.append(g)
        return dgl.batch(g_list)
    
    def forward(self, x, edge_index, feature_map):
        B, N, C_in = x.shape
        
        # 动态生成图结构
        batch_g = self.setup_graph(edge_index, B, x.device)
        
        x_reshaped = x.reshape(B * N, C_in)
        z_spatial = self.conv(batch_g, x_reshaped).view(B, N, -1)
        assert torch.isfinite(z_spatial).all(), "z_spatial contains NaN or Inf"
        
        # 将 z_spatial 和 x 映射到相同的特征维度
        z_spatial_mapped = self.map_z(z_spatial)  # [B, N, feature_channels]
        x_mapped = self.map_x(x)  # [B, N, feature_channels]
        
        # 处理视觉特征：展平空间维度
        B, C, T, H, W = feature_map.shape
        feature_map_reshaped = feature_map.permute(0, 2, 1, 3, 4).reshape(B, T, C, H * W)  # [B, T, C, H*W]
        feature_map_reshaped = feature_map_reshaped.permute(0, 1, 3, 2)  # [B, T, H*W, C]
        
        # 注意：这里 z_spatial_mapped 和 x_mapped 的形状是 [B, T, feature_channels]，需要添加一个伪空间维度
        z_spatial_mapped_expanded = z_spatial_mapped.unsqueeze(2).expand(-1, -1, H * W, -1)  # [B, T, H*W, feature_channels]
        x_mapped_expanded = x_mapped.unsqueeze(2).expand(-1, -1, H * W, -1)  # [B, T, H*W, feature_channels]
        
        I_z_Y = infonce_loss(z_spatial_mapped_expanded, feature_map_reshaped, self.temperature)
        I_z_X = infonce_loss(z_spatial_mapped_expanded, x_mapped_expanded, self.temperature)
        loss_spatial = I_z_Y + self.beta * I_z_X
        
        self.I_z_Y = I_z_Y
        self.I_z_X = I_z_X
        
        # print(f"IBLayerSpatial - I_z_Y: {I_z_Y.item()}, I_z_X: {I_z_X.item()}, loss_spatial: {loss_spatial.item()}")
        return z_spatial, loss_spatial

class IBLayerAttribute(nn.Module):
    def __init__(self, in_channels, out_channels, feature_channels, beta=1.0, temperature=0.1):
        super(IBLayerAttribute, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads=4)
        self.beta = beta
        self.temperature = temperature
        # 添加映射层，将 z_attribute 和 x 映射到相同的特征维度
        self.map_z = nn.Linear(out_channels, feature_channels)
        self.map_x = nn.Linear(in_channels, feature_channels)

    def forward(self, x, feature_map):
        x_original = x
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        z_attribute, _ = self.attention(x, x, x)
        z_attribute = z_attribute.permute(1, 0, 2)
        z_attribute = F.relu(z_attribute)
        assert torch.isfinite(z_attribute).all(), "z_attribute contains NaN or Inf"
        
        # 将 z_attribute 和 x 映射到相同的特征维度
        z_attribute_mapped = self.map_z(z_attribute)  # [B, N, feature_channels]
        x_mapped = self.map_x(x_original)  # [B, N, feature_channels]
        
        # 处理视觉特征：展平空间维度
        B, C, T, H, W = feature_map.shape
        feature_map_reshaped = feature_map.permute(0, 2, 1, 3, 4).reshape(B, T, C, H * W)  # [B, T, C, H*W]
        feature_map_reshaped = feature_map_reshaped.permute(0, 1, 3, 2)  # [B, T, H*W, C]
        
        # 注意：这里 z_attribute_mapped 和 x_mapped 的形状是 [B, T, feature_channels]，需要添加一个伪空间维度
        z_attribute_mapped_expanded = z_attribute_mapped.unsqueeze(2).expand(-1, -1, H * W, -1)  # [B, T, H*W, feature_channels]
        x_mapped_expanded = x_mapped.unsqueeze(2).expand(-1, -1, H * W, -1)  # [B, T, H*W, feature_channels]
        
        I_z_Y = infonce_loss(z_attribute_mapped_expanded, feature_map_reshaped, self.temperature)
        I_z_X = infonce_loss(z_attribute_mapped_expanded, x_mapped_expanded, self.temperature)
        loss_attribute = I_z_Y + self.beta * I_z_X
        
        self.I_z_Y = I_z_Y
        self.I_z_X = I_z_X
        
        # print(f"IBLayerAttribute - I_z_Y: {I_z_Y.item()}, I_z_X: {I_z_X.item()}, loss_attribute: {loss_attribute.item()}")
        return z_attribute, loss_attribute
import torch
import torch.nn as nn
import torch.nn.functional as F

class CategorySpecificRefinement(nn.Module):
    def __init__(self, feature_channels, embed_channels, height, width):
        super(CategorySpecificRefinement, self).__init__()
        self.feature_channels = feature_channels
        self.height = height
        self.width = width
        
        # 映射类别表征到特征图的通道维度
        self.embed_to_feature = nn.Linear(embed_channels, feature_channels)
        
        # 使用注意力机制增强特征融合
        self.attention = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature_map, class_representation):
        B, C, T, H, W = feature_map.shape
        B_cr, T_cr, embed_channels = class_representation.shape
        assert T == T_cr, f"feature_map 的 T 维度 ({T}) 与 class_representation 的 T 维度 ({T_cr}) 不一致"
        assert self.feature_channels == C, f"feature_channels ({self.feature_channels}) 与 feature_map 的 C 维度 ({C}) 不一致"
        
        embed_mapped = self.embed_to_feature(class_representation)  # [B, T, C]
        embed_mapped = embed_mapped.permute(0, 2, 1)  # [B, C, T]
        embed_mapped = embed_mapped.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, H, W)  # [B, C, T, H, W]
        
        # 使用注意力机制融合特征
        combined = torch.cat([feature_map, embed_mapped], dim=1)  # [B, 2C, T, H, W]
        combined = rearrange(combined, 'B C T H W -> (B T) C H W')  # [B*T, 2C, H, W]
        attention_weights = self.attention(combined)  # [B*T, C, H, W]
        attention_weights = rearrange(attention_weights, '(B T) C H W -> B C T H W', B=B)  # [B, C, T, H, W]
        
        refined_feature = feature_map + attention_weights * embed_mapped  # [B, C, T, H, W]
        
        return refined_feature

class NonlinearCombination(nn.Module):
    def __init__(self, in_channels, hidden_channels, embed_channels, feature_channels, beta=1.0, temperature=0.1, loss_weight=0.1):
        super(NonlinearCombination, self).__init__()
        self.ib_spatial = IBLayerSpatial(in_channels, hidden_channels, feature_channels, beta, temperature)
        self.ib_attribute = IBLayerAttribute(in_channels, hidden_channels, feature_channels, beta, temperature)
        self.fc_combination = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, embed_channels)
        )
        self.loss_weight = loss_weight
    
    def update_beta(self, beta):
        self.ib_spatial.beta = beta
        self.ib_attribute.beta = beta

    def forward(self, x, edge_index, feature_map):
        z_spatial, loss_spatial = self.ib_spatial(x, edge_index, feature_map)
        z_attribute, loss_attribute = self.ib_attribute(x, feature_map)
        z_combined = torch.cat([z_spatial, z_attribute], dim=-1)
        out = self.fc_combination(z_combined)
        total_loss = (loss_spatial + loss_attribute) * self.loss_weight
        return F.log_softmax(out, dim=-1), total_loss

class GNNFeatureMapOptimizationWithRefinement(nn.Module):
    def __init__(self, feature_channels, embed_channels, height, width, beta=1.0, temperature=0.1, loss_weight=0.1):
        super(GNNFeatureMapOptimizationWithRefinement, self).__init__()
        self.gnn = NonlinearCombination(
            in_channels=embed_channels,
            hidden_channels=feature_channels, 
            embed_channels=embed_channels,
            feature_channels=feature_channels,
            beta=beta,
            temperature=temperature,
            loss_weight=loss_weight
        )
        self.feature_refinement = CategorySpecificRefinement(feature_channels, embed_channels, height, width)

    def forward(self, feature_map, edge_index, node_embeddings):
        class_repr, loss = self.gnn(node_embeddings, edge_index, feature_map)
        refined_map = self.feature_refinement(feature_map, class_repr.exp())
        return refined_map, loss
# ----------------------------------------------------------------------------------------------------------#
def get_edge_index_from_xlsx(xlsx_file, entity_list):
    df = pd.read_excel(xlsx_file, header=None, usecols=[0, 1, 2])
    edge_index = [[], []]
    for _, row in df.iterrows():
        head_entity = row[0]
        tail_entity = row[2]
        if head_entity in entity_list and tail_entity in entity_list:
            head_index = entity_list.index(head_entity)
            tail_index = entity_list.index(tail_entity)
            edge_index[0].append(head_index)
            edge_index[1].append(tail_index)
        # print(edge_index)
    return edge_index


class Aggregator(nn.Module):
    def __init__(self, 
        kg_path = "",
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32, 16),
        decoder_guidance_dims=(256, 128, 64),
        decoder_guidance_proj_dims=(32, 16, 8),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
        classname = None,
    ) -> None:
        """
        CAT-Seg的成本聚合模型
        参数:
            text_guidance_dim: 文本引导的维度
            text_guidance_proj_dim: 投影后的文本引导的维度
            appearance_guidance_dim: 视觉引导的维度
            appearance_guidance_proj_dim: 投影后的视觉引导的维度
            decoder_dims: 上采样解码器的维度
            decoder_guidance_dims: 上采样解码器引导的维度
            decoder_guidance_proj_dims: 上采样解码器引导投影后的维度
            num_layers: 聚合器的层数
            nheads: 注意力头的数量
            hidden_dim: 变压器块的隐藏维度
            pooling_size: 类别聚合层的池化大小
                        为了减少计算量，我们在类别聚合块中应用池化操作，以减少训练期间的令牌数量。
            feature_resolution: 空间聚合的特征分辨率
            window_size: 空间聚合中Swin块的窗口大小
            attention_type: 类别聚合的注意力类型。 
            prompt_channel: 用于集成文本特征的提示数量。默认值: 1
            pad_len: 类别聚合的填充长度。默认值: 256
                    pad_len 强制类别聚合块对所有输入都具有固定长度的令牌。
                    这意味着在类别聚合中，它要么用可学习的令牌填充序列，要么用初始的CLIP余弦相似度分数截断类别。
                    将 pad_len 设置为0可禁用此功能。

            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # self.layers = nn.ModuleList([
        #     AggregatorLayer(
        #         hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
        #         nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
        #     ) for _ in range(num_layers)
        # ])

        
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.decoder3 = Up(decoder_dims[1], decoder_dims[2], decoder_guidance_proj_dims[2])
        self.head = nn.Conv2d(decoder_dims[2], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

        #-----------------------------------------------------------------------------------------#
        self.HighResDecoder = HighResDecoder(
            hidden_dim=128,
            guidance_dims=[32, 16]  # 与projected_decoder_guidance维度匹配
        )
        #-----------------------------------------------------------------------------------------#
        self.edge_index = get_edge_index_from_xlsx(kg_path, classname)
        self.GIB = GNNFeatureMapOptimizationWithRefinement(feature_channels=128, embed_channels=512, height=24, width=24)

    def feature_map(self, img_feats, text_feats):
        """
        生成用于特征聚合基线的拼接特征体。
        Args:
            img_feats (torch.Tensor): 图像特征，形状为 (B, C, H, W)。
            text_feats (torch.Tensor): 文本特征，形状为 (B, T, P, C)。
        Returns:
            torch.Tensor: 拼接后的特征体，形状为 (B, 2C, T, H, W)。
        """
        # 对图像特征在通道维度上进行归一化操作，形状为 (B, C, H, W)
        img_feats = F.normalize(img_feats, dim=1) 
        # 重复图像特征，在类别维度上进行扩展，形状变为 (B, C, T, H, W)
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        # 对文本特征在最后一个维度上进行归一化操作，形状为 (B, T, P, C)
        text_feats = F.normalize(text_feats, dim=-1) 
        # 对不同提示下的文本特征求平均值，形状变为 (B, T, C)
        text_feats = text_feats.mean(dim=-2) 
        # 再次对文本特征在最后一个维度上进行归一化操作，形状为 (B, T, C)
        text_feats = F.normalize(text_feats, dim=-1) 
        # 重复文本特征，在高度和宽度维度上进行扩展，形状变为 (B, C, T, H, W)
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        # 在通道维度上拼接图像特征和文本特征，形状为 (B, 2C, T, H, W)
        return torch.cat((img_feats, text_feats), dim=1) 


    def correlation(self, img_feats, text_feats):
        """
        计算图像特征和文本特征之间的相关性。
        Args:
            img_feats (torch.Tensor): 图像特征，形状为 (B, C, H, W)。
            text_feats (torch.Tensor): 文本特征，形状为 (B, T, P, C)。
        Returns:
            torch.Tensor: 相关性张量，形状为 (B, P, T, H, W)。
        """
        # 对图像特征在通道维度上进行归一化操作，形状为 (B, C, H, W)
        img_feats = F.normalize(img_feats, dim=1) 
        # 对文本特征在最后一个维度上进行归一化操作，形状为 (B, T, P, C)
        text_feats = F.normalize(text_feats, dim=-1) 
        # 使用爱因斯坦求和约定计算图像特征和文本特征的相关性，形状为 (B, P, T, H, W)
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr


    def corr_embed(self, x):
        """
        扩充成本体积维度
        Args:
            x (torch.Tensor): 输入的相关性张量，形状为 (B, P, T, H, W)。
        Returns:
            torch.Tensor: 嵌入后的相关性张量，形状为 (B, C, T, H, W)。
        """
        # 获取输入张量的批量大小
        B = x.shape[0]
        # 使用 einops 的 rearrange 函数将输入张量的维度进行重排
        # 从 (B, P, T, H, W) 重排为 ((B * T), P, H, W)
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        # 将重排后的张量输入到卷积层 self.conv1 中进行卷积操作
        corr_embed = self.conv1(corr_embed)
        # 再次使用 einops 的 rearrange 函数将卷积后的张量的维度进行重排
        # 从 ((B * T), C, H, W) 重排为 (B, C, T, H, W)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        # 返回嵌入后的相关性张量
        return corr_embed

    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        """
        对输入的特征图进行上采样操作。
        Args:
            x (torch.Tensor): 输入的特征图，形状为 (B, C, T, H, W)，
                              其中 B 为批量大小，C 为通道数，T 为类别数，H 为高度，W 为宽度。
        Returns:
            torch.Tensor: 上采样后的特征图，形状为 (B, C, T, 2*H, 2*W)。
        """
        # 获取输入特征图的批量大小
        B = x.shape[0]
        # 使用 einops 的 rearrange 函数将输入特征图的维度进行重排
        # 从 (B, C, T, H, W) 重排为 ((B * T), C, H, W)
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        # 使用 PyTorch 的 F.interpolate 函数对重排后的特征图进行双线性插值上采样
        # 上采样的比例因子为 2，即高度和宽度都变为原来的 2 倍
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        # 再次使用 einops 的 rearrange 函数将上采样后的特征图的维度进行重排
        # 从 ((B * T), C, 2*H, 2*W) 重排为 (B, C, T, 2*H, 2*W)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        # 返回上采样后的特征图
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0] # print(x.shape, guidance[0].shape, guidance[1].shape) #torch.Size([3, 128, 4, 24, 24]) torch.Size([3, 32, 48, 48]) torch.Size([3, 16, 96, 96]) 
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W') # print(corr_embed.shape) # torch.Size([12, 128, 24, 24])
        # 上采样，和引导特征拼接再卷积
        corr_embed = self.decoder1(corr_embed, guidance[0]) # print(corr_embed.shape) # torch.Size([12, 64, 48, 48])
        corr_embed = self.decoder2(corr_embed, guidance[1]) # print(corr_embed.shape) [12, 32, 96, 96]
        corr_embed = self.decoder3(corr_embed, guidance[2]) # print(corr_embed.shape) # torch.Size([12, 16, 96, 96])
        # 将增加的维度映射到1维
        corr_embed = self.head(corr_embed) # print(corr_embed.shape) # torch.Size([12, 1, 96, 96])
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B) # print(corr_embed.shape) # torch.Size([3, 4, 96, 96])
        return corr_embed

    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        前向传播函数，计算图像特征、文本特征和视觉引导的相关性，并生成最终的预测结果。
        Arguments:
            img_feats: (B, C, H, W) 输入的图像特征，B 是批量大小，C 是通道数，H 是高度，W 是宽度。
            text_feats: (B, T, P, C)    # (批次, 类别数, Prompt数, clip嵌入维度)
            apperance_guidance: tuple of (B, C, H, W) 视觉引导信息，是一个元组，每个元素是 (B, C, H, W) 形状的张量。
        """
        classes = None

        text_embed = text_feats.squeeze(2)

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            # 调整相关性张量的维度并展平最后三个维度，然后取最大值
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            # 选择前 pad_len 个最大的类别索引
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            # 对文本特征在最后一个维度上进行归一化
            th_text = F.normalize(text_feats, dim=-1)
            # 根据选择的类别索引提取文本特征
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            # 保存原始的类别数
            orig_clases = text_feats.size(1)
            # 对图像特征在通道维度上进行归一化
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            # 更新文本特征
            text_feats = th_text
            # 重新计算图像特征和文本特征的相关性
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)

        corr_embed = self.corr_embed(corr) # [5, 128, 6, 24, 24] 批次；通道；类别；H；W

        if not isinstance(self.edge_index, torch.Tensor):
            self.edge_index = torch.tensor(self.edge_index)
        edge_index = self.edge_index.pin_memory()
        edge_index = edge_index.to(torch.device('cuda'))
        
        corr_embed, GIBloss = self.GIB(corr_embed, edge_index, text_embed)


        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]
        if self.text_guidance_projection is not None:
            # 对文本特征在提示维度上求平均
            text_feats = text_feats.mean(dim=-2)
            # 对文本特征进行归一化
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            # 对文本特征进行投影
            projected_text_guidance = self.text_guidance_projection(text_feats)

        # print(projected_decoder_guidance[0].shape, projected_decoder_guidance[1].shape, projected_decoder_guidance[2].shape) 
        # torch.Size([3, 32, 48, 48]) torch.Size([3, 16, 96, 96]) torch.Size([3, 8, 192, 192])

        # 通过卷积解码器得到预测的 logit
        # print(projected_guidance[0].shape, projected_guidance[1].shape) torch.Size([128, 24, 24]) torch.Size([128, 24, 24])

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)
        # print(logit.shape) # torch.Size([3, 4, 192, 192])

        # 如果存在选择的类别索引
        if classes is not None:
            # 初始化输出张量，填充为 -100.
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            # 根据选择的类别索引将 logit 填充到输出张量中
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            # 更新 logit
            logit = out
        # 返回预测的 logit
        return logit, GIBloss

