import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from einops import rearrange, repeat


def gain_neighborhood_band(x_train, band, band_patch, patch_all):
    nn = band_patch // 2
    pp = (patch_all) // 2
    x_train_band = torch.zeros((x_train.shape[0], patch_all * band_patch, band), dtype=float)  # 64*27*200

    x_train_band[:, nn * patch_all:(nn + 1) * patch_all, :] = x_train

    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch_all:(i + 1) * patch_all, :i + 1] = x_train[:, :, band - i - 1:]
            x_train_band[:, i * patch_all:(i + 1) * patch_all, i + 1:] = x_train[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train[:, 0:1, :(band - nn + i)]

    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch_all:(nn + i + 2) * patch_all, :band - i - 1] = x_train[:, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch_all:(nn + i + 2) * patch_all, band - i - 1:] = x_train[:, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train[:, 0:1, (i + 1):]
    return x_train_band


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1

        return x


class ViT(nn.Module):
    def __init__(self, n_gcn, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=16,
                 dropout=0., emb_dropout=0., mode='CAF'):
        super().__init__()

        patch_dim = n_gcn

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        x = x.to(torch.float32)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        pos = self.pos_embedding[:, :(n + 1)]
        x += pos
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = self.to_latent(x[:, 0])
        x = self.mlp_head(x)

        return x


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(2)
        batch, l = D.shape
        D1 = torch.reshape(D, (batch * l, 1))
        D1 = D1.squeeze(1)
        D2 = torch.pow(D1, -0.5)
        D2 = torch.reshape(D2, (batch, l))
        D_hat = torch.zeros([batch, l, l], dtype=torch.float)
        for i in range(batch):
            D_hat[i] = torch.diag(D2[i])
        return D_hat.cpu()

    def forward(self, H, A):
        nodes_count = A.shape[1]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)

        (batch, l, c) = H.shape
        H1 = torch.reshape(H, (batch * l, c))
        H2 = self.BN(H1)
        H = torch.reshape(H2, (batch, l, c))
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))  # 点乘
        A_hat = I + A_hat
        output = torch.matmul(A_hat, self.GCN_liner_out_1(H))  # 矩阵相乘
        output = self.Activition(output)
        return output


class neigh_Conv(nn.Module):
    def __init__(self, channel, neigh_number):
        super(neigh_Conv, self).__init__()
        self.neigh_Branch = nn.Sequential()
        self.neigh_number = neigh_number
        for i in range(channel - neigh_number + 1):
            self.neigh_Branch.add_module('neigh_Branch' + str(i),
                                         nn.Conv2d(neigh_number, 1, kernel_size=(1, 1), stride=1))

    def forward(self, x):
        batch, c, w, h = x.shape
        for i in range(c - self.neigh_number + 1):
            if i == 0:
                A = self.neigh_Branch[i](x[:, i:i + self.neigh_number, :, :])
            if i > 0:
                B = self.neigh_Branch[i](x[:, i:i + self.neigh_number, :, :])
                A = torch.cat((A, B), 1)
        return A


class neigh_Conv2(nn.Module):
    def __init__(self, channel, neigh_number):
        super(neigh_Conv2, self).__init__()
        self.neigh_Branch = nn.Sequential()
        self.neigh_number = neigh_number
        for i in range(channel):
            self.neigh_Branch.add_module('neigh_Branch' + str(i),
                                         nn.Conv2d(neigh_number, 1, kernel_size=(1, 1), stride=1))

    def forward(self, x):
        batch, c, w, h = x.shape
        start = int((self.neigh_number - 1) / 2)  # 3 1
        end = int(c - 1 - start)  # c-1
        for i in range(c):
            self_c = x[:, i, :, :]
            self_c = self_c.unsqueeze(1)
            if i == 0:
                A = self_c + self.neigh_Branch[i](x[:, i:i + self.neigh_number, :, :])  # [64 1 21 1]
            if i > 0:
                if i < start:
                    B = self_c + self.neigh_Branch[i](x[:, 0:self.neigh_number, :, :])  # [64 1 21 1]
                if i >= start and i <= end:
                    B = self_c + self.neigh_Branch[i](
                        x[:, (i - start):(i - start + self.neigh_number), :, :])  # [64 1 21 1]
                if i > end:
                    B = self_c + self.neigh_Branch[i](x[:, c - self.neigh_number:c, :, :])  # [64 1 21 1]
                A = torch.cat((A, B), 1)
        return A


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class CPCA_ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        # 使用 1x1 卷积来减少通道维度 (input_channels -> internal_neurons)
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        # 使用 1x1 卷积恢复通道维度 (internal_neurons -> input_channels)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels  # 保存输入通道数

    def forward(self, inputs):
        # 使用自适应平均池化获取每个通道的全局信息
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)  # 通道维度压缩
        x1 = F.relu(x1, inplace=True)  # 激活函数
        x1 = self.fc2(x1)  # 恢复通道维度
        x1 = torch.sigmoid(x1)  # 使用 Sigmoid 激活函数
        # 使用自适应最大池化获取每个通道的全局信息
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)  # 通道维度压缩
        x2 = F.relu(x2, inplace=True)  # 激活函数
        x2 = self.fc2(x2)  # 恢复通道维度
        x2 = torch.sigmoid(x2)  # 使用 Sigmoid 激活函数
        # 将平均池化和最大池化的结果加权求和
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)  # 重新调整形状
        return inputs * x  # 将输入与通道注意力加权后相乘


# CPCA模块
class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()
        # 初始化通道注意力模块
        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        # 初始化深度可分离卷积层（分别处理通道和空间信息）
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)  # 5x5 深度可分离卷积
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3),
                                  groups=channels)  # 1x7 深度可分离卷积
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0),
                                  groups=channels)  # 7x1 深度可分离卷积
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5),
                                   groups=channels)  # 1x11 深度可分离卷积
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0),
                                   groups=channels)  # 11x1 深度可分离卷积
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10),
                                   groups=channels)  # 1x21 深度可分离卷积
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0),
                                   groups=channels)  # 21x1 深度可分离卷积
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)  # 1x1 标准卷积
        self.act = nn.GELU()  # GELU 激活函数

    def forward(self, inputs):
        # Global Perceptron：通过 1x1 卷积和激活函数生成初始的全局表示
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        # 通过通道注意力模块调整通道权重
        inputs = self.ca(inputs)
        # 使用不同的卷积核处理空间信息，分别获得不同尺度的特征
        x_init = self.dconv5_5(inputs)  # 5x5 卷积
        x_1 = self.dconv1_7(x_init)  # 1x7 卷积
        x_1 = self.dconv7_1(x_1)  # 7x1 卷积
        x_2 = self.dconv1_11(x_init)  # 1x11 卷积
        x_2 = self.dconv11_1(x_2)  # 11x1 卷积
        x_3 = self.dconv1_21(x_init)  # 1x21 卷积
        x_3 = self.dconv21_1(x_3)  # 21x1 卷积
        # 合并不同尺度的信息
        x = x_1 + x_2 + x_3 + x_init
        # 使用 1x1 卷积进行最终的空间注意力特征生成
        spatial_att = self.conv(x)
        # 将空间注意力与输入特征相乘
        out = spatial_att * inputs
        # 最后进行一次卷积
        out = self.conv(out)
        return out  # 返回最终的输出


class GCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int):
        super(GCN, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        layers_count = 4
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                if i == 0:
                    self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(self.channel, 128))
                else:
                    self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 128))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 64))
        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

        # self.ca = ChannelAttention(64)
        self.cpca = CPCA(64)
        self.neigh_C = neigh_Conv2(64, 3)
        self.BN = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor, A: torch.Tensor, indexs_train):
        (batch, h, w, c) = x.shape
        _, in_num = indexs_train.shape

        H = torch.reshape(x, (batch, h * w, c))
        for i in range(len(self.GCN_Branch)):
            H = self.GCN_Branch[i](H, A)

        _, _, c_gcn = H.shape
        gcn_out = torch.zeros((batch, in_num, c_gcn), dtype=float)
        gcn_out = gcn_out.type(torch.FloatTensor)
        for i in range(batch):
            gcn_out[i] = H[i][indexs_train[i]]

        gcn_out = gcn_out.transpose(1, 2)
        gcn_out = gcn_out.unsqueeze(3)
        gcn_out = self.cpca(gcn_out) * gcn_out
        gcn_out = self.neigh_C(gcn_out)
        gcn_out = gcn_out.squeeze(3)
        gcn_out = self.BN(gcn_out)
        gcn_out = gcn_out.transpose(1, 2)

        tr_in = gcn_out.transpose(1, 2)
        return tr_in.cpu()
