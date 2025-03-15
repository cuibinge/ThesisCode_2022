import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupBatchnorm2d16(nn.Module):
    def __init__(self,
                 c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d16, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):

        N, C, H, W = x.size()
        x = x.contiguous().view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU16(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d16(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct2(x_1, x_2)
        return x


    def reconstruct2(self, x_1, x_2):

        x_1_part = x_1[:, :3 * x_1.size(1) // 4, :, :]
        x_2_part = x_2[:, :x_2.size(1) // 4, :, :]
        return torch.cat((x_1_part, x_2_part), dim=1)





class GroupBatchnorm2d4(nn.Module):
    def __init__(self,
                 c_num: int,
                 group_num: int = 4,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d4, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):

        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU4(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d4(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct2(x_1, x_2)
        return x


    def reconstruct2(self, x_1, x_2):

        x_1_part = x_1[:, :3 * x_1.size(1) // 4, :, :]
        x_2_part = x_2[:, :x_2.size(1) // 4, :, :]
        return torch.cat((x_1_part, x_2_part), dim=1)
