from collections import OrderedDict
from typing import Dict

from PSMF import PSFM
from BSSA import SBA
import torch
import torch.nn as nn
from torch import Tensor
import math
#from timm.models.layers import DropPath
from backbone_resnet import resnet50
#from Unet_decode import Up, OutConv
import numpy as np
import cv2
import os
from torch.nn import functional as F

class Conv2d_r(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=3,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            # bias=not (use_batchnorm),
        )
        relu =nn.SiLU()
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2d_r, self).__init__(conv, bn, relu)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class unet_resnet50(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
    # def __init__(self, in_channels, out_channels, height,num_classes, pretrain_backbone: bool = False):      #修改
        super(unet_resnet50, self).__init__()
        backbone = resnet50()
        if pretrain_backbone:
            backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

        self.conv_fuse3 =  nn.Conv2d(in_channels=3584, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_decode1 = nn.Conv2d(537, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_decode2 = nn.Conv2d(2560, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.stage_out_channels = [64, 256, 512, 1024, 2048]
        return_layers = {'relu': 'out0', 'layer1': 'out1', 'layer2': 'out2', 'layer3': 'out3', 'layer4': 'out4'}

        return_layers1 = {'layer4': 'out4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.backbone_edge = IntermediateLayerGetter(backbone, return_layers=return_layers1)
        self.new_conv_edeg = nn.Conv2d(1024 ,2048 , kernel_size=1, stride=1, padding=0 , bias=False)

        self.new_conv_edeg2 = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.new_conv_edeg3 = nn.Conv2d(256, 256, kernel_size=11, stride=2, padding=0, dilation=1 , bias=False)

        self.new_conv_edeg4 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer,num_layers=4)  # transformer修改   num_layers=4采用4层trnasformer
        self.positional_encoding = PositionalEncoding(d_model=2048,max_len=16*16)

        self.PSMF_ = PSFM(Channel=2048)
        self.SBA_1 = SBA(input_dim=256,output_dim=256)

        self.SBA_2 = SBA(input_dim=256, output_dim=256)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.max_pool3= nn.MaxPool2d(kernel_size=2, stride=3)
        self.max_pool6 = nn.MaxPool2d(kernel_size=2, stride=6)
        self.max_pool8 = nn.MaxPool2d(kernel_size=2, stride=8)
        self.index = 0

    def forward(self, x: torch.Tensor ) -> Dict[str, torch.Tensor]:

        input_shape = x.shape[-2:]
        result = OrderedDict()

        backbone_out = self.backbone(x)

       # backbone_out['out3'] =  F.interpolate(backbone_out['out3'], size=(backbone_out['out2'].size()[2], backbone_out['out2'].size()[2]), mode='bilinear', align_corners=False)
        backbone_out['out4'] = F.interpolate(backbone_out['out4'], size=(backbone_out['out3'].size()[2], backbone_out['out3'].size()[2]), mode='bilinear', align_corners=False)

        backbone_out['out3'] = self.new_conv_edeg(backbone_out['out3'])

        x_psmf = self.PSMF_([backbone_out['out3']  ,  backbone_out['out4'] ])

        x_pool = self.new_conv_edeg2(x)

        x_pool1 = self.max_pool1(x_pool)
        x_pool3 = self.max_pool3(x_pool)
        x_pool6 = self.max_pool6(x_pool)
        x_pool8 = self.max_pool8(x_pool)

        x_sba1 = self.SBA_1(x_pool1 ,  x_pool6)
        x_sba2 = self.SBA_2(x_pool3 ,  x_pool8)

        x_sba1 = self.new_conv_edeg3(x_sba1)
        x_sba2 = self.new_conv_edeg4(x_sba2)


        x_cat_sba_psmf = F.normalize(torch.cat((x_sba1,x_sba2, x_psmf ), dim=1) , p=2, dim=1)

        xy_decode2 = self.conv_decode2(x_cat_sba_psmf)

        xy_decode2 = F.interpolate(xy_decode2, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = xy_decode2
        return xy_decode2

    def feature_vis(self,feats):  # feaats形状: [b,c,h,w]

        output_shape = (512, 512)  # 输出形状
        channel_mean = torch.mean(feats, dim=1, keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
        channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
        channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().detach().numpy()  # 四维压缩为二维
        channel_mean = (
                ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
            np.uint8)
        savedir = './crop_combine/combine512/CAM/'
        if not os.path.exists(savedir + 'feature_vis4'): os.makedirs(savedir + 'feature_vis4')
        channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
        filename = savedir + 'feature_vis4/{:01}.tif'.format(self.index)
        cv2.imwrite(filename, channel_mean)
        self.index +=1
if __name__ == '__main__':

    model = unet_resnet50(num_classes=5)
    x = torch.randn(4,3,224,224)
    print(model(x ).size())