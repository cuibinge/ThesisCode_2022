from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
import math
# from timm.models.layers import DropPath
from backbone_resnet import resnet50
from Unet_decode import Up, OutConv
import numpy as np
import cv2
import os
from torch.nn import functional as F
from FreqFusion_12_11_test import FreqFusion

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

class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # 拼接两个输入特征图
        #concatenated = torch.cat((x1, x2), dim=1)
        concatenated = x1+  x2
        # 生成门控信号
        gate = self.gate_conv(concatenated)
        gate = self.sigmoid(gate)

        # 计算门控融合后的特征图
        fused = gate * x1 + (1 - gate) * x2
        return fused

class unet_resnet50(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):

        super(unet_resnet50, self).__init__()
        backbone = resnet50()
        if pretrain_backbone:
            backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

        self.GatedFusion1 = GatedFusion(64)
        self.GatedFusion2 = GatedFusion(256)
        self.GatedFusion3 = GatedFusion(512)

        self.freqfusion1 = FreqFusion(hr_channels=64, lr_channels=128)
        self.freqfusion2 = FreqFusion(hr_channels=256, lr_channels=256)
        self.conv_fuse3 =  nn.Conv2d(in_channels=3584, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_decode1 = nn.Conv2d(537, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_decode2 = nn.Conv2d(153, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.stage_out_channels = [64, 256, 512, 1024, 2048]
        return_layers = {'relu': 'out0', 'layer1': 'out1', 'layer2': 'out2', 'layer3': 'out3', 'layer4': 'out4'}

        return_layers1 = {'layer4': 'out4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.backbone_edge = IntermediateLayerGetter(backbone, return_layers=return_layers1)
        self.new_conv_edeg = nn.Conv2d(1 ,3 , kernel_size=1, stride=1, padding=0 , bias=False)
        self.index = 0

    def forward(self, x: torch.Tensor , edge: torch.Tensor) -> Dict[str, torch.Tensor]:


        input_shape = x.shape[-2:]
        result = OrderedDict()

        backbone_out = self.backbone(x)

        y_edge = self.new_conv_edeg(edge)
        backbone_edge_out = self.backbone(y_edge)

        backbone_out['out3'] =  F.interpolate(backbone_out['out3'], size=(backbone_out['out2'].size()[2], backbone_out['out2'].size()[2]), mode='bilinear', align_corners=False)
        backbone_out['out4'] = F.interpolate(backbone_out['out4'], size=(backbone_out['out2'].size()[2], backbone_out['out2'].size()[2]), mode='bilinear', align_corners=False)

        backbone_edge_out['out3'] = F.interpolate(backbone_edge_out['out3'],size=(backbone_edge_out['out2'].size()[2], backbone_edge_out['out2'].size()[2]),mode='bilinear', align_corners=False)
        backbone_edge_out['out4'] = F.interpolate(backbone_edge_out['out4'],size=(backbone_edge_out['out2'].size()[2], backbone_edge_out['out2'].size()[2]),mode='bilinear', align_corners=False)

        x_fuse3 = F.normalize(torch.cat((backbone_out['out2'], backbone_out['out3'], backbone_out['out4']), dim=1) , p=2, dim=1)
        x_fuse3 = self.conv_fuse3(x_fuse3)

        y_edge_fuse3 = F.normalize(torch.cat((backbone_edge_out['out2'], backbone_edge_out['out3'], backbone_edge_out['out4']), dim=1), p=2,dim=1)
        y_edge_fuse3 = self.conv_fuse3(y_edge_fuse3)

        xy_gate1 = self.GatedFusion1(backbone_out['out0'], backbone_edge_out['out0'])

        xy_gate2 = self.GatedFusion2(backbone_out['out1'], backbone_edge_out['out1'])

        xy_gate3 = self.GatedFusion3(x_fuse3, y_edge_fuse3)

        xy_freqfusion1 = self.freqfusion1(xy_gate1 , xy_gate2)

        xy_freqfusion2 = self.freqfusion2(xy_gate2, xy_gate3)
        print(backbone_out['out3'])

        xy_gate3 = F.interpolate(xy_gate3,size=(xy_freqfusion2.size()[2], xy_freqfusion2.size()[2]), mode='bilinear',align_corners=False)

        xy_decode1 = F.normalize(torch.cat((xy_gate3, xy_freqfusion2), dim=1), p=2,dim=1)
        xy_decode1 = self.conv_decode1(xy_decode1)

        xy_decode1 = F.interpolate(xy_decode1, size=(xy_freqfusion1.size()[2], xy_freqfusion1.size()[2]), mode='bilinear',align_corners=False)

        xy_decode2 = F.normalize(torch.cat((xy_decode1, xy_freqfusion1), dim=1), p=2, dim=1)
        xy_decode2 = self.conv_decode2(xy_decode2)
        #self.feature_vis(x)
        xy_decode2 = F.interpolate(xy_decode2, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = xy_decode2
        return xy_decode2

    def feature_vis(self,feats):

        output_shape = (512, 512)  # 输出形状
        channel_mean = torch.mean(feats, dim=1, keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
        channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
        channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().detach().numpy()
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
    y = torch.randn(4,1,224,224)
    print(model(x , y).size())