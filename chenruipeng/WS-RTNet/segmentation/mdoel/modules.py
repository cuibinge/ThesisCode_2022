import torch 
import torch.nn as nn
import torch.nn.functional as F

class Bi_DirectionalDecoder(nn.Module):
    def __init__(self, interpolate_size):
        super().__init__()
        self.interpolate_size = interpolate_size
        self.top_down_path_conv1x1_stage3 = nn.Conv2d(in_channels=512, out_channels=320, kernel_size=1)
        self.top_down_path_conv1x1_stage2 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1)
        self.top_down_path_conv1x1_stage1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.bottom_up_path_conv1x1_stage2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.bottom_up_path_conv1x1_stage3 = nn.Conv2d(in_channels=128, out_channels=320, kernel_size=1)
        self.bottom_up_path_conv1x1_stage4 = nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1)
        self.down2x_by_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=512,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
    
    def top_down_path(self, encoder1, encoder2, encoder3, encoder4,):
        stage4 = F.interpolate(encoder4, size=(self.interpolate_size, self.interpolate_size), mode='bilinear', align_corners=True)
        stage3 = self.top_down_path_conv1x1_stage3(stage4) + F.interpolate(encoder3, size=(self.interpolate_size, self.interpolate_size), mode='bilinear', align_corners=True)
        stage2 = self.top_down_path_conv1x1_stage2(stage3) + F.interpolate(encoder2, size=(self.interpolate_size, self.interpolate_size), mode='bilinear', align_corners=True)
        stage1 = self.top_down_path_conv1x1_stage1(stage2) + F.interpolate(encoder1, size=(self.interpolate_size, self.interpolate_size), mode='bilinear', align_corners=True)
        return torch.cat([stage1 ,stage2 , stage3, stage4],dim=1)
    
    def bottom_up_path(self, encoder1, encoder2, encoder3, encoder4):
        stage2 = encoder2+self.bottom_up_path_conv1x1_stage2(self.down2x_by_maxpool(encoder1))
        stage3 = encoder3+self.bottom_up_path_conv1x1_stage3(self.down2x_by_maxpool(stage2))
        stage4 = encoder4+self.bottom_up_path_conv1x1_stage4(self.down2x_by_maxpool(stage3))
        return torch.cat([F.interpolate(encoder1, size=(self.interpolate_size, self.interpolate_size), mode='bilinear',align_corners=True),\
                F.interpolate(stage2,size=(self.interpolate_size, self.interpolate_size), mode='bilinear',align_corners=True),\
                F.interpolate(stage3,size=(self.interpolate_size, self.interpolate_size), mode='bilinear',align_corners=True),\
                F.interpolate(stage4,size=(self.interpolate_size, self.interpolate_size), mode='bilinear',align_corners=True)],dim=1)
    
    def forward(self, input1, input2, input3, input4):
        top_down_feats = self.top_down_path(input1, input2, input3, input4)
        bottom_up_feats = self.bottom_up_path(input1, input2, input3, input4)
        return self.cbr(torch.cat([top_down_feats, bottom_up_feats], dim=1))
    
class StripPyramidPool(nn.Module):
    def __init__(self, in_chans, trans_chans) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.maxpool_3x3 = nn.MaxPool2d(kernel_size=3)
        self.maxpool_5x5 = nn.MaxPool2d(kernel_size=5)
        self.maxpool_7x7 = nn.MaxPool2d(kernel_size=7)
        self.conv1x1 = nn.Conv2d(in_channels=in_chans,out_channels=in_chans//5, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels=(in_chans//5)*4 + in_chans, out_channels=trans_chans, kernel_size=3, padding=1)
    
    def forward(self,x):
        global_pool = F.interpolate(self.conv1x1(self.global_pool(x)),size=x.size()[2:],mode='bilinear',align_corners=True)
        maxpool_3x3 = F.interpolate(self.conv1x1(self.maxpool_3x3(x)),size=x.size()[2:],mode='bilinear',align_corners=True)
        maxpool_5x5 = F.interpolate(self.conv1x1(self.maxpool_5x5(x)),size=x.size()[2:],mode='bilinear',align_corners=True)
        maxpool_7x7 = F.interpolate(self.conv1x1(self.maxpool_7x7(x)),size=x.size()[2:],mode='bilinear',align_corners=True)
        return self.conv3x3(torch.cat([global_pool,maxpool_3x3,maxpool_5x5,maxpool_7x7,x],dim=1))
    
class FeatureAggregationModule(nn.Module):
    def __init__(self, cnn_chans, trans_chans) -> None:
        super().__init__()
        self.branch1 = StripPyramidPool(cnn_chans, trans_chans)
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=cnn_chans+trans_chans, out_channels=trans_chans,kernel_size=3,padding=1),
                                     nn.BatchNorm2d(trans_chans),
                                     nn.GELU())
        self.branch3 = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                     nn.Conv2d(in_channels=trans_chans, out_channels=trans_chans // 4, kernel_size=1),
                                     nn.GELU(),
                                     nn.Conv2d(in_channels=trans_chans//4, out_channels=trans_chans, kernel_size=1),
                                     nn.Sigmoid())
        self.head = nn.Sequential(nn.Conv2d(in_channels=trans_chans*3, out_channels=trans_chans, kernel_size=3,padding=1),
                                  nn.BatchNorm2d(trans_chans),
                                  nn.GELU())
    
    def forward(self, cnn_block, trans_block):
        branch1 = self.branch1(cnn_block)
        branch2 = self.branch2(torch.cat([cnn_block, trans_block], dim=1))
        branch3 = self.branch3(branch2) * trans_block
        return self.head(torch.cat([branch1, branch2, branch3],dim=1))
    
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=1, stride=1,padding=0,dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_chans,
            out_channels=in_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_chans
        )
        self.bn = nn.BatchNorm2d(num_features=in_chans)
        self.pointwise = nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=1
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x
