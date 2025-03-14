import torch
import torch.nn as nn
from torchvision import models
from torchstat import stat
from torchsummary import summary

class Model(nn.Module):
    def __init__(self, backbone, num_classes, dropout_ratio=0.6):
        super(Model,self).__init__()

        assert backbone in ["vgg19", "resnet18", "resnet50", "resnet101", "densenet161"]
        self.backbone = eval(f"models.{backbone}")(pretrained=True)
        self.layer1 = nn.Conv2d(in_channels=feature_dim_1, out_channels=feature_dim_1, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(in_channels=feature_dim_1, out_channels=feature_dim_1, kernel_size=1, stride=1, padding=0)
        self.classifier = nn.Linear((feature_dim_1+feature_dim_2), num_classes)

    def forward(self,x):
        out1 = self.backbone(x)     
        out1 = self.layer1(out1)  
        out = self.layer2(out1)  
        x = self.classifier(out)

        return out.detach().cpu().numpy(), x
