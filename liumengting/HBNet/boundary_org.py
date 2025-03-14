import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
def Upsample(x, size):
    return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)
class ResBlock(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1):
        super(ResBlock, self).__init__()
        ## conv branch
        self.left = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chs)
        )
        ## shortcut branch
        self.short_cut = nn.Sequential()
        if stride != 1 or in_chs != out_chs:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chs))

    ### get the residual
    def forward(self, x):
        return F.relu(self.left(x) + self.short_cut(x))
class ModuleHelper:
    @staticmethod
    def BNReLU(num_features, inplace=True):
        return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=inplace)
        )

    @staticmethod
    def BatchNorm2d(num_features):
        return nn.BatchNorm2d(num_features)

    @staticmethod
    def Conv3x3_BNReLU(in_channels, out_channels, stride=1, dilation=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=groups, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1_BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)
class Conv1x1(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv1x1, self).__init__()

        self.conv = ModuleHelper.Conv1x1_BNReLU(in_chs, out_chs)

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)

 #将任意维度的特征转化为通道数为3的特征
class ConvTo3Channels(nn.Module):
    def __init__(self, in_channels):
        super(ConvTo3Channels, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out
class BoundaryEnhancementModule(nn.Module):
    def __init__(self, in_chs=3, out_chs=128):#更改通道数，以求对应
        
        super(BoundaryEnhancementModule, self).__init__()
        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(in_chs, 128, (1, 7)),
            ModuleHelper.BNReLU(128),
#             ResBlock(128, 128),
            
            Conv1x1(128, 1)
        )  # bs,1,352,346
        self.conv1x1_h = Conv1x1(2, 8)

        self.vertical_conv = nn.Sequential(
            nn.Conv2d(in_chs, 128, (7, 1)),
            ModuleHelper.BNReLU(128),
#             ResBlock(128, 128),
            
            Conv1x1(128, 1)
            
        )
        
        self.conv1x1_v = Conv1x1(2, 8)
        self.conv_out = Conv1x1(16, out_chs)

        self.transto3 = ConvTo3Channels(in_chs)

    def forward(self, x):
        #print('传进来的是'+ str(x.shape))
        #x:ou2:16,256,80,80
        bs, chl, w, h = x.size()[0], x.size()[1], x.size()[2], x.size()[3]
        x_h = self.horizontal_conv(x)
        x_h = Upsample(x_h, (w, h))
        x_v = self.vertical_conv(x)
        x_v = Upsample(x_v, (w, h))
        x_arr = x.cpu().detach().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((bs, 1, w, h))
        for i in range(bs):
            canny[i] = cv2.Canny(x_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cpu().float().to(x_h.device)

        h_canny = torch.cat((x_h, canny), dim=1)
        v_canny = torch.cat((x_v, canny), dim=1)
        h_v_canny = torch.cat((self.conv1x1_h(h_canny), self.conv1x1_v(v_canny)), dim=1)
        h_v_canny_out = self.conv_out(h_v_canny)

        return h_v_canny_out
def initialize(self):
        initialize_weights(self)

def main():
    # cfg = Dataset.Config(datapath='./data/my_data_large', savepath=SAVE_PATH, mode='train', batch=batch, lr=1e-3,momen=0.9, decay=5e-4, epoch=50)
    # 随机生成与数据大小相同的数组
    random_array = np.random.rand(2, 3, 64, 64)
    #print(random_array)
    # 将随机数组转换为 PyTorch 的 Tensor 格式
    input_tensor = torch.Tensor(random_array)

    model = BoundaryEnhancementModule()
    #out2, out3, out4, out5 = model(input_tensor, 'Test')
    # 设置模型为评估模式
    model.eval()
    # 使用模型进行推理
    with torch.no_grad():
        output = model(input_tensor)
        # print(output)
        # print(output.shape)

if __name__ == '__main__':
    main()