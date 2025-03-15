# -*- coding: UTF-8 -*-
"""
@Author  ：zhangyanping of SDUST
@Date    ：2024/1/2 18:57
"""
import os
import numpy as np
import torch
import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from dataset.TT_Dataset  import MyDataset   # 读取数据所用函数
import sys
import io
import torch.nn as nn
# from model.DeeplabV3Plus import Deeplabv3plus_res50, Deeplabv3plus_res101, Deeplabv3plus_vitbase
# from model.FCN_ResNet import FCN_ResNet
# from model.vit_model import vit_fcn_model
# from model.HRNet import HighResolutionNet
#from model.pspnet.pspnet import PSPNet

#from model.segformer.segformer import SegFormer
#from model.ResNet import ResNet
#from model.MAC.MACUNet import MACUNet
#from model.Upernet import UPerNet


from model.zyp.FRCFNet import FRCFNet

#from model.zyp.UNet_GMFR import UNet_GMFR
#from model.zyp.UNet_GMFR import UNet_GMFR
#from model.UNet_MCIF import UNet_MCIF 
#from model.unet import UNet
#from model.HRNet import HRNet
#from model.MANet import MANet
#from model.ResNet import ResNet
#from model.segformer.segformer import SegFormer
#from model.DCBPNet.DCBPNet import DCBP
#from model.FRCFNet import FRCFNet
#from model.AlageNet import AlageNet
# 创建一个新的stdout对象，并将其编码设置为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  #自己加


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#超参数
num_classes = 2
#batch_size = 16
batch_size = 8#自改
start_ep = 0
end_ep = 200
epoches = end_ep - start_ep
save_epoch = 10  #每隔save_epoch个epoch 保存一个模型
#model_name = "segformer"
#model_name = "ResNet"
#model_name = "RCFSNet" 
#model_name = "FRCFNet"
#model_name = "AlageNet"
#model_name = "RCFSNet"
#model_name = "Upernet"

#model_name = "DCBPNet"


model_name = "zyp"
#model_name = "UNet_GMFR"
#model_name = "UNet_MCIF"
#model_name = "unet"
#model_name = "HRNet"
#model_name = "MANet"
#model_name = "ResNet"
#model_name = "segformer"
#model_name = "MACUNet"
#保存路径

save_path = r"/zyp/maweizao/save/"+model_name+"/"

if os.path.exists(save_path) == False:
    os.makedirs(save_path)

# 读取数据


# 马尾藻数据集
imagePath = r"/zyp/maweizao/data/image"  #图像路径
labelPath = r"/zyp/maweizao/data/label"  #真值路径

Dataset = MyDataset(imagePath, labelPath)
# 划分训练集和验证集
print(len(Dataset))
train_len = int(0.9 * len(Dataset))
val_len = len(Dataset) - train_len
trainDataset, valDataset = random_split(Dataset, [train_len, val_len])

trainDatasetloader = DataLoader(trainDataset, batch_size, shuffle=True)
valDatasetloader = DataLoader(valDataset, batch_size, shuffle=True)
trainLen = len(trainDatasetloader)
valLen = len(valDatasetloader)
print(f'the lenth of trainDatasetloader: {trainLen}\n')
print(f'the lenth of valDatasetloader: {valLen}')

# 初始化早停参数
patience = 20
best_val_loss = float('inf')
counter = 0

#构建模型、优化器、损失
if model_name == "unet":
    model = UNet(num_classes=num_classes, in_channels=4).to(device)
elif model_name == "AlageNet":
    model = AlageNet(band_num=4, class_num=2).to(device)
elif model_name == "MACUNet":
    model = MACUNet(band_num=4, class_num=2).to(device)
elif model_name == "UNet_GMFR":
    model = UNet_GMFR(band_num=4, class_num=2).to(device)  
elif model_name == "UNet_MCIF":
    model = UNet_MCIF(band_num=4, class_num=2).to(device)
#elif model_name == "zyp":
    #model = UNet_GMFR(band_num=4, class_num=2).to(device)
elif model_name == "FRCFNet":
    model = FRCFNet(band_num=4, class_num=2).to(device)
  
elif model_name == "zyp":
    model = FRCFNet(band_num=4, class_num=2).to(device)

elif model_name == "RCFSNet":
    model = RCFSNet(num_classes=2, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                base_channel=32).to(device)
elif model_name == "ResNet":
    model = ResNet (blocks=[3, 4, 6, 3], num_classes=2).to(device)
elif model_name == "MANet":
    model = MANet(num_channels=4, num_classes=2)
elif model_name == "HRNet":
    model =HRNet(num_classes=2).to(device)
elif model_name == "DCBPNet":
    model =DCBP(num_classes = 2, backbone = 'hrnetv2_w18', pretrained = False).to(device)
    
elif model_name == "deeplabv3p":
    model = Deeplabv3plus_res50(num_classes=num_classes, os=16, pretrained=False).to(device)
elif model_name == "fcn_resnet":
    model = FCN_ResNet(num_classes=num_classes, backbone='resnet50').to(device)
elif model_name == "deeplabv3p_resnet101":
    model = Deeplabv3plus_res101(num_classes=num_classes, os=16, pretrained=False).to(device)
elif model_name == "deeplabv3p_vitbase":
    model = Deeplabv3plus_vitbase(num_classes=num_classes, image_size=256, os=16, pretrained=True).to(device)
elif model_name == "vit_fcn":
    model = vit_fcn_model(num_classes=num_classes, pretrained=True).to(device)

elif model_name == "Upernet":
    model = UPerNet(num_classes=num_classes,image_size=128).to(device)
elif model_name == "upernet_vit":
    model = UPerNet(num_classes=num_classes, image_size=256).to(device)
elif model_name == "segformer":
    model = SegFormer(num_classes=2).to(device)
elif model_name == "pspnet":
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, num_classes=num_classes, zoom_factor=1, use_ppm=True).to(device)
else:
    print("请选择一个模型！")
model = model.to(device)  # 将模型迁移到指定设备
#image = image.to(device)  # 将输入数据迁移到指定设备
#output = model(image)  # 进行前向传播

#加载权重文件
weights_file = ""

if weights_file != "":
    model.load_state_dict(torch.load(weights_file))
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
loss = CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001, betas=(0.9, 0.95))

#开始训练
for epoch in range(epoches):
    # 训练阶段
    model.train()
    epoch = start_ep
    print(
        f"\n----------------------------------------------epoch: {epoch}----------------------------------------------")
    loss_total = 0
    for i,data in enumerate(trainDatasetloader):
        
        #image, label = data
        image, label = data
        image = image.to(device)
        label = label.to(device)
        #梯度清零
        optimizer.zero_grad()
        output = model(image)
        loss_step = loss(output, label.long())
        print("\r train: epoch: {}, step: {}/{},  loss: {}  ".format(epoch, i, trainLen,loss_step ,end=''))
        loss_step.backward()
        optimizer.step()
        loss_total += loss_step
    ep = epoch
    loss_epoch = loss_total/trainLen
    print("\r epoch: {}, epoch_loss: {}".format(epoch, round(float(loss_epoch), 8)), end='')

    # 验证阶段
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for i, data in enumerate(valDatasetloader):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss_step = loss(output, label.long())
            
            val_loss_total += loss_step
    val_loss_epoch = val_loss_total / len(valDatasetloader)
    print("\n\r epoch: {}, val_loss: {}".format(epoch, round(float(val_loss_epoch), 8)), end='')

    # # 早停检查
    # if val_loss_epoch < best_val_loss:
    #     best_val_loss = val_loss_epoch
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Early stopping")
    #         break
    #保存模型
    weight_path =save_path+"epcho_"+str(epoch)+"_loss_"+str(round(float(loss_epoch), 8))+".pt"
    if (epoch + 1) % save_epoch == 0 :
        print("aaaaaaaaaaaaaaaaa")
        torch.save(model.state_dict(),weight_path)
    # 最后10个epoch： 每个epoch保存一个
    start_ep = start_ep+1
