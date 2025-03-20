# -*- coding: UTF-8 -*-
"""
@Author  ：ChenRuipeng of SDUST
@Date    ：2023/11/13 15:46
"""
import os
import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from dataset.TT_Dataset  import MyDataset   # 读取数据所用函数
from mdoel.SFFNet.SFFNet import SFFNet
from mdoel.unet import UNet
# from mdoel.UNet.unet_model import UNet
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#超参数
num_classes = 2
batch_size = 32
start_ep = 0
end_ep = 100
epoches = end_ep - start_ep
save_epoch = 2  #每隔save_epoch个epoch 保存一个模型
model_name = "sffnet"

#保存路径
# save_path = r"/chenruipeng/RedTideDetection/save_model_redtide/"+model_name+"/"
#save_path = r"/chenruipeng/RedTideDetection/save_model_redtide/"+model_name+"/"

save_path = r"/root/neSeg/save_model_fanhua_"+model_name+"mask"+"/"

if os.path.exists(save_path) == False:
    os.makedirs(save_path)



# # 赤潮数据集
imagePath = r"/root/neSeg/data/train_fanhua/img"  #图像路径
labelPath = r"/root/neSeg/data/train_fanhua/mask"  #真值路径

# imagePath = r"./data_newest/img"  #图像路径
# labelPath = r"./data_newest/label"  #真值路径

Dataset = MyDataset(imagePath, labelPath)
# 划分训练集和验证集
# train_len = int(1 * len(Dataset))
# val_len = len(Dataset) - train_len
# trainDataset, valDataset = random_split(Dataset, [train_len, val_len])

trainDatasetloader = DataLoader(Dataset, batch_size, shuffle=True,pin_memory=True, num_workers=10)
# valDatasetloader = DataLoader(valDataset, batch_size, shuffle=True,pin_memory=True, num_workers=10)
trainLen = len(trainDatasetloader)
# valLen = len(valDatasetloader)
print(f'the lenth of trainDatasetloader: {trainLen}\n')
# print(f'the lenth of valDatasetloader: {valLen}')

# 初始化早停参数
patience = 20
best_val_loss = float('inf')
counter = 0

# SFFNet
model = SFFNet().to(device)
# model = UNet(2,3).to(device)

#加载权重文件
weights_file = ""
if weights_file != "":
    model.load_state_dict(torch.load(weights_file))
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
loss_ce = CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001, betas=(0.9, 0.95))

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
        # print(image.shape)
        image = image.to(device)
        label = label.to(device)
        #梯度清零

        output = model(image)
        loss  = loss_ce(output,label.long())
        loss_step =loss
        optimizer.zero_grad()
        print("\r train: epoch: {}, step: {}/{}, loss: {} ".format(epoch, i, trainLen,loss_step, end=''))
        loss_step.backward()
        optimizer.step()
        loss_total += loss_step
    ep = epoch
    loss_epoch = loss_total/trainLen
    print("\r epoch: {}, epoch_loss: {}".format(epoch, round(float(loss_epoch), 8)), end='')

    # 验证阶段
#     model.eval()
#     val_loss_total = 0
#     with torch.no_grad():
#         for i, data in enumerate(valDatasetloader):
#             image, label = data
#             image = image.to(device)
#             label = label.to(device)
#             output = model(image)

#             loss_step = loss(output, label.long())
#             val_loss_total += loss_step
#     val_loss_epoch = val_loss_total / len(valDatasetloader)
#     print("\n\r epoch: {}, val_loss: {}".format(epoch, round(float(val_loss_epoch), 8)), end='')

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
        torch.save(model.state_dict(),weight_path)
    # 最后10个epoch： 每个epoch保存一个
    start_ep = start_ep+1
