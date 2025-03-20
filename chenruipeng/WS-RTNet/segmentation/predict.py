# -*- coding: UTF-8 -*-
"""
@Author  ：ChenRuipeng of SDUST
@Date    ：2023/11/13 15:46
"""
import os.path
import torch.nn
import torch.nn as nn
from torchvision import transforms
import imageio
from PIL import Image
import numpy as np
from datetime import datetime
from mdoel.DeeplabV3Plus import Deeplabv3plus_res50, Deeplabv3plus_res101, Deeplabv3plus_vitbase
from mdoel.FCN_ResNet import FCN_ResNet
from mdoel.vit_model import vit_fcn_model
from mdoel.HRNet import HighResolutionNet
from mdoel.Upernet import UPerNet
from mdoel.segformer.segformer import SegFormer
from mdoel.FTUNetFormer import ft_unetformer
from mdoel.ABCNet import ABCNet
# from mdoel.UNetFormer import UNetFormer
from mdoel.CFCTNet import CTCFNet
from mdoel.SFFNet.SFFNet import SFFNet
import glob
from datetime import datetime
from util import Logger
from pathlib import Path
import sys
import io
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf8",line_buffering=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

now = datetime.now()
now = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

def estimate(y_gt, y_pred):
    y_gt = np.asarray(y_gt, dtype=np.bool)
    y_pred = np.asarray(y_pred, dtype=np.bool)
    # Accuracy
    acc = np.mean(np.equal(y_gt, y_pred))
    # IOU
    # 计算交集和并集
    intersection = np.logical_and(y_gt, y_pred)
    union = np.logical_or(y_gt, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    # Recall
    # 计算真阳性（True Positive）和假阴性（False Negative）
    tp = np.sum(np.logical_and(y_gt, y_pred))
    fn = np.sum(np.logical_and(y_gt, np.logical_not(y_pred)))
    fp = np.sum(np.logical_and(np.logical_not(y_gt), y_pred))
    # 计算召回率
    recall = tp / (tp + fn)
    # 计算精确率
    precision = tp / (tp + fp)
    # 计算F1-score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return acc, iou, recall, precision, f1, y_pred

def readimage(dir):
    images_path_list = glob.glob(os.path.join(dir, '*.png'))
    return images_path_list

def readlabel(dir):
    labels_path_list = glob.glob(os.path.join(dir, '*.png'))
    return labels_path_list

def model_predict(model, img_data, lab_data, img_size):

    row, col, dep = img_data.shape

    if row % img_size != 0 or col % img_size != 0:
        print('{}: Need padding the predict image...'.format(datetime.now().strftime('%c')))
        # 计算填充后图像的 hight 和 width
        padding_h = (row // img_size + 1) *img_size
        padding_w = (col // img_size + 1) *img_size
    else:
        print('{}: No need padding the predict image...'.format(datetime.now().strftime('%c')))
        # 不填充后图像的 hight 和 width
        padding_h = (row // img_size) *img_size
        padding_w = (col // img_size) *img_size

    # 初始化一个 0 矩阵，将图像的值赋值到 0 矩阵的对应位置
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='float32')
    padding_img[:row, :col, :] = img_data[:row, :col, :]

    #初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
    padding_pre = np.zeros((padding_h, padding_w), dtype='uint8')

    # 对 img_size * img_size 大小的图像进行预测
    count = 0  # 用于计数
    for i in list(np.arange(0, padding_h, img_size)):
        if (i + img_size) > padding_h:
            continue
        for j in list(np.arange(0, padding_w, img_size)):
            if (j + img_size) > padding_w:
                continue

            # 取 img_size 大小的图像，在第一维添加维度，变成四维张量，用于模型预测
            img_data_ = padding_img[i:i+img_size, j:j+img_size, :]
            toTensor = transforms.ToTensor()
            img_data_ = toTensor(img_data_)
            img_data_ = img_data_[np.newaxis, :, :, :]
            # img_data_ = np.transpose(img_data_, (0, 3, 1, 2))
            # img_data_.float()
            # 预测，对结果进行处理
            # print(img_data_.shape)
            y_pre = model.forward(img_data_.to(device))
            # print(y_pre)
            # y_pre = model.predict(img_data_)
            # y_pre = y_pre[0]
            y_pre = np.squeeze(y_pre, axis = 0)
            y_pre = torch.argmax(y_pre, axis = 0)
            # y_pre = y_pre.astype('uint8')

            # 将预测结果的值赋值到 0 矩阵的对应位置
            padding_pre[i:i+img_size, j:j+img_size] = y_pre[:img_size, :img_size].cpu()
            count += 1


            print('\r{}: Predited {:<5d}({:<5d})'.format(datetime.now().strftime('%c'), count, int((padding_h/img_size)*(padding_w/img_size))), end='')

    # 评价指标
    acc, iou, recall, precision, f1, y_pred = estimate(lab_data, padding_pre[:row, :col])

    return acc, iou, recall, precision, f1, y_pred

#参数
num_classes = 2
os.system("ls")
image_size = 64
modelname = "sffnet"
imagedir =  r"/root/neSeg/data/test_fanhua/img"  #测试的图像
labeldir = r"/root/neSeg/data/test_fanhua/gt"  #真值
modelPath = r"save_model_fanhua_"+modelname+"mask" #模型路径
print(modelPath)
modelPath= glob.glob(os.path.join(modelPath, '*.pt'))
print(f"模型数量为：{len(modelPath)}")

savePath = r"/root/neSeg/save_res_fanhua/"+modelname  #结果保存路径
log_path =  r"/root/neSeg/save_res_fanhua/" + r"/"+modelname +r"/"+ now+ ".log" #log文件

if os.path.exists(savePath) == False:
    os.makedirs(savePath)
#日志文件
f = open(log_path, 'w')
f.close()
log = Logger(log_path, level='debug')
log.logger.info('Start! Train image size  ' + str(image_size))

imagelist = sorted(readimage(imagedir))
labellist = sorted(readlabel(labeldir))

print(f"测试数量为：{len(imagelist)}")
# 加载模型
if modelname == "UNet":
    model = UNet(num_classes=num_classes, in_channels=3).to(device)
elif modelname == "UNet_new":
    model = UNet(3,2).to(device)
elif modelname == "deeplabv3p":
    model = Deeplabv3plus_res50(num_classes=num_classes, os=16, pretrained=False).to(device)
elif modelname == "deeplabv3p_resnet101":
    model = Deeplabv3plus_res101(num_classes=num_classes, os=16, pretrained=False).to(device)
elif modelname == "fcn_resnet":
    model = FCN_ResNet(num_classes=num_classes, backbone='resnet50').to(device)
elif modelname == "deeplabv3p_vitbase":
    model = Deeplabv3plus_vitbase(num_classes=num_classes, image_size=128, os=16, pretrained=False).to(device)
elif modelname == "vit_fcn":
    model = vit_fcn_model(num_classes=num_classes, pretrained=False).to(device)
elif modelname == "hrnet":
    model = HighResolutionNet(num_classes=num_classes).to(device)
elif modelname == "upernet":
    model = UPerNet(num_classes=num_classes).to(device)
elif modelname == "segformer":
    model = SegFormer(num_classes=num_classes).to(device)
elif modelname == "ft_unetformer":
    model = ft_unetformer().to(device)
elif modelname == "unetformer":
    model = UNetFormer(num_classes=2,pretrained=False).to(device)
elif modelname == "cfctnet":
    model = CTCFNet(img_size=256, in_chans=3, class_dim=2,
                    patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm, depths=[3, 3, 6, 3], sr_ratios=[8, 4, 2, 1]).to(device)
elif modelname == "sffnet":
    model = SFFNet().to(device)
elif modelname == "abcnet":
    model = ABCNet(3, 2).to(device)
else:
    print("请选择预测的模型！")
import numpy as np
import torch
import cv2
import os
from pathlib import Path
import imageio

# 循环加载模型并预测
if len(modelPath) != 0:
    for i in range(len(modelPath)):
        print(f'------------------模型: {modelPath[i]} 开始预测！------------------\n\n')
        log.logger.info('modelPath: ' + modelPath[i])
        name = Path(modelPath[i]).stem
        if not os.path.exists(os.path.join(savePath, name)):
            os.makedirs(os.path.join(savePath, name))

        mp = modelPath[i]
        model.load_state_dict(torch.load(mp, map_location=device))
        model.eval()
        
        acc_all = 0
        iou_all = 0
        recall_all = 0
        precision_all = 0
        f1_all = 0
        
        for j in range(len(imagelist)):
            # 读取图像和标签
            image = imageio.imread(imagelist[j])
            B1, B2, B3 = cv2.split(image)
            B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
            B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
            B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')
            # B4_normalization = ((B4 - np.min(B4)) / (np.max(B4) - np.min(B4)) * 1).astype('float32')
            #B4_normalization = ((B4 - np.min(B4)) / (np.max(B4) - np.min(B4)) * 1).astype('float32')
            image = cv2.merge([B1_normalization, B2_normalization, B3_normalization])
            # image = cv2.merge([B3, B2, B1])
            label = imageio.imread(labellist[j])
            # image = image.astype(float)
            # label = label.astype(float)
            # 预测结果
            acc, iou, recall, precision, f1, y_pred = model_predict(model.to(device), image, label, img_size=image_size)
            # print(y_pred)
           # 将标签中的 254 转换为 1，其他保持不变
            label = np.where(label == 254, 1, label)  # 将 254 替换为 1
            label = np.where(label == 255, 1, label)  # 将 254 替换为 1

            # 创建空白的RGB图像来存储结果
            final_image = np.zeros((y_pred.shape[0], y_pred.shape[1], 3), dtype=np.uint8)

            # 确保 y_pred 和 label 是二值化的
            y_pred = np.round(y_pred).astype(np.uint8)  # 四舍五入确保二值化
            label = label.astype(np.uint8)

            # 设置正确预测的像素（白色）
            correct_pred = (y_pred == label) & (y_pred == 1)      # 正确预测
            final_image[correct_pred] = [255, 255, 255]           # 白色

            # 设置背景像素（黑色）
            background = (y_pred == label) & (y_pred == 0)        # 背景像素
            final_image[background] = [0, 0, 0]                   # 黑色

            # 设置漏提的像素（绿色）
            missed_detection = (label == 1) & (y_pred == 0)       # 漏提的像素
            final_image[missed_detection] = [0, 255, 0]           # 绿色

            # 设置错提的像素（红色）
            false_detection = (label == 0) & (y_pred == 1)        # 错提的像素
            final_image[false_detection] = [255, 0, 0]            # 红色

            # 保存最终图像
            img_name = Path(imagelist[j]).stem
            save = os.path.join(savePath, name, img_name + ".png")
            imageio.imwrite(save, final_image)

            # 统计评价指标
            acc_all += acc
            iou_all += iou
            recall_all += recall
            precision_all += precision
            f1_all += f1

            # 记录日志
            log.logger.info(f"{img_name}的准确率： {acc}")
            log.logger.info(f"{img_name}的IOU： {iou}")
            log.logger.info(f"{img_name}的召回率： {recall}")
            log.logger.info(f"{img_name}的精度： {precision}")
            log.logger.info(f"{img_name}的F1_score： {f1}")
        
        # 平均统计
        log.logger.info(f"平均准确率： {acc_all / len(imagelist)}")
        log.logger.info(f"平均IOU： {iou_all / len(imagelist)}")
        log.logger.info(f"平均召回率： {recall_all / len(imagelist)}")
        log.logger.info(f"平均精度： {precision_all / len(imagelist)}")
        log.logger.info(f"平均F1_score： {f1_all / len(imagelist)}")
