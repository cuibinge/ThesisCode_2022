# -*- coding: UTF-8 -*-
"""
@Author  ：zhangyanping of SDUST
@Date    ：2024/1/2 18:57
"""
import os.path
import torch.nn
from torchvision import transforms
import imageio
from PIL import Image
import numpy as np
from datetime import datetime
# from mdoel.FCN_ResNet import FCN_ResNet
# from mdoel.vit_model import vit_fcn_model
# from mdoel.HRNet import HighResolutionNet
#from model.Upernet import UPerNet
#from model.segformer.segformer import SegFormer
#from model.unet import UNet
from model.ResNet import ResNet
import glob
from datetime import datetime
from util import Logger
from pathlib import Path
import sys
import io
import cv2
import rasterio
from rasterio import Affine
#from model.FRCFNet.FRCFNet import FRCFNet
#from model.MAC.MACUNet import MACUNet
#from model.RCFSNet.RCFSNet import RCFSNet
#from model.MANet import MANet
#from model.DCBPNet.DCBPNet import DCBP

#from model.zyp.FRCFNet import FRCFNet
#from model.zyp.UNet_GMFR import UNet_GMFR
#from model.UNet_MCIF import UNet_MCIF
#from model.HRNet import HRNet
#from model.unet import UNet
#from model.MANet import MANet
#from model.segformer.segformer import SegFormer

import torch.nn as nn
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
    if np.sum(union)!=0:
        iou = np.sum(intersection) / np.sum(union)
    else:
        iou=0.0
    # Recall
    # 计算真阳性（True Positive）和假阴性（False Negative）
    tp = np.sum(np.logical_and(y_gt, y_pred))
    fn = np.sum(np.logical_and(y_gt, np.logical_not(y_pred)))
    fp = np.sum(np.logical_and(np.logical_not(y_gt), y_pred))
     # 计算召回率
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    # 计算F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return acc, iou, recall, precision, f1, y_pred
def readimage(dir):
    images_path_list = glob.glob(os.path.join(dir, '*.tif'))
    return images_path_list

def readlabel(dir):
    labels_path_list = glob.glob(os.path.join(dir, '*.png'))
    return labels_path_list

def model_predict(model, img_data, lab_data, img_size):
    img_data = torch.permute(img_data, (1, 2, 0))
    row, col, dep = img_data.shape

    if row % img_size != 0 or col % img_size != 0:
        print('{}: Need padding the predict image...'.format(datetime.now().strftime('%c')))
        # 计算填充后图像的 hight 和 width
        padding_h = (row // img_size + 1) *img_size
        padding_w = (col // img_size + 1) *img_size
    else:
        #print('{}: No need padding the predict image...'.format(datetime.now().strftime('%c')))
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

            # 预测，对结果进行处理
            y_pre, = model.forward(img_data_.to(device))
            # y_pre = model.predict(img_data_)
            y_pre = np.squeeze(y_pre, axis = 0)
            y_pre = torch.argmax(y_pre, axis = 0)
            # y_pre = y_pre.astype('uint8')

            # 将预测结果的值赋值到 0 矩阵的对应位置
            padding_pre[i:i+img_size, j:j+img_size] = y_pre[:img_size, :img_size].cpu()
            count += 1


            #print('\r{}: Predited {:<5d}({:<5d})'.format(datetime.now().strftime('%c'), count, 
                                                        # int((padding_h/img_size)*(padding_w/img_size))), end='')

    # 评价指标
    acc, iou, recall, precision, f1, y_pred = estimate(lab_data, padding_pre[:row, :col])

    return acc, iou, recall, precision, f1, y_pred

#参数
num_classes = 2
os.system("ls")
image_size = 128
#modelname = "MANet"
#modelname = "Upernet"
#modelname = "RCFSNet"
#modelname = "MACUNet"
#modelname = "unet"
#modelname = "FRCFNet"
#modelname = "DCBPNet"

#modelname = "zyp"
#modelname = "UNet_GMFR"
#modelname = "UNet_MCIF"
#modelname = "HRNet"
#modelname = "unet"
#modelname = "MANet"
#modelname = "segformer"
modelname = "ResNet"

ndvi = False

#自改
imagedir = r"/zyp/maweizao/test/img"  #测试的图像
labeldir = r"/zyp/maweizao/test/lab" #真值

modelPath = r"/zyp/maweizao/save/"+modelname #模型路径


modelPath= glob.glob(os.path.join(modelPath, '*.pt'))
print(f"模型数量为：{len(modelPath)}")

savePath = r"/zyp/maweizao/save_re/"+modelname  #结果保存路径
log_path = r"/zyp/maweizao/save_re/" + r"/"+modelname +r"/"+ now+ ".log" #log文件

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
if modelname == "unet":
    model = UNet(num_classes=num_classes, in_channels=4).to(device)
elif modelname == "UNet_MCIF":
    model = UNet_MCIF(band_num=4, class_num=2).to(device) 
elif modelname == "UNet_GMFR":
    model = UNet_GMFR(band_num=4, class_num=2).to(device)   
       
elif modelname == "zyp":
    model = FRCFNet(band_num=4, class_num=2).to(device)    
elif modelname == "MACUNet":
    model = MACUNet(band_num=4, class_num=2).to(device)     
elif modelname == "RCFSNet":
    model = RCFSNet(num_classes=2, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                base_channel=16).to(device)    
elif modelname == "MANet":
    model = MANet(num_channels=4, num_classes=2)
elif modelname == "HRNet":
    model =HRNet(num_classes=2).to(device)
elif modelname == "DCBPNet":
    model =DCBP(num_classes = 2, backbone = 'hrnetv2_w18', pretrained = False).to(device)
    
elif modelname == "ResNet":
    model = ResNet (blocks=[3, 4, 6, 3], num_classes=2).to(device)
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
elif modelname == "Upernet":
    model = UPerNet(num_classes=num_classes).to(device)
elif modelname == "segformer":
    model = SegFormer(num_classes=num_classes).to(device)
else:
    print("请选择预测的模型！")
if len(modelPath) != 0:
    for i in range(len(modelPath)):
        print(f'------------------模型: {modelPath[i]} 开始预测！------------------\n\n')
        log.logger.info('modelPath: ' + modelPath[i])
        name = Path(modelPath[i]).stem

        if os.path.exists(os.path.join(savePath, name)) == False:
            os.makedirs(os.path.join(savePath, name))

        mp = modelPath[i]
        model.load_state_dict(torch.load(mp, map_location=device))
        model.eval()

        acc_all = 0
        iou_all = 0
        recall_all = 0
        precision_all = 0
        f1_all = 0

        for i in range(len(imagelist)):
            print(imagelist[i])
            image = imageio.imread(imagelist[i])
            m, n, d = image.shape
            print(image.shape)
          
            B1, B2, B3, B4 = cv2.split(image)

            if ndvi:
                B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
                B1_normalization[np.isnan(B1_normalization)] = 0

                B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
                B2_normalization[np.isnan(B2_normalization)] = 0

                B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')
                B3_normalization[np.isnan(B3_normalization)] = 0

                NDVI = (B4 - B3) / ((B4 + B3) + 1e-6)
                NDVI[(NDVI > 1) | (NDVI < -1)] = 0

                NDVI_normalization = ((NDVI - np.min(NDVI)) / (np.max(NDVI) - np.min(NDVI))).astype('float32')
                NDVI_normalization[np.isnan(NDVI_normalization)] = 0
                #         # #计算NDVI 和 NDWI
                # image = cv2.merge([B1, B2, B3, NDVI])
                image = cv2.merge([B1_normalization, B2_normalization, B3_normalization, NDVI_normalization])
            #         # NDWI = (B2 - B4) / (B2 + B4)

            else:
                image = cv2.merge([B1, B2, B3, B4])

           # print(f"image.shp{image.shape}")

            label = imageio.imread(labellist[i]) / 255  # 1
            # 预测结果
            # acc, iou, recall, precision, f1, y_pred = model_predict(model.to(device), image, label, img_size=image_size)
           # print(image.max(), image.min())

            image = np.array(image).astype('float32')
            # label = (np.array(label) / 255).astype('int64')
            trans = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize()
            ])

            # y_pred = model_predict(model.to(device), std_lize(image), img_size=image_size)
            acc, iou, recall, precision, f1, y_pred = model_predict(model.to(device), trans(image),
                                                                    lab_data=label,
                                                                    img_size=image_size)

            #print(y_pred.shape)
          
            # 创建一个 RGB 彩色图像s

            img_color = np.zeros((y_pred.shape[0], y_pred.shape[1], 3), dtype=np.uint8)

            # # 预测正确的像素保持不变
            correct_pixels = (y_pred == 1)
            img_color[correct_pixels] = [0, 255, 0]  # RGB for white

            # 误提的像素赋值红色
            false_alarm_pixels = (label == 0) & (y_pred == 1)
            img_color[false_alarm_pixels] = [255, 0, 0]

            # 漏提的像素赋值白色
            false_alarm_pixels = (label == 1) & (y_pred == 0)
            img_color[false_alarm_pixels] = [255, 255, 255]
            # 保存彩色图像
            img_name = Path(imagelist[i]).stem
            save = os.path.join(savePath, name, img_name + ".tif")
            print(save)
            imageio.imwrite(save, img_color)

            acc_all += acc
            iou_all += iou
            recall_all += recall
            precision_all += precision
            f1_all += f1
            log.logger.info(f"{img_name}的准确率： {acc}")
            log.logger.info(f"{img_name}的IOU： {iou}")
            log.logger.info(f"{img_name}的召回率： {recall}")
            log.logger.info(f"{img_name}的精度： {precision}")
            log.logger.info(f"{img_name}的F1_score： {f1}")
            
            """ 
            #地理坐标
            dataset = gdal.Open(imagelist[i])  # 提供地理坐标信息和几何信息的栅格底图
            projection = dataset.GetProjection()
            transform = dataset.GetGeoTransform()
            """
            """
            # 打开图像文件
            with rasterio.open(imagelist[i]) as dataset:
                projection= dataset.crs
                # 获取地理变换信息
                transform = dataset.transform 
            savePath_o = r"/zyp/maweizao/save_re/zuobiao/" + r"/" + modelname + r"/" + now + "/" + name + '/'
            log_path = r"/zyp/maweizao/save_re/" + r"/" + modelname + r"/" + now + ".log"
            if not os.path.exists(savePath_o):
                os.makedirs(savePath_o)
            savePath_o = savePath_o + img_name + ".tif"
            #arr2raster(y_pred, savePath_o, prj=projection, trans=transform)"""
            savePath_o = r"/zyp/Rmaweizao/save_re/zuobiao/" + r"/" + modelname + r"/" + now + "/" + name + '/'
            log_path = r"/zyp/Rmaweizao/save_re/" + r"/" + modelname + r"/" + now + ".log"
            # 假设 y_pred 是一个 NumPy 数组，savePath_o 是要保存的文件路径
            y_pred = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            savePath_o = savePath_o + img_name + ".tif"

            # 定义地理变换和投影信息
            transform = Affine.translation(y_pred.shape[1] / 2, y_pred.shape[0] / 2)
            projection = {'init': 'epsg:4326'}  # 使用 WGS84 坐标系

            # 创建元数据
            meta = {
                'driver': 'GTiff',
                'dtype': y_pred.dtype,
                'nodata': None,
                'width': y_pred.shape[1],
                'height': y_pred.shape[0],
                'count': 1,
                'crs': projection,
                'transform': transform
             }           
        print("\n")
        log.logger.info("---------------------最终结果----------------------\n")
        log.logger.info(f"平均准确率： {acc_all / (len(imagelist))}")
        log.logger.info(f"平均iou： {iou_all / (len(imagelist))}")
        log.logger.info(f"平均召回率： {recall_all / (len(imagelist))}")
        log.logger.info(f"平均精度： {precision_all / (len(imagelist))}")
        log.logger.info(f"平均F1_score： {f1_all / (len(imagelist))}")
else:
    print(f'------------------模型开始预测！------------------\n\n')
