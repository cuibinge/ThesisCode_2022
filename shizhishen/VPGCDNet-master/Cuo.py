import os
import numpy as np
from PIL import Image

# 定义文件夹路径
pred_dir = r'My_method\001'
gt_dir = r'test\gt01'
save_dir = r'New_method\002'

# 创建保存文件夹
os.makedirs(save_dir, exist_ok=True)
# 设置阈值，超过阈值的像素值将被转换为1，低于或等于阈值的将被转换为0
threshold = 127
# 遍历预测图文件夹中的所有文件
for filename in os.listdir(pred_dir):
    # if filename.endswith(".png") or filename.endswith(".jpg"):
    if filename.endswith(".png"):
        # 读取预测图和真值图
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)

        pred = np.array(Image.open(pred_path).convert('L'))
        gt = np.array(Image.open(gt_path).convert('L'))

        # 将像素值从0-255转换为0和1
        pred = (pred > threshold).astype(np.uint8)
        gt = (gt > threshold).astype(np.uint8)
        # 创建错误图，初始化为黑色（无错误）
        error_map = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        # 正确提取：真值为1，预测为1 -> 白色
        error_map[(gt == 1) & (pred == 1)] = [255, 255, 255]

        # 漏提：真值为1，预测为0 -> 红色
        error_map[(gt == 1) & (pred == 0)] = [255, 0, 0]

        # 误提：真值为0，预测为1 -> 绿色
        error_map[(gt == 0) & (pred == 1)] = [0, 255, 0]

        # 保存错误图c
        save_path = os.path.join(save_dir, filename)
        Image.fromarray(error_map).save(save_path)
