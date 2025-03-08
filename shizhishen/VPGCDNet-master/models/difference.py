import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置后端
import torch
from PIL import Image

matplotlib.use('TkAgg')  # 或者 'Agg'


def compute_difference(x1, x2):
    # 确保转换为numpy数组并且数据类型为uint8
    x1 = np.array(x1).astype(np.uint8)
    x2 = np.array(x2).astype(np.uint8)

    differences = []
    thresholds = []

    for i in range(x1.shape[0]):  # 遍历批次中的每一对图像
        # 调整维度从 (C, H, W) 到 (H, W, C)
        img1 = np.transpose(x1[i], (1, 2, 0))  # (C, H, W) -> (H, W, C)
        img2 = np.transpose(x2[i], (1, 2, 0))

        # 检查通道数
        if img1.ndim == 3 and img1.shape[2] == 3:  # RGB
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Invalid number of channels in x1[{i}]: {img1.shape}")

        if img2.ndim == 3 and img2.shape[2] == 3:  # RGB
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Invalid number of channels in x2[{i}]: {img2.shape}")

        # 计算差异图
        difference = cv2.absdiff(gray1, gray2)

        # 应用阈值处理，生成二值图
        _, thresh = cv2.threshold(difference, 240, 255, cv2.THRESH_BINARY)
        difference_rgb = cv2.merge([difference, difference, difference])
        # 将差异图转换为PIL.Image类型
        difference_image = Image.fromarray(difference_rgb)
        # 保存差异图和二值图
        differences.append(difference_image)
        thresholds.append(thresh)

    return differences, thresholds

