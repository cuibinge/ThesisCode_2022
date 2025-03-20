import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb


def generate_superpixels_rgb(image_path, scale=100, sigma=0.5, min_size=50, show_results=True):
    """
    使用 Felzenszwalb 图分割生成超像素，并输出 24 位 RGB 格式的图像。

    参数：
        image_path (str): 输入图像路径。
        scale (int): 控制超像素大小的尺度参数（值越高，超像素越大）。
        sigma (float): 高斯模糊的标准差，用于平滑图像。
        min_size (int): 控制最小分割区域的像素数量。
        show_results (bool): 是否显示原图和超像素分割结果。

    返回：
        rgb_image (ndarray): 24位 RGB 格式的超像素分割图。
    """
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 Felzenszwalb 图分割算法生成超像素
    segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size).astype(np.int32)

    # 将超像素整数标签映射为 24 位 RGB 格式
    r = (segments & 0xFF).astype(np.uint8)  # 取低 8 位
    g = ((segments >> 8) & 0xFF).astype(np.uint8)  # 取次高 8 位
    b = ((segments >> 16) & 0xFF).astype(np.uint8)  # 取最高 8 位
    rgb_image = np.stack((r, g, b), axis=-1)  # 合并为 RGB 图像

    if show_results:
        # 显示原图与超像素分割结果
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title("原始图像")
        axes[0].axis("off")

        axes[1].imshow(rgb_image)
        axes[1].set_title("超像素分割图（24位 RGB）")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    return rgb_image


def process_dataset_rgb(dataset_dir, output_dir, scale=100, sigma=0.5, min_size=50):
    """
    对数据集中所有图像生成 24 位 RGB 格式的超像素分割，并保存结果。

    参数：
        dataset_dir (str): 输入图像文件夹路径。
        output_dir (str): 超像素分割结果保存路径。
        scale, sigma, min_size: 控制超像素分割的参数（传递给 Felzenszwalb 算法）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(dataset_dir, image_file)
        output_path = os.path.join(output_dir, image_file)  # 输出图像文件名与输入文件一致

        # 生成 24 位 RGB 格式的超像素分割图
        rgb_image = generate_superpixels_rgb(image_path, scale, sigma, min_size, show_results=False)

        # 保存结果为 24 位 RGB 图像
        rgb_image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # 转为 BGR 以兼容 OpenCV 保存
        cv2.imwrite(output_path, rgb_image_bgr)

        print(f"超像素分割完成：{image_file} -> {output_path}")


# 示例：对图像文件夹中的所有图像生成 24 位 RGB 超像素分割图
dataset_dir = "/chenruipeng/weakly/WRTNet/WRTNet/data/train/img"  # 输入图像文件夹路径
output_dir = "/chenruipeng/weakly/WRTNet/WRTNet/data/train/superpixel"  # 保存超像素分割结果的目标文件夹

process_dataset_rgb(dataset_dir, output_dir, scale=100, sigma=0.5, min_size=50)
