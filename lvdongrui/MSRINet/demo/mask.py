# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import glob
import multiprocessing as mp
import os
import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import sys
sys.path.append('..')
from cat_seg import add_cat_seg_config
from predictor import VisualizationDemo

def convert_sem_seg_to_color_mask(sem_seg, metadata, output_path=None):
    """
    将 semantic segmentation 预测结果转换为彩色掩码图
    
    参数:
    sem_seg: numpy.ndarray, 语义分割预测结果 (H, W)
    metadata: MetadataCatalog object, 包含 stuff_colors 等元数据
    output_path: str, 可选的输出文件路径，如果提供则保存图片
    
    返回:
    colored_mask: numpy.ndarray, 彩色掩码图 (H, W, 3) in BGR format
    """
    # 获取颜色映射
    stuff_colors = metadata.stuff_colors  # 列表形式，每个元素是 (R, G, B)
    
    # 初始化输出彩色掩码数组
    height, width = sem_seg.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 将每个像素值映射到对应的颜色
    for class_id in np.unique(sem_seg):
        # 检查 class_id 是否在 stuff_colors 的范围内
        if class_id < len(stuff_colors):
            # 获取对应的颜色 (RGB)
            color_rgb = stuff_colors[class_id]
            # 转换为 BGR 格式 (因为 OpenCV 使用 BGR)
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            # 将该类别的像素设置为对应的颜色
            colored_mask[sem_seg == class_id] = color_bgr
    
    # 如果提供了输出路径，则保存图片
    if output_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, colored_mask)
    
    return colored_mask
    
# constants
WINDOW_NAME = "MaskFormer demo"

def setup_cfg(config_file, opts):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    config_file = "/root/CAT-Seg/configs/vitb_GID5.yaml"
    input_paths = "GID_show/*.png"
    output_dir = "output/"
    confidence_threshold = 0.5
    opts = ["MODEL.WEIGHTS", "/root/CAT-Seg/GID_1w_5k/model_final.pth"]

    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info(f"Config file: {config_file}")
    logger.info(f"Input paths: {input_paths}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"Opts: {opts}")

    cfg = setup_cfg(config_file, opts)
    demo = VisualizationDemo(cfg)

    # 获取数据集元数据（包含正确的颜色映射）
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else "__unused")

    input_files = glob.glob(os.path.expanduser(input_paths))
    assert input_files, "No input files found in the specified path!"

    for path in tqdm.tqdm(input_files, disable=not output_dir):
        img = read_image(path, format="BGR")
        start_time = time.time()
        
        predictions, visualized_output = demo.run_on_image(img)
        
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
            logger.info(f"{path}: sem_seg min: {sem_seg.min()}, max: {sem_seg.max()}")
            logger.info(f"{path}: unique classes: {np.unique(sem_seg)}")
            
            # 使用自定义函数生成彩色掩码
            if output_dir:
                base_name = os.path.basename(path)
                name, ext = os.path.splitext(base_name)
                out_filename = os.path.join(output_dir, f"{name}_mask{ext}")
                colored_mask = convert_sem_seg_to_color_mask(sem_seg, metadata, output_path=out_filename)
            else:
                colored_mask = convert_sem_seg_to_color_mask(sem_seg, metadata)
        else:
            raise ValueError("Predictions do not contain semantic segmentation output!")

        if not output_dir:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, colored_mask)
            if cv2.waitKey(0) == 27:
                break

    if not output_dir:
        cv2.destroyAllWindows()