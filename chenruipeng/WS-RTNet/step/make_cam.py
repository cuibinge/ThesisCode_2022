from sklearn.manifold import TSNE
import torch
from matplotlib import pyplot as plt
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import os.path as osp

import dataset.dataloader
from utils import torchutils, imutils
import cv2
import imageio

cudnn.enabled = True

from .par import PAR


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        if args.par_refine:
            par = PAR(num_iter=20, dilations=[1, 4, 16, 32, 48, 64])
            par = par.cuda()

        all_features = []
        all_labels = []
        all_confidences = []  # 用于存储每张图片的置信度

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            # 获取更高分辨率的CAM
            strided_up_size = imutils.get_strided_up_size(size, 16)
            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']]
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear', align_corners=False)
                           for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]
            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            refined_cam = highres_cam

            # 使用 PAR 进行细化
            if args.par_refine:
                img = pack['img'][0][0][0][None, ...].cuda()
                refined_cam = refined_cam[None, ...]
                refined_cam = par(img, refined_cam)
                refined_cam = refined_cam[0]
            refined_cam = refined_cam.cpu().numpy()

            # 将原始图像与CAM叠加并保存
            refined_cam_vvv = np.squeeze(refined_cam, axis=0)
            original_image = pack['img'][0][0][0].cpu().numpy().transpose(1, 2, 0)  # 转换为HWC格式
            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())  # 归一化

            # 调整叠加图像大小为256x256
            cam_image_path = f'/chenruipeng/weakly/WRTNet/WRTNet/data/workspace/cams/{img_name}_cam.png'

            # 创建叠加图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(original_image)  # 绘制原图
            ax.imshow(refined_cam_vvv, cmap='jet', alpha=0.5)  # 绘制叠加的CAM
            ax.axis('off')
            
            # 临时保存为高分辨率图像
            temp_path = f'/tmp/{img_name}_temp.png'
            fig.savefig(temp_path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)

            # 调整为256x256大小
            from PIL import Image
            resized_image = Image.open(temp_path).resize((256, 256), Image.ANTIALIAS)
            resized_image.save(cam_image_path)



#             类似的mask生成代码继续保留
            refined_cam = np.pad(refined_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = np.pad(pack['label'][0] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(refined_cam, axis=0)
            cls_labels = keys[cls_labels]

            cls_labels[cls_labels != 1] = 0
            cls_labels[cls_labels == 1] = 255

            cls_labels = cv2.resize(cls_labels, (256, 256))
            imageio.imwrite(os.path.join(args.mask_dir, img_name + '.png'), cls_labels.astype(np.uint8))

        # 计算整体平均置信度
#         overall_mean_confidence = np.mean(all_confidences)
#         overall_std_confidence = np.std(all_confidences)
#         overall_max_confidence = np.max(all_confidences)
#         overall_min_confidence = np.min(all_confidences)

#         # 输出整体平均置信度结果
#         print(f"Overall Mean Confidence: {overall_mean_confidence}")
#         print(f"Overall Std Confidence: {overall_std_confidence}")
#         print(f"Overall Max Confidence: {overall_max_confidence}")
#         print(f"Overall Min Confidence: {overall_min_confidence}")

        
#         print("merging")
#         # 将所有特征和标签合并
#         all_features = np.concatenate(all_features, axis=0)
#         all_labels = np.concatenate(all_labels, axis=0)
#         print(len(all_features))
#         # t-SNE降维
#         tsne = TSNE(n_components=2, random_state=42)
#         features_2d = tsne.fit_transform(all_features)

#         # 保存t-SNE结果
#         tsne_save_dir = '/root/proj/ProCAM/data/workspace/tsne_visualizations'
#         os.makedirs(tsne_save_dir, exist_ok=True)

#         plt.figure(figsize=(8, 8))
#         colors = ['red' if label == 0 else 'green' for label in all_labels]
#         plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, s=20)
#         # plt.colorbar()
#         plt.title(f't-SNE visualization (Process {process_id})')
#         plt.savefig(os.path.join(tsne_save_dir, f'tsne_result_{process_id}.png'), bbox_inches='tight', pad_inches=0)
#         plt.close()


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    data = dataset.dataloader.SeafogClassificationDatasetMSF(args.eval_img_list, data_root=args.data_root,
                                                             resize_long=(256, 256), scales=args.cam_scales)
    data = torchutils.split_dataset(data, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, data, args), join=True)
    print(']')

    torch.cuda.empty_cache()
