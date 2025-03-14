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

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']]  # b x 2 x w x h
            # print("wo kankan output shape:",outputs)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear', align_corners=False)
                           for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            refined_cam = highres_cam.cpu().numpy()
            cam_nopar = refined_cam
            cam_nopar = cam_nopar
            # 去除第一个维度大小为 1 的维度，使其变成二维数组
            cam_nopar = np.squeeze(cam_nopar, axis=0)

            # 使用 Matplotlib 显示高分辨率CAM
            plt.imshow(cam_nopar, cmap='jet', alpha=0.5)  # 使用 jet colormap，alpha 为透明度
            plt.colorbar()  # 添加颜色条
            plt.savefig(f'/root/proj/ProCAM/data/procam/cam/{img_name}_cam.png')  # 将可视化的CAM保存为图片
            plt.close()  # 关闭Matplotlib的图形窗口，不显示图像

            if args.par_refine:
                img = pack['img'][0][0][0][None, ...].cuda()
                refined_cam = refined_cam[None, ...]
                refined_cam = par(img, refined_cam)
                refined_cam = refined_cam[0]

            # 去除第一个维度大小为 1 的维度，使其变成二维数组
            refined_cam_v = np.squeeze(refined_cam, axis=0)

            # 使用 Matplotlib 显示高分辨率CAM
            plt.imshow(refined_cam_v, cmap='jet', alpha=0.5)  # 使用 jet colormap，alpha 为透明度
            plt.colorbar()  # 添加颜色条
            plt.savefig(f'/root/proj/ProCAM/data/procam/cam_r/{img_name}_cam_r.png')  # 将可视化的CAM保存为图片
            plt.close()  # 关闭Matplotlib的图形窗口，不显示图像

            refined_cam = np.pad(refined_cam, ((1, 0), (0, 0), (0, 0)), mode='constant',
                                 constant_values=args.cam_eval_thres)
            keys = np.pad(pack['label'][0] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(refined_cam, axis=0)
            cls_labels = keys[cls_labels]

            cls_labels[cls_labels != 1] = 0
            cls_labels[cls_labels == 1] = 255

            cls_labels = cv2.resize(cls_labels, (128, 128))

            imageio.imwrite(os.path.join(args.mask_dir, img_name + '.png'),
                            cls_labels.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    data = dataset.dataloader.SeafogClassificationDatasetMSF(args.img_list, data_root=args.data_root,
                                                             resize_long=(128, 128), scales=args.cam_scales)
    data = torchutils.split_dataset(data, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, data, args), join=True)
    print(']')

    torch.cuda.empty_cache()