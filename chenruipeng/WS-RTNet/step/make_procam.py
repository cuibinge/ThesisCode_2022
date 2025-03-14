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
import net.resnet50_cam
import cv2
import imageio
cudnn.enabled = True

from .par import PAR

def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    
    refine_classifier = net.resnet50_cam.Refine_Classifier(2, args.feature_dim, args.momentum)
    refine_classifier.load_state_dict(torch.load(osp.join(args.procam_weight_dir, 'refine_classifier_' + str(args.procam_num_epoches) + '.pth')))
    
    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        refine_classifier.cuda()

        if args.par_refine:
            par = PAR(num_iter=20, dilations=[1, 4, 16, 32, 48, 64])
            par = par.cuda()

        all_features = []
        all_labels = []
        
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_up_size = imutils.get_strided_up_size(size, 16)
            outputs = [model.forward1(img[0].cuda(non_blocking=True), refine_classifier.classifier.weight) for img in pack['img']]  # b x 2 x w x h

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0]

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            highres_cam_v = highres_cam
            highres_cam = highres_cam.cpu().numpy()
            highres_cam = np.squeeze(highres_cam, axis=0)

            refined_cam = highres_cam_v
            if args.par_refine:
                img = pack['img'][0][0][0][None, ...].cuda()
                refined_cam = refined_cam[None, ...]
                refined_cam = par(img, refined_cam)
                refined_cam = refined_cam[0]

            refined_cam = refined_cam.cpu().numpy()
            refined_cam_v = np.squeeze(refined_cam, axis=0)

            # 获取原始图像并将其转为HWC格式
            original_image = pack['img'][0][0][0].cpu().numpy().transpose(1, 2, 0)
            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())  # 归一化

            # 叠加原始图像和CAM
            plt.imshow(original_image)  # 绘制原图
            plt.imshow(refined_cam_v, cmap='jet', alpha=0.5)  # 使用 jet colormap，alpha 为透明度
            plt.axis('off')  # 隐藏坐标轴
            plt.gca().set_position([0, 0, 1, 1])  # 去掉所有的边距

            # 保存叠加后的图像
            cam_image_path = f'/chenruipeng/weakly/WRTNet/WRTNet/data/workspace/procam/{img_name}_procam_overlay.png'
            plt.savefig(cam_image_path, bbox_inches='tight', pad_inches=0, transparent=True)  # 保存图片时去掉边框和留白
            plt.close()  # 关闭Matplotlib的图形窗口

            # 调整 refined_cam 的大小并生成掩码
            refined_cam = np.pad(refined_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            keys = np.pad(pack['label'][0] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(refined_cam, axis=0)
            cls_labels = keys[cls_labels]
            cls_labels[cls_labels != 1] = 0
            cls_labels[cls_labels == 1] = 255
            
            cls_labels = cv2.resize(cls_labels, (256, 256))

            # 保存类别标签图像
            imageio.imwrite(os.path.join(args.mask_dir, img_name + '.png'), cls_labels.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

            
#         print("merging")
#         # 将所有特征和标签合并
#         all_features = np.concatenate(all_features, axis=0)
#         all_labels = np.concatenate(all_labels, axis=0)
#         print(len(all_features))
#         # t-SNE降维
#         tsne = TSNE(n_components=2, random_state=42)
#         features_2d = tsne.fit_transform(all_features)

#         # 保存t-SNE结果
#         tsne_save_dir = '/root/proj/ProCAM/data/workspace/tsne_pro_visualizations'
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
    model.load_state_dict(torch.load(osp.join(args.procam_weight_dir,'res50_procam_'+str(args.procam_num_epoches) + '.pth')))
    model.eval()

    n_gpus = torch.cuda.device_count()

    data = dataset.dataloader.SeafogClassificationDatasetMSF(args.img_list, data_root=args.data_root, resize_long=(256,256), scales=args.cam_scales)
    data = torchutils.split_dataset(data, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, data, args), join=True)
    print(']')

    torch.cuda.empty_cache()