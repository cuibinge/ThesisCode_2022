4189921#!/usr/bin/python3
#coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np
import random
from PIL import Image
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset, DataLoader
from lib.data_prefetcher import DataPrefetcher

# def random_bright(img, p=0.5):
#     if torch.rand(1) < p:
#         shift_value = 10
#         shift = torch.rand(1) * 2 * shift_value - shift_value
#         img = img.float()
#         img += shift
#         img = torch.clamp(img, 0, 255)
#         #img = img.byte()
#     return img

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'my_data_large' in self.kwargs['datapath']:
            self.mean      = np.array([[[61.25,77.57,55.45]]])
            self.std       = np.array([[[4.74, 8.33,4.93]]])
        elif 'my_data_1000' in self.kwargs['datapath']:
            self.mean      = np.array([[[61.59,77.65,55.42]]])
            self.std       = np.array([[[4.57, 7.61,4.61]]])
        else:
            #raise ValueError
#             self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
#             self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])
            
#             self.mean      = np.array([[[61.76,78.18,55.95]]])
#             self.std       = np.array([[[4.83,8.43,4.98]]])
            
                      #1100  1000涂鸦100真值
            self.mean      = np.array([[[61.68,77.64,55.38]]])
            self.std       = np.array([[[4.55,7.45,4.55]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                imagepath = cfg.datapath + '/image/' + line.strip() + '.png'
                maskpath  = cfg.datapath + '/scribble/'  + line.strip() + '.png'
                self.samples.append([imagepath, maskpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.RandomHorizontalFlip(),
                                                    transform.RandomCrop(320, 320),
                                                    transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath = self.samples[idx]
        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        H, W, C             = mask.shape
        image, mask         = self.transform(image, mask)
#         image = random_bright(image, p=0.5)
        mask[mask == 0.] = 255.
        mask[mask == 2.] = 0.
        return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='./data')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)
    batch_idx = -1
    image, mask = prefetcher.next()
    image = image[0].permute(1,2,0).cpu().numpy()*cfg.std + cfg.mean
    mask  = mask[0].cpu().numpy()
    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.subplot(122)
    plt.imshow(mask)
    input()

