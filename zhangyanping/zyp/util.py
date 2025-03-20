"""
Time:     2020/11/30 下午5:02
Author:   Ding Cheng(Deeachain)
File:     utils.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from utils.colorize_mask import cityscapes_colorize_mask, paris_colorize_mask, road_colorize_mask, \
    austin_colorize_mask, isprs_colorize_mask
import logging
from logging import handlers

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):

    if output_grey:
        if dataset == 'cityscapes':
            output[np.where(output==18)] = 33
            output[np.where(output==17)] = 32
            output[np.where(output==16)] = 31
            output[np.where(output==15)] = 28
            output[np.where(output==14)] = 27
            output[np.where(output==13)] = 26
            output[np.where(output==12)] = 25
            output[np.where(output==11)] = 24
            output[np.where(output==10)] = 23
            output[np.where(output==9)] = 22
            output[np.where(output==8)] = 21
            output[np.where(output==7)] = 20
            output[np.where(output==6)] = 19
            output[np.where(output==5)] = 17
            output[np.where(output==4)] = 13
            output[np.where(output==3)] = 12
            output[np.where(output==2)] = 11
            output[np.where(output==1)] = 8
            output[np.where(output==0)] = 7
        output_grey = Image.fromarray(output)
        output_grey.save(os.path.join(save_path, img_name + '.png'))

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'paris':
            output_color = paris_colorize_mask(output)
        elif dataset == 'road':
            output_color = road_colorize_mask(output)
        elif dataset == 'austin':
            output_color = austin_colorize_mask(output)
        elif dataset == 'postdam' or dataset == 'vaihingen':
            output_color = isprs_colorize_mask(output)
        output_color.save(os.path.join(save_path, img_name + '_color.png'))

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'paris':
            gt_color = paris_colorize_mask(gt)
        elif dataset == 'road':
            gt_color = road_colorize_mask(gt)
        elif dataset == 'austin':
            gt_color = austin_colorize_mask(gt)
        elif dataset == 'postdam' or dataset == 'vaihingen':
            gt_color = isprs_colorize_mask(gt)
        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters
#获取当前时间
def get_time():
    time_now = datetime.now()
    time_now_year = str(time_now.year)
    time_now_mouth = time_now.month
    if time_now_mouth < 10:
        time_now_mouth = '0' + str(time_now_mouth)
    else:
        time_now_mouth = str(time_now_mouth)
    time_now_day = time_now.day
    if time_now_day < 10:
        time_now_day = '0' + str(time_now_day)
    else:
        time_now_day = str(time_now_day)
    return time_now_year + "年" + time_now_mouth+ "月" + time_now_day + "日"

# 1d绝对sin_cos编码
def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb

#二维 相对
def create_2d_absolute_sin_cos_embedding(h, w, dim):
    # 奇数列和偶数列sin_cos，还有h和w方向，因此维度是4的倍数
    assert dim % 4 == 0, "wrong dimension"

    pos_emb = torch.zeros([h*w, dim])
    m1, m2 = torch.meshgrid(torch.arange(h), torch.arange(w))
    # [2, h, 2]
    coords = torch.stack([m1, m2], dim=0)
    # 高度方向的emb
    h_emb =create_1d_absolute_sin_cos_embedding(torch.flatten(coords[0]).numel(), dim // 2)
    # 宽度方向的emb
    w_emb =create_1d_absolute_sin_cos_embedding(torch.flatten(coords[1]).numel(), dim // 2)
    # 拼接起来
    pos_emb[:, :dim//2] = h_emb
    pos_emb[:, dim//2:] = w_emb
    return pos_emb

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)
