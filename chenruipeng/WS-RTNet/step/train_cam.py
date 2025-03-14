import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import dataset.dataloader
from utils import pyutils, torchutils
import sssc.up as up
from sssc.SemanticSimilarity import SemanticSimilarity
from sssc.SpatialSimilarity import SpatialSimilarity
from sssc.up import FeatureAdjuster


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    print("cam_r model:",model)
    train_dataset = dataset.dataloader.SeafogClassificationDataset(args.img_list, data_root=args.data_root,
                                                                resize_long=None, hor_flip=True,
                                                                crop_size=256, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches


    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()
    # 初始化超像素相似性
    spatial_similarity = SpatialSimilarity()
    semantic_similarity = SemanticSimilarity().cuda()
    feature_adjuster = FeatureAdjuster(2048).cuda()
    # 均方误差损失
    # mse = nn.MSELoss()
    timer = pyutils.Timer()
     # 混合精度
    # scaler = torch.cuda.amp.GradScaler()


    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)
            # 20240526 add spatial-semantic simlilarity consistence
            # 20240524 add  high level features
            # 可行性待检验 高级特征y的shape待检验
            x,y = model(img)

            optimizer.zero_grad()
            # classfication loss
            loss = F.multilabel_soft_margin_loss(x, label)
            # sssc loss
            # 计算空间语义相似性一致损失
            # target_size = (256, 256)
            # # print(y.shape)
            # feature = feature_adjuster(y)
            # # print(feature.shape)
            # su_index, su_fea, su_sim = spatial_similarity.process(img)
            # # # print("121212121:",su_index.shape)
            # su_sim = su_sim.cuda()
            # sem = semantic_similarity(feature, su_index)
            # sem = sem[1]
            # loss_mse = torch.norm(sem-su_sim, p='fro')
            # +0.5*mse
            loss.backward()
            # print("mse loss:",loss_mse)
            # print("ce loss:",loss_ce)
            # print(f"step:{step};loss: {loss}")
            avg_meter.add({'loss': loss.item()})
            # avg_meter.add({'loss_ce': loss_ce.item()})
            # avg_meter.add({'loss_mse': loss_mse.item()})
            optimizer.step()
            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss')),
                       # 'loss_ce:%.4f' % (avg_meter.pop('loss_ce')),
                       # 'loss_mse:%.4f' % (avg_meter.pop('loss_mse')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        
        timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name)
    torch.cuda.empty_cache()