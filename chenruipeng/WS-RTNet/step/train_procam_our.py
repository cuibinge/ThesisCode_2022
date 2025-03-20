import torch
import  torch.nn as nn
import os.path as osp
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import cv2
import importlib

import dataset.dataloader
import net.resnet50_cam
from utils import pyutils, torchutils

def mx_norm(pred):
    n, c, _= pred.shape
    pred = torch.clamp(pred,0)
    pred = pred.reshape(n, c, -1)
    pred_max = torch.max(pred, dim=2, keepdim=True)[0]
    pred /= pred_max + 1e-5
    return pred

def alignment_loss(pred, sp, label, criterion):
    superpixel = sp.float()

    superpixel = F.interpolate(superpixel.unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1)
    _, h2, w2 = superpixel.shape

    n, c, _, _ = pred.shape

    sp_num = int(superpixel.max().numpy()) + 1
    superpixel = superpixel.to(torch.int64)
    sp_flat = F.one_hot(superpixel, num_classes=sp_num)
    sp_flat = sp_flat.permute(0, 3, 1, 2)
    pred = F.interpolate(pred, size=(h2, w2), mode='bilinear')

    sp_flat = sp_flat.reshape(n, sp_num, -1)
    score_flat = pred.reshape(n, c, -1) * label.reshape(n, c, 1)
    sp_flat_avg = sp_flat / np.maximum(sp_flat.sum(axis=2, keepdims=True), 1e-5)
    sp_flat_avg = sp_flat_avg.float()
    score_sum = torch.matmul(score_flat, torch.transpose(sp_flat, 1, 2).cuda().float())
    score_sum = score_sum.reshape(n, c, sp_num, 1)
    sp_flat_avg = sp_flat_avg.unsqueeze(1).repeat(1, c, 1, 1)
    final_averaged_value = torch.mul(sp_flat_avg.cuda(), score_sum)
    final_averaged_value = final_averaged_value.sum(2)
    final_averaged_value = final_averaged_value * label.reshape(n, c, 1)
    original_value = mx_norm(score_flat)
    final_averaged_value = mx_norm(final_averaged_value)

    original_value = original_value[label == 1]
    oriv = original_value.detach().cpu().numpy()
    oriv = oriv.reshape(-1, h2, w2)
    thr = 0.3
    for i in range(oriv.shape[0]):
        single_att = oriv[i]
        region = np.zeros_like(single_att)
        region[single_att > thr] = 1.0
        kernel = np.ones((8, 8), np.uint8)
        region = cv2.erode(region.astype('uint8'), kernel, iterations=1)
        oriv[i] = oriv[i] * region

    oriv = torch.from_numpy(oriv).reshape(-1, h2 * w2).cuda()

    final_averaged_value = final_averaged_value[label == 1].detach()
    final_averaged_value = torch.where(final_averaged_value > oriv, final_averaged_value, oriv)
    a_loss = criterion(original_value, final_averaged_value)

    return a_loss.float()

def run(args):

    # mse_loss = nn.MSELoss(size_average=True).cuda()

    torch.backends.cudnn.benchmark = True
    print('train_procam')
    model = getattr(importlib.import_module(args.cam_network), 'Net_CAM')()
    print("procam_628 model:",model)
    param_groups = model.trainable_parameters()
    #不加载cam权重
    # model.load_state_dict(torch.load(args.cam_weights_name), strict=True)
    model = torch.nn.DataParallel(model).cuda()

    refine_classifier = net.resnet50_cam.Refine_Classifier(2, args.feature_dim, args.momentum)
    refine_classifier = torch.nn.DataParallel(refine_classifier).cuda()
    refine_classifier.train()

    train_dataset = dataset.dataloader.SeafogClassificationPairDataset(args.img_list, data_root=args.data_root,
                                                                resize_long=None, hor_flip=True,
                                                                crop_size=256, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.procam_num_epoches

    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 0.1*args.procam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 0.1*args.procam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': refine_classifier.parameters(), 'lr': args.procam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.procam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
   
    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()
    global_step = 0

    # 初始化超像素相似性
    # spatial_similarity = SpatialSimilarity()
    # semantic_similarity = SemanticSimilarity().cuda()
    # feature_adjuster = FeatureAdjuster(2048, 3).cuda()
    # # 均方误差损失
    # mse = torch.nn.MSELoss()

    # 混合精度
    scaler = torch.cuda.amp.GradScaler()

    criterion = torch.nn.MSELoss()

    for ep in range(args.procam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.procam_num_epoches))
        model.train()

        for step, pack in enumerate(train_data_loader):
            # print(f'now is step {step}')
            pos_img = pack['pos_img'].cuda()
            pos_label = pack['pos_label'].cuda(non_blocking=True)
            pos_sp = pack['pos_spixel']

            with torch.cuda.amp.autocast():
                pos_x,pos_feat,pos_cam,pos_x_int = model(pos_img)

                neg_img = pack['neg_img'].cuda()
                neg_label = pack['neg_label'].cuda(non_blocking=True)
                neg_x,neg_feat,neg_cam,neg_x_int = model(neg_img)

                if ep >5:
                    loss_align = alignment_loss(pos_x_int, pos_sp, pos_label, criterion)
                    # print(loss_align)
                    # get cam_mask by threshold
                    b, _, h, w = pos_cam.shape
                    cam_mask = pos_cam.detach()[:,1,:,:].unsqueeze(1) # cam for foreground  bs*1*h*w
                    cam_mask[cam_mask>args.cam_mask_thres] = 1
                    cam_mask[cam_mask<=args.cam_mask_thres] = 0
                    cam_mask = cam_mask.float()
                    
                    # masked average pooling
                    masked_feat = cam_mask * pos_feat
                    b, c, h, w = masked_feat.shape
                    masked_feat = masked_feat.permute(0, 2, 3, 1).contiguous().view(-1, c)
                    local_proto = masked_feat.sum(dim=0, keepdims=True) / (cam_mask!=0).sum()
                    local_proto = F.normalize(local_proto, p=2, dim=1)
                    
                    # global prototype update
                    refine_classifier.module.update(local_proto.detach())
                    global_proto = refine_classifier.module.prototype
                    
                    # prototype re-activation
                    pos_feat_norm = F.normalize(pos_feat.permute(0, 2, 3, 1).contiguous().view(-1, c), p=2, dim=1)
                    cos_sim = pos_feat_norm @ local_proto.t()
                    cos_sim = cos_sim.view(b, 1, h, w)
                    cos_sim[cos_sim>args.activation_thres] = 1
                    cos_sim[cos_sim<=args.activation_thres] = 0
                    cos_sim = cos_sim.detach()
                    class_feat = pos_feat * cos_sim
                    
                    # pixel confidence map
                    pcm = class_feat.softmax(dim=1)
                    weighted_feat = pcm * class_feat
                    
                    feat_ = torch.cat([weighted_feat, neg_feat], dim=0)
                    x_ = torchutils.gap2d(feat_, keepdims=True)
                    
                    logits = refine_classifier(x_)
                    loss_pra = F.multilabel_soft_margin_loss(logits, label)
                    
                    # pixel-to-prototype contrast
                    mask = cam_mask.permute(0, 2, 3, 1).contiguous().view(-1, )
                    fg_embedding = pos_feat_norm[mask == 1]
                    bg_embedding = neg_feat.permute(0, 2, 3, 1).contiguous().view(-1, c)
                    bg_embedding = F.normalize(bg_embedding, p=2, dim=1)
                    neg_embedding = torch.cat([bg_embedding, global_proto], dim=0)
                    logit_neg = fg_embedding @ neg_embedding.t() / args.temperature
                    A1 = torch.exp(torch.sum(fg_embedding @ global_proto.t(), dim=-1) / args.temperature)
                    A2 = torch.sum(torch.exp(logit_neg), dim=-1)
                    loss_ppc = torch.mean(-1 * torch.log(A1 / A2))
                    #  # self-augmented regularization
#                     stride_cam_mask = F.interpolate(cam_mask, pos_img.shape[-2:], mode='bilinear')
#                     stride_cam_mask[stride_cam_mask>0] = 1
#                     stride_cam_mask[stride_cam_mask<0] = 0
#                     aug_img = pos_img * stride_cam_mask + neg_img * (1-stride_cam_mask)
#                     _,_,aug_cam,aug_x_int = model(aug_img)
#                     aug_cam_gt = torch.cat([(1-cam_mask),cam_mask], dim=1)
#                     loss_sar = F.mse_loss(aug_cam, aug_cam_gt.detach())
                else:
                    loss_pra = torch.zeros(1).cuda()
                    loss_ppc = torch.zeros(1).cuda()
                    loss_align = torch.zeros(1).cuda()
#                     loss_sar = torch.zeros(1).cuda()

                x = torch.cat([pos_x, neg_x], dim=0)
                label = torch.cat([pos_label, neg_label], dim=0)
                loss_cls = F.multilabel_soft_margin_loss(x, label)



               

                # structure-semantic similiarity consistence
                # target_size = (256, 256)
                # feature = feature_adjuster(pos_feat, target_size)
                # su_index, su_fea, su_sim = spatial_similarity.process(pos_img)
                # # # print("121212121:",su_index.shape)
                # su_sim = su_sim.cuda()
                # sem = semantic_similarity(feature, su_index)
                # sem = sem[1]
                # loss_mse = mse(sem, su_sim)
                # print("mse loss:",loss_mse)
                # print("ce loss:",loss_ce)
                # print(f"step:{step};loss: {loss}")


                loss = loss_cls + args.procam_loss_weight*loss_pra + args.contrastive_loss_weight*loss_ppc+ + args.sp_loss_weight * loss_align
                # + args.sp_loss_weight * loss_align
                 # + args.reg_loss_weight*loss_sar
#                 loss = loss_cls + args.contrastive_loss_weight*loss_ppc

                avg_meter.add({'loss_cls': loss_cls.item()})
                avg_meter.add({'loss_pra': loss_pra.item()})
                avg_meter.add({'loss_ppc': loss_ppc.item()})
#                 avg_meter.add({'loss_align': loss_align.item()})
                avg_meter.add({'loss_sar': loss_sar.item()})

                optimizer.zero_grad()

                # loss.backward()

                # 打印 profiler 结果
                # print(prof.key_averages().table(sort_by="cpu_time_total"))
                # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if (global_step-1)%100 == 0:
                timer.update_progress(global_step / max_step)

                print('step:%5d/%5d' % (global_step - 1, max_step),
                      'loss_cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'loss_pra:%.4f' % (avg_meter.pop('loss_pra')),
                      'loss_ppc:%.4f' % (avg_meter.pop('loss_ppc')),
#                       'loss_align:%.4f' % (avg_meter.pop('loss_align')),
                      'loss_sar:%.4f' % (avg_meter.pop('loss_sar')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[2]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        
        timer.reset_stage()
        # torch.save(model.module.state_dict(), osp.join(args.procam_weight_dir,'res50_procam_'+ str(ep+1) + '.pth'))    
        # torch.save(refine_classifier.module.state_dict(), osp.join(args.procam_weight_dir,'refine_classifier_'+ str(ep+1) + '.pth'))
    torch.save(model.module.state_dict(), osp.join(args.procam_weight_dir,'res50_procam_'+ str(args.procam_num_epoches) + '.pth'))    
    torch.save(refine_classifier.module.state_dict(), osp.join(args.procam_weight_dir,'refine_classifier_' + str(args.procam_num_epoches) + '.pth'))
    torch.cuda.empty_cache()