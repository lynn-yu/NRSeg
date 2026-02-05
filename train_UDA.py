import os

os.environ["CUDA_VISIBLE_DEVICES"] = '5,'
import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from data.Stream_nus_seg import compile_stream_data_domain
from model.LSS_stream_model import LiftSplatShoot_edl
from tools import *
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from model.confusion import BinaryConfusionMatrix, singleBinary
from data.sampler.group_sampler import InfiniteGroupEachSampleInBatchSampler
import torch.nn.functional as F


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def get_val_info(model, valloader, device, eval_temporal=False, use_tqdm=True):
    model.eval()
    confusion = BinaryConfusionMatrix(6)
    confusion_new = BinaryConfusionMatrix(6)
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    print(eval_temporal)
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, _, meta = batch
            # print(idx)
            preds,preds_edl = model(allimgs.to(device), rots.to(device), trans.to(device), intrins.to(device),
                                      post_rots.to(device), post_trans.to(device), lidars.to(device), meta=meta, test=True, eval_temporal=eval_temporal)

            inv_binimgs = get_inv_gt(binimgs[:, :6])
            scores = preds.sigmoid().cpu()
            confusion.update(scores > score_th, binimgs[:, :6] > 0)
            scores = get_prob_2(preds_edl)
            scores = scores.cpu()
            confusion_new.update(scores > score_th_prob, binimgs[:, :6] > 0)
    iou = confusion.iou
    miou = confusion.mean_iou
    iou_new = confusion_new.iou
    miou_new = confusion_new.mean_iou
    return {'iou': iou, 'miou': miou, 'iou_new': iou_new, 'miou_new': miou_new, }
    #return { 'iou': iou, 'miou': miou,  }


def get_val_info_mini(model, valloader, device, eval_temporal=False, use_tqdm=True):
    model.eval()
    confusion = BinaryConfusionMatrix(6)
    confusion_c = BinaryConfusionMatrix(1)
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs, \
                un_img_ori, un_post_rots_ori, un_post_trans_ori, seg_mask_un, seg_mask_o = batch

            preds, _, _, _, _ = model(allimgs.to(device), rots.to(device), trans.to(device), intrins.to(device),
                                      post_rots.to(device), post_trans.to(device), lidars.to(device), meta=meta, test=True, eval_temporal=eval_temporal)

            scores = preds.cpu().sigmoid()
            confusion.update(scores > score_th, binimgs[:, :6] > 0)

    iou = confusion.iou
    miou = confusion.mean_iou
    return {'iou': iou, 'miou': miou, }




def get_inv_gt(mask):
    mask_road = mask[:, 0] + mask[:, 2]
    mask_inv = mask_road.unsqueeze(1) - mask
    mask_inv[:, 0] = mask[:, 2]
    mask_inv[:, 2] = mask[:, 0]
    mask_inv = mask_inv>0
    return mask_inv*1.0


def train(logdir, grid_conf, data_aug_conf, version, dataroot, nsweeps, domain_gap, source, target, bsz, nworkers, lr, weight_decay, nepochs,
          max_grad_norm=5.0, gpuid=0, num_iters_to_seq=1,select_data=False,data_type=None):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logging.basicConfig(filename=os.path.join(logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))


    strainloader, ttrainloader, tvalloader = compile_stream_data_domain(version, dataroot, data_aug_conf, grid_conf, nsweeps,
                                                                        domain_gap, source, target, bsz, nworkers, num_iters_to_seq=num_iters_to_seq,
                                                                        flip=True,data_type=data_type,select_data=select_data )
    straingenerator = iter(strainloader)
    ttraingenerator = iter(ttrainloader)

    datalen = len(strainloader)

    device = torch.device(f'cuda:{gpuid}')

    student_model = LiftSplatShoot_edl(grid_conf, data_aug_conf, outC=6, stream_bev=True, bsz=bsz,aux_out=6)
    teacher_model = LiftSplatShoot_edl(grid_conf, data_aug_conf, outC=6, stream_bev=True, bsz=bsz,aux_out=6, teacher=True)
    student_model.to(device)
    teacher_model.to(device)
    student_model.train()
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.detach_()


    logger.info(f"source: {source}  ")
    logger.info(f"data from: {data_type}  ")
    logger.info(f"is select: {select_data}  ")

    epoch = int(nepochs)

    depth_weights = torch.Tensor([2.4, 1.2, 1.0, 1.1, 1.2, 1.4, 1.8, 2.3, 2.7, 3.5,
                                  3.6, 3.9, 4.8, 5.8, 5.4, 5.3, 5.0, 5.4, 5.3, 5.9,
                                  6.5, 7.0, 7.5, 7.5, 8.5, 10.3, 10.9, 9.8, 11.5, 13.1,
                                  15.1, 15.1, 16.3, 16.3, 17.8, 19.6, 21.8, 24.5, 24.5, 28.0,
                                  28.0]).to(device)
    loss_depth = DepthLoss(depth_weights).cuda(gpuid)
    loss_pv = Dice_Loss().cuda()
    loss_final = Dice_Loss().cuda()
    loss_unc = Beta_EDL_Loss().cuda()

    wp = 0.5
    loss_nan_counter = 0
    np.random.seed()
    uplr = int(datalen * 0.7)
    single_epoch_iter = datalen // bsz
    num_iters = epoch * single_epoch_iter
    logger.info(f"single_epoch_iter: {single_epoch_iter}  ")
    rand_iter = num_iters_to_seq * single_epoch_iter
    logger.info(f"rand it: {rand_iter}  ")

    opt = AdamW(student_model.parameters(), lr=lr)
    # sched = StepLR(opt, 10*datalen//bsz, 0.1)
    sched = PolyLR(opt, max_iters=num_iters, min_lr=0.0)

    w_edl = 1

    for iteration in range(num_iters):

        ###source student
        try:
            imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s, meta = next(straingenerator)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            straingenerator = iter(strainloader)
            imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s, meta = next(straingenerator)

        # vis_img(imgs[0])
        # vis_pv_mask(seg_mask_s[0])
        lidars_i = rearrange(lidars, 'b n c h w -> (b n) c h w')
        seg_mask = rearrange(seg_mask_s, 'b n c h w -> (b n) c h w')

        preds, depth, pv_out, x_bev, trans_loss,pred_edl = student_model(imgs.to(device), rots.to(device),
                                                                trans.to(device), intrins.to(device),
                                                                post_rots.to(device),
                                                                post_trans.to(device),
                                                                lidars_i.to(device), meta=meta, )

        binimgs = binimgs_s[:, :6].to(device)
        gt_weight = []
        for i in range(len(meta)):
            gt_weight.append(meta[i]['weight'])
        gt_weight = torch.tensor(gt_weight).cuda()
        loss_f= loss_final(preds, binimgs,is_weight=gt_weight)#
        loss_d, d_isnan = loss_depth(depth, lidars.to(device))
        loss_d = 0.1 * loss_d
        loss_p = wp * loss_pv(pv_out, seg_mask.cuda())

        annealing_coef = min(1, iteration / num_iters)
        loss_e = w_edl * loss_unc(pred_edl,  binimgs,annealing_coef=annealing_coef,is_weight=gt_weight) # , is_weight=gt_weight
        pred_edl = get_prob_2(pred_edl)
        w_consist = 0.01
        loss_consistent = w_consist*torch.square((preds.sigmoid() > score_th) * 1.0 - (pred_edl > score_th_prob) * 1.0).mean()

        ###target
        try:
            un_image, un_rots, un_trans, un_intrins, un_post_rots, un_post_trans, un_lidars, un_binimgs_t, \
                un_img_ori, un_post_rots_ori, un_post_trans_ori, seg_mask_t, seg_mask_o, un_meta = next(ttraingenerator)

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            ttraingenerator = iter(ttrainloader)
            un_image, un_rots, un_trans, un_intrins, un_post_rots, un_post_trans, un_lidars, un_binimgs_t, \
                un_img_ori, un_post_rots_ori, un_post_trans_ori, seg_mask_t, seg_mask_o, un_meta = next(ttraingenerator)
        un_binimgs = un_binimgs_t[:, :6].to(device)
        inv_un_binimgs = get_inv_gt(un_binimgs)
        seg_mask_un = rearrange(seg_mask_t, 'b n c h w -> (b n) c h w')
        with torch.no_grad():
            preds_un_ori, x_bev_un_ori, prev_map,preds_edl_un_ori,mask_map = teacher_model(un_img_ori.to(device), un_rots.to(device),
                                                                               un_trans.to(device), un_intrins.to(device), un_post_rots_ori.to(device),
                                                                               un_post_trans_ori.to(device), None, meta=un_meta, target=True)

        preds_un,depth_un, pv_out_un, x_bev_un, trans_loss_un,preds_edl_un = student_model(un_image.to(device), un_rots.to(device),
                                                                        un_trans.to(device), un_intrins.to(device),
                                                                        un_post_rots.to(device), un_post_trans.to(device), None,
                                                                        meta=un_meta, target=True)  #
        it = iteration
        weightofcon = sigmoid_rampup(it, uplr)
        w1 = min(0.002 + 0.1 * weightofcon, 0.1)
        loss_p_un = wp * loss_pv(pv_out_un, seg_mask_un.cuda())

        preds_edl_un_1 = get_prob_2(preds_edl_un)  # get_prob(preds_edl_un,get_inv_gt_2(preds_edl_un))
        preds_edl_un_ori_1 = get_prob_2(preds_edl_un_ori)  # get_prob(preds_edl_un_ori, get_inv_gt_2(preds_edl_un_ori))

        a = torch.square(preds_un.sigmoid() - preds_un_ori.sigmoid()) + w_edl * torch.square(preds_edl_un_1 - preds_edl_un_ori_1)

        a = a.mean() + w_consist * b.mean()

        loss_bev = 0.1 * w1 * torch.square((x_bev_un - x_bev_un_ori)).mean() + w1 * a


        # visual(preds_un_ori[0].sigmoid().cpu() > 0.5)
        # visual(preds_edl_un_ori_1[0].cpu() > score_th_prob)
        opt.zero_grad()
        if iteration == rand_iter or iteration < rand_iter:
            loss = loss_p + loss_p_un + loss_bev + loss_f + loss_e * 0.5
        else:
            loss = loss_p + loss_p_un + loss_bev + 0.1 * trans_loss + 0.1 * trans_loss_un + loss_f + loss_e*0.5
        if d_isnan:
            loss_nan_counter += 1
        else:
            loss = loss + loss_d
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
        opt.step()
        sched.step()

        alpha = min(1 - 1 / (iteration + 1), 0.999)  # when iteration>=999, alpha==0.999
        with torch.no_grad():
            model_state_dict = student_model.state_dict()
            ema_model_state_dict = teacher_model.state_dict()
            for entry in ema_model_state_dict.keys():
                ema_param = ema_model_state_dict[entry].clone().detach()
                param = model_state_dict[entry].clone().detach()
                new_param = (ema_param * alpha) + (param * (1. - alpha))
                ema_model_state_dict[entry] = new_param
            teacher_model.load_state_dict(ema_model_state_dict)

        if iteration > 0 and iteration % 10 == 0:
            iou, miou = singleBinary(preds.sigmoid() > score_th, binimgs > 0, )
            _, miou2 = singleBinary(pred_edl > score_th_prob, binimgs > 0, )
            _, miou_un = singleBinary(preds_un_ori.sigmoid() > score_th, un_binimgs > 0, )
            _, miou_un2 = singleBinary(preds_edl_un_ori_1 > score_th_prob, un_binimgs > 0, )

            logger.info(f"Train: [{iteration:>4d}/{num_iters}]:    "
                        f"lr {opt.param_groups[0]['lr']:>.2e}   "
                        #f"w1  {w1:>7.3f}   "
                        f"loss: {loss.item():>7.4f}   "
                        #f"loss_tmp: {a.item():>7.4f}   "
                       f"loss_f: {loss_f.item():>7.4f}   "
                             f"loss_e: {loss_e:>7.4f}   "
                        f"loss_c: {loss_consistent.item():>7.4f}   "
                            #f"loss_f_ori: {loss_f_ori.item():>7.4f}   "
                            f"mIoU: {miou:>7.4f}  "
                            f"mIoU2: {miou2:>7.4f}  "
                            f"mIoU_un: {miou_un:>7.4f}  "
                            f"mIoU_un2: {miou_un2:>7.4f}  "
                        )

        if (iteration % single_epoch_iter == 0 and iteration > 0) or iteration == (num_iters - 1):
            if rand_iter<1:
                eval_temporal = False
            else:
                eval_temporal = True if iteration == single_epoch_iter else False
            if source != 'boston' and version == 'v1.0-mini':
                val_info = get_val_info_mini(teacher_model, ttrainloader, device, eval_temporal=eval_temporal, )
            else:
                val_info = get_val_info(teacher_model, tvalloader, device, eval_temporal=eval_temporal, )
            logger.info(f"TargetVAL[{iteration:>2d}]:    "
                        f"TargetIOU: {np.array2string(val_info['iou'].cpu().numpy(), precision=3, floatmode='fixed')}  "
                        f"mIOU: {val_info['miou']:>7.4f}  "
                        f"TargetIOU: {np.array2string(val_info['iou_new'].cpu().numpy(), precision=3, floatmode='fixed')}  "
                        f"mIOU_neg: {val_info['miou_new']:>7.4f}  "
                        )

            mname = os.path.join(logdir, "model{}.pt".format(iteration))
            print('saving', mname)
            torch.save(teacher_model.state_dict(), mname)
            teacher_model.reset_memory()


if __name__ == '__main__':

    version = 'v1.0-trainval'  #'v1.0-mini'# 'v1.0-mini'#
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],  # [-30.0, 00.0, 0.15],#
        'ybound': [-50.0, 50.0, 0.5],  #
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                     'Ncams': 6, 'up_scale': 4,
                     'rand_resize': True, 'resize_lim': (0.20, 0.235),
                     'H': 900, 'W': 1600, 'new_H': 256, 'new_W': 384,
                     'rand_flip': True, 'rot_lim': (-5.4, 5.4), 'bot_pct_lim': (0.0, 0.22),
                     'color_jitter': True, 'color_jitter_conf': [0.2, 0.2, 0.2, 0.1],
                     'GaussianBlur': False, 'gaussion_c': (0, 2),
                     'final_dim': (128, 352),  # (224,480),#
                     'Aug_mode': 'hard',  # 'simple',#
                     'backbone': "efficientnet-b0",
                      'fre_change':False,
                     'pv_mode': 'mask2former',#'san',#'sanprob',#
                     'pv_mask_path':'san',#'mask2former',#
                     }
    b = 12
    if version == 'v1.0-mini':
        b = 4
    if data_aug_conf['backbone'] == "efficientnet-b4":
        lr = 4e-3 * b / 32
    else:
        lr = 1e-3 * b / 4
    source_name_list = ['boston', 'singapore', 'day', 'dry']
    target_name_list = ['singapore', 'boston', 'night', 'rain']
    n = 0

    source = source_name_list[n]
    target = target_name_list[n]
    if n==3:
        num_iters_to_seq = 2
    else:
        num_iters_to_seq = 1

    select_data = True
    data_type = ['Perl','Magic','PerMag']  # 'Perl','Ori'
    train(logdir='./ours' + source + '_' + target, version=version, dataroot='/data0/lsy/dataset/nuScenes/v1.0',
          grid_conf=grid_conf, data_aug_conf=data_aug_conf,
          select_data=select_data,
          domain_gap=True, source=source, target=target, nsweeps=3,
          bsz=b, nworkers=5, lr=lr, weight_decay=1e-2, nepochs=30,num_iters_to_seq=num_iters_to_seq,data_type=data_type)