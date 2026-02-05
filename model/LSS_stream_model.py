import matplotlib.pyplot as plt
from cv2 import absdiff
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from data.tools import gen_dx_bx, QuickCumsum
import torch.nn.functional as F
from model.erfnet import Encoder, Decoder
from einops import rearrange
import math
import numpy as np
from tools import *
from math import ceil
import random
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode
from .memory_buff.memory_buffer import StreamTensorMemory
from .memory_buff.gru import ConvGRU
from torch.distributions.laplace import Laplace
from torch.distributions import MultivariateNormal
from torch.nn.modules.module import T
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        b, c, h, w = x2.shape
        if x1.shape[2] != h or x1.shape[3] != w:
            x1 = F.interpolate(x1, (h, w), mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear',
                               align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        x1 = self.up4(x1)
        x2 = self.up2(x2)
        h, w = x3.shape[-2], x3.shape[-1]
        x1 = F.interpolate(x1, (h, w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, (h, w), mode='bilinear', align_corners=True)
        x3 = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x3)


class BevEncode(nn.Module):
    def __init__(self, inC, outC, aux_out=0):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        self.out_1 = nn.Conv2d(128, outC, kernel_size=1, padding=0)


        self.out_2 = nn.Conv2d(128, aux_out, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.up1(x3, x1)
        x = self.up2(x)
        x_new = self.out_1(x)
        x_neg = self.out_2(x)

        # if self.aux:
        #     #x_pos = self.out_2(x)
        #     x_neg = self.out_3(x)
        #     return x_new,x_new,x_neg
        return x_new,x_new,x_neg


class CamEncode_2(nn.Module):
    def __init__(self, D, C, upsample, aus=False, backbone="efficientnet-b0"):
        super(CamEncode_2, self).__init__()
        self.D = D
        self.C = C
        self.upsample = upsample

        self.trunk = EfficientNet.from_pretrained(backbone)

        self.decoder_pv = Decoder(6, k=2)
        if backbone == "efficientnet-b0":
            self.up2 = Up2(320 + 112 + 40, 512)
        elif backbone == "efficientnet-b4":
            self.up2 = Up2(448 + 160 + 56, 512)
        self.aux = aus
        if self.aux:
            self.depthnet = nn.Conv2d(512, self.D + self.D + self.C, kernel_size=1, padding=0)
        else:
            self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)
        self.conc = nn.Conv2d(self.C, 128, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, img):
        x, ft = self.get_eff_depth(img)
        # Depth

        x_ = self.depthnet(x)
        if self.aux:
            dep_p = x_[:, :self.D]  # self.decoder_de(self.encoder_de(img))
            depth = self.get_depth_dist(x_[:, self.D:self.D + self.D])
            pv_fea = x_[:, self.D + self.D:(self.D + self.D + self.C)]
        else:
            dep_p = x_[:, :self.D]
            depth = self.get_depth_dist(x_[:, :self.D])
            pv_fea = x_[:, self.D:(self.D + self.C)]

        new_x = depth.unsqueeze(1) * pv_fea.unsqueeze(2)
        pv_out = self.decoder_pv(self.conc(pv_fea))

        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        return depth, new_x, x, pv_out, dep_p

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        # print(x.shape)

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(x.shape)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        x = self.up2(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'])

        return x, 0

    def forward(self, x, lidars):
        depth, new_x, x_mid, pv_out, dep_p = self.get_depth_feat(x)

        return depth, new_x, x_mid, pv_out, dep_p


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, teacher=False, stream_bev=False, aux_out=1, bsz=4):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        self.xbound = self.grid_conf['xbound']
        self.ybound = self.grid_conf['ybound']
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = dx.cuda()
        self.bx = bx.cuda()
        self.nx = nx.cuda()

        self.downsample = 32 // (int)(self.data_aug_conf['up_scale'])
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

        self.camencode = CamEncode_2(self.D, self.camC, self.data_aug_conf['up_scale'], aus=False, backbone=self.data_aug_conf['backbone'])

        self.bevencode = BevEncode(inC=self.camC, outC=outC, aux_out=aux_out)
        self.use_quickcumsum = True
        self.droup = nn.Dropout(0.5)
        self.bn = torch.nn.BatchNorm2d(64)
        self.stream_bev = stream_bev
        if self.stream_bev:
            self.roi_size = (self.xbound[1] - self.xbound[0], self.ybound[1] - self.ybound[0])
            self.stream_fusion_neck = ConvGRU(self.camC)
            self.stream_fusion_neck.init_weights()
            self.batch_size = bsz
            self.bev_memory = StreamTensorMemory(self.batch_size, )
            self.bev_memory_target = StreamTensorMemory(self.batch_size, )
            self.gt_memory = StreamTensorMemory(self.batch_size, )
            self.map_memory = StreamTensorMemory(self.batch_size)
            self.map_memory_target = StreamTensorMemory(self.batch_size)
            xmin, xmax = self.xbound[0], self.xbound[1]
            ymin, ymax = self.ybound[0], self.ybound[1]
            x = torch.linspace(xmin, xmax, self.nx[0])
            y = torch.linspace(ymax, ymin, self.nx[1])
            y, x = torch.meshgrid(y, x)
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            self.plane = torch.stack([x, y, z, ones], dim=-1).float()
            self.trans_loss = torch.nn.L1Loss()
        self.teach = teacher

        self.act = torch.nn.Softplus()
        self.relu = torch.nn.ReLU()#torch.sigmoid


    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x, lidars, mask=None):
        """Return B x N x D x H/downsample x W/downsample x C
        mask b*n c h w
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        depth, x, x_mid, pv_out, dep_p = self.camencode(x, lidars)

        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, imH // self.downsample, imW // self.downsample)

        return depth, x, x_mid, pv_out

    def voxel_pooling(self, geom_feats, x, is_av=False):
        # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample, C]
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        vi = geom_feats[0][0][0].detach().cpu().numpy()
        geom_feats = geom_feats.view(Nprime, 3)

        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)

        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, lidars, cross_bev=None):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        depth, x, x_mid, pv_out = self.get_cam_feats(x, lidars)

        # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample, C]
        x = self.voxel_pooling(geom, x)

        return depth, x, x_mid, pv_out


    def update_bev_feature(self, curr_bev_feats, img_metas, bev_memory_store, eval_temporal=False, ):

        bs = curr_bev_feats.size(0)
        fused_feats_list = []

        memory = bev_memory_store.get(img_metas)
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']
        trans_loss = torch.tensor([0.0]).cuda()
        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            # s=img_metas[i]['scene_name']
            # print('is or not first',is_first_frame,i,s)
            if eval_temporal:
                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                fused_feats_list.append(new_feat)
            else:
                if is_first_frame:
                    new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                    fused_feats_list.append(new_feat)
                else:
                    # print(666)
                    # else, warp buffered bev feature to current pose
                    translation_prev = pose_memory[i]['ego2global_translation']
                    rotation_prev = pose_memory[i]['ego2global_rotation']
                    translation_curr = img_metas[i]['ego2global_translation']
                    rotation_curr = img_metas[i]['ego2global_rotation']

                    de = translation_curr - translation_prev
                    patch_angle = quaternion_yaw(rotation_curr) / np.pi * 180
                    rot = patch_angle / 180 * np.pi
                    patch_angle_pre = quaternion_yaw(rotation_prev) / np.pi * 180
                    delta_translation_global = torch.tensor(
                        [de[0], de[1]]).unsqueeze(1)  # .permute(1, 0)#2 1
                    curr_transform_matrix = torch.tensor([[np.cos(-rot), -np.sin(-rot)],
                                                          [np.sin(-rot), np.cos(-rot)]])  # (2, 2) 世界坐标转换为当前帧自车坐标

                    delta_translation_curr = torch.matmul(
                        curr_transform_matrix.float(),
                        delta_translation_global.float()).squeeze()  # (bs, 2, 1) -> (bs, 2) = (bs, 2, 2）* (bs, 2, 1) (真实坐标)
                    delta_translation_curr = delta_translation_curr // 0.5  # (bs, 2) = (bs, 2) / (1, 2) (格子数)
                    delta_translation_curr = delta_translation_curr.round().tolist()

                    theta = patch_angle - patch_angle_pre
                    tran = [-delta_translation_curr[1], -delta_translation_curr[0]]  # nuscenes是x为h，y为w
                    warped_feat = affine(bev_memory[i],  # 必须是(c, h, w)形式的tensor
                                         angle=theta,
                                         # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                                         translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                                         scale=1,  # 浮点数，中心缩放尺度
                                         shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                                         interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                                         fill=[0.0])  # 先旋转再平移
                    trans_loss += self.trans_loss(warped_feat, curr_bev_feats[i])
                    new_feat = self.stream_fusion_neck(warped_feat, curr_bev_feats[i])
                    fused_feats_list.append(new_feat)

        fused_feats = torch.stack(fused_feats_list, dim=0)
        trans_loss = trans_loss / bs
        bev_memory_store.update(fused_feats, img_metas)

        return fused_feats, trans_loss

    def store_bev_feature(self, curr_bev_feats, img_metas, bev_memory_store,eval_temporal=False, ):

        bs = curr_bev_feats.size(0)
        prev_feats_list = []  ##

        memory = bev_memory_store.get(img_metas)
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']
        mask = torch.ones_like(curr_bev_feats.detach())
        mask_map=[]
        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if eval_temporal:
                prev_feats_list.append(curr_bev_feats[i].detach())
                mask_map.append(mask[i])
            else:
                if is_first_frame:
                    now_map = curr_bev_feats[i].detach()
                    c,h,w = now_map.shape
                    mask_new = torch.rand((c,h,w)).cuda()
                    mask_new = mask_new > 0.4
                    now_map = mask_new * now_map
                    prev_feats_list.append(now_map)
                    mask_map.append(torch.ones_like(now_map))
                else:
                    translation_prev = pose_memory[i]['ego2global_translation']
                    rotation_prev = pose_memory[i]['ego2global_rotation']
                    translation_curr = img_metas[i]['ego2global_translation']
                    rotation_curr = img_metas[i]['ego2global_rotation']

                    de = translation_curr - translation_prev
                    patch_angle = quaternion_yaw(rotation_curr) / np.pi * 180
                    rot = patch_angle / 180 * np.pi
                    patch_angle_pre = quaternion_yaw(rotation_prev) / np.pi * 180
                    delta_translation_global = torch.tensor(
                        [de[0], de[1]]).unsqueeze(1)  # .permute(1, 0)#2 1
                    curr_transform_matrix = torch.tensor([[np.cos(-rot), -np.sin(-rot)],
                                                          [np.sin(-rot), np.cos(-rot)]])  # (2, 2) 世界坐标转换为当前帧自车坐标

                    delta_translation_curr = torch.matmul(
                        curr_transform_matrix.float(),
                        delta_translation_global.float()).squeeze()  # (bs, 2, 1) -> (bs, 2) = (bs, 2, 2）* (bs, 2, 1) (真实坐标)
                    delta_translation_curr = delta_translation_curr // 0.5  # (bs, 2) = (bs, 2) / (1, 2) (格子数)
                    delta_translation_curr = delta_translation_curr.round().tolist()

                    theta = patch_angle - patch_angle_pre
                    tran = [-delta_translation_curr[1], -delta_translation_curr[0]]  # nuscenes是x为h，y为w
                    warped_feat = affine(bev_memory[i],  # 必须是(c, h, w)形式的tensor
                                         angle=theta,
                                         # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                                         translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                                         scale=1,  # 浮点数，中心缩放尺度
                                         shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                                         interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                                         fill=[0.0])  # 先旋转再平移
                    warped_mask = affine(mask[i],  # 必须是(c, h, w)形式的tensor
                                         angle=theta,
                                         # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                                         translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                                         scale=1,  # 浮点数，中心缩放尺度
                                         shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                                         interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                                         fill=[0.0])
                    prev_feats_list.append(warped_feat)
                    mask_map.append(warped_mask)

        prev_map = torch.stack(prev_feats_list, dim=0)
        bev_memory_store.update(curr_bev_feats, img_metas)
        mask_map = torch.stack(mask_map)
        return prev_map,mask_map

    def reset_memory(self):

        self.bev_memory.reset_single(0)
        # self.bev_memory_target.reset_single(0)

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, lidars, eval_temporal=False, meta=None, test=False, target=False,droup=False):

        depth, x_bev_init, x_mid, pv_out = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans, lidars)
        if self.stream_bev:
            if test:
                self.bev_memory.eval()
                self.bev_memory_target.eval()
            else:
                self.bev_memory.train()
                self.bev_memory_target.train()
            if target:
                bev_memory_store = self.bev_memory_target
            else:
                bev_memory_store = self.bev_memory
            x_bev_init, trans_loss = self.update_bev_feature(x_bev_init, meta, bev_memory_store=bev_memory_store, eval_temporal=eval_temporal)
        else:
            trans_loss=None
        if droup:
            x_bev_init = nn.Dropout2d(0.5)(x_bev_init)

        x, x_pos,x_neg = self.bevencode(x_bev_init)
        if test:
            return x
        if self.teach and not test:
            prev_map,mask_map = self.store_bev_feature(x.sigmoid(), meta, bev_memory_store=self.map_memory)
            return x, x_bev_init, prev_map,mask_map
        return x, depth, pv_out, x_bev_init, trans_loss

class LiftSplatShoot_edl(LiftSplatShoot):

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, lidars, eval_temporal=False, meta=None, test=False, target=False):
        depth, x_bev_init, x_mid, pv_out = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans, lidars)
        if self.stream_bev:
            if test:
                    self.bev_memory.eval()
                    self.bev_memory_target.eval()
            else:
                    self.bev_memory.train()
                    self.bev_memory_target.train()
            if target:
                    bev_memory_store = self.bev_memory_target
            else:
                    bev_memory_store = self.bev_memory
            x_bev_init, trans_loss = self.update_bev_feature(x_bev_init, meta, bev_memory_store=bev_memory_store, eval_temporal=eval_temporal)
        else:
            trans_loss =None
        x, _,x_neg = self.bevencode(x_bev_init)

        if test:
            return x,x_neg
        if self.teach and not test:
            if self.stream_bev:
                prev_map,mask_map = self.store_bev_feature(x.sigmoid(), meta, bev_memory_store=self.map_memory)
            else:
                prev_map, mask_map =None,None
            return x,x_bev_init, prev_map,x_neg,mask_map
        return x,  depth, pv_out, x_bev_init, trans_loss,x_neg

