import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode
from collections import OrderedDict
import torch
from shapely.strtree import STRtree
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.utils.data_classes import LidarPointCloud
from data.splits import create_splits_scenes
from data.tools import *
from nuscenes.utils.data_classes import Box
from data.sampler.group_sampler import InfiniteGroupEachSampleInBatchSampler,GroupSampler
from data.sampler.distributed_sampler import DistributedSampler
import cv2
from torch import distributed as dist

IMG_ORIGIN_H = 900
IMG_ORIGIN_W = 1600
MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}
def get_single_mask(vertices):
    tri_mask1 = np.zeros((200, 200))
    pts = vertices.reshape((-1, 1, 2))
    cv2.fillPoly(tri_mask1, [pts], color=1)
    return tri_mask1
def get_bool(s):
    # 去掉方括号
    s = s.strip('[]')
    # 按空格分割字符串
    parts = s.split()
    # 将每个部分转换为布尔值
    result = [part == 'True' for part in parts]

    return result

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
def sample_dataloader(dataset,bsz,nworkers,num_iters_to_seq,seed=None,is_val=False,collect=None):
    rank, world_size = get_dist_info()
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    if is_val:
        sampler = DistributedSampler(dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=False,
                                         seed=seed)
        batch_sampler = None
        loder = torch.utils.data.DataLoader(dataset,
                                            batch_size=bsz,
                                            sampler=sampler,
                                            batch_sampler=batch_sampler,
                                            num_workers=nworkers,
                                            collate_fn=collect
                                            )
    else:
        sampler = None
        batch_sampler =  InfiniteGroupEachSampleInBatchSampler(
            seq_split_num=2,
            num_iters_to_seq=num_iters_to_seq,
            random_drop=0.0,
            dataset=dataset,
            samples_per_gpu=bsz,
            num_replicas=world_size,
            rank=rank,
            seed=seed)
        loder = torch.utils.data.DataLoader(dataset,
                                sampler=sampler,
                                batch_sampler=batch_sampler,
                                num_workers=nworkers,
                                worker_init_fn=init_fn,
                                        pin_memory=False,
                                            collate_fn=collect
                                        )
    return loder

def compile_stream_data_domain(version, dataroot, data_aug_conf, grid_conf,nsweeps,  domain_gap,source,target, bsz,nworkers,num_iters_to_seq=1,
                               flip=False,select_data=False,data_type = ['Perl'], ):

    if source=='day':
        straindata = SemanticNuscData_Day(version, dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps, domain_gap=domain_gap, is_train=True,
                                      domain=source, domain_type='strain', is_source=True, select_data=True, data_type=data_type)  # 'Magic'

    elif source=='dry':
        straindata = SemanticNuscData_Dry(version, dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps, domain_gap=domain_gap, is_train=True,
                                          domain=source, domain_type='strain', is_source=True, select_data=True, data_type=data_type)  # 'Magic'
    else:
        straindata = SemanticNuscData(version,dataroot,  data_aug_conf, grid_conf, nsweeps=nsweeps,  domain_gap=domain_gap,is_train=True,
                                  domain=source, domain_type='strain',is_source=True,select_data=True,data_type=data_type)#'Magic'
    if version=='v1.0-mini' and source!='boston':
        ttraindata = FlipSemanticNuscData(version, dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps,
                                          domain_gap=domain_gap, is_train=True, domain=source, domain_type='strain')
    else:
        ttraindata = FlipSemanticNuscData(version, dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps,
                                      domain_gap=domain_gap, is_train=True, domain=target, domain_type='ttrain')


    valdata = SemanticNuscData(version,dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps,  domain_gap=domain_gap,is_train=False,domain=target, domain_type='tval')
    num_iters_to_seq = straindata.__len__()*num_iters_to_seq//bsz
    print('num_iters_to_seq=', num_iters_to_seq)
    strainloader = sample_dataloader(straindata, bsz, nworkers, num_iters_to_seq, collect=collate_wrapper)
    ttrainloade = sample_dataloader(ttraindata, bsz, nworkers, num_iters_to_seq, collect=collate_wrapper_flip)
    valloader = sample_dataloader(valdata,1,nworkers,num_iters_to_seq,is_val=True,collect=collate_wrapper)

    return strainloader, ttrainloade,valloader

def worker_rnd_init(x):
    np.random.seed(13 + x)

def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage",
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

class SemanticNuscData(torch.utils.data.Dataset):
    def __init__(self, version,dataroot, data_aug_conf, grid_conf, nsweeps, is_train, domain_gap,domain, domain_type,
                 is_source=False,select_data=False,data_type=None,is_osm=False,):
        self.nusc = NuScenes(version=version,
                    dataroot=dataroot,
                    verbose=False)
        self.filter_overlap = True
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.pv_mode = data_aug_conf['pv_mode'] if 'pv_mode' in data_aug_conf  else 'mask2former'
        self.pv_mask_path = data_aug_conf['pv_mask_path'] if 'pv_mask_path' in data_aug_conf else 'san'
        self.is_lidar = data_aug_conf['is_lidar'] if 'is_lidar' in data_aug_conf  else True
        self.rot = data_aug_conf['rot'] if 'rot' in data_aug_conf else True
        self.new_split = data_aug_conf['new_split'] if 'new_split' in data_aug_conf else False
        self.is_source = is_source
        print('new_split:',self.new_split)
        ###domain
        self.domain_gap = domain_gap
        self.domain = domain
        self.domain_type = domain_type
        ### lidar
        self.downsample = 32 // (int)(self.data_aug_conf['up_scale'])
        self.nsweeps = nsweeps
        self.select_data = select_data
        self.new_data_iou = dict()
        if domain=='day':
            self.perl_file_name = '/data2/lsy/nus_gen_night_data'
            self.magic_file_name = '/data2/lsy/magic_gen_night_data'
            self.SD_file_name = '/data2/lsy/SD_gen_data'
            middle_perl = 0
            middle_magic = 0
        elif domain=='dry':
            # self.perl_file_name = '/data2/lsy/nus_gen_data'
            # self.magic_file_name = '/data2/lsy/magic_gen_data'
            self.perl_file_name = '/data2/lsy/nus_gen_rain_data'
            self.magic_file_name = '/data2/lsy/magic_gen_rain_data'
            self.SD_file_name = '/data2/lsy/SD_gen_data'
            middle_perl = 0#.74
            middle_magic = 0#.75
        else:
            self.perl_file_name = '/data2/lsy/nus_gen_data'
            self.magic_file_name = '/data2/lsy/magic_gen_data'
            self.SD_file_name = '/data2/lsy/SD_gen_data'
            self.BEVCon_file_name='/data2/lsy/BEVCon_gen_data'
            middle_perl = 0
            middle_magic = 0
        if self.is_source:
            print(self.perl_file_name,self.magic_file_name)
        if self.select_data:
            for i in range(len(data_type)):
                name_i = data_type[i]
                self.new_data_iou[name_i]=dict()
                if name_i=='Perl':
                    if self.pv_mode=='san':
                        log_file = os.path.join(self.perl_file_name,'trainval_san_analyze.log')#'/data2/lsy/nus_gen_data/trainval_san_analyze.log'
                    elif self.pv_mode=='mask2former':
                        log_file = os.path.join(self.perl_file_name,'new_trainval_mask2former_analyze.log')#'/data2/lsy/nus_gen_data/trainval_mask2former_analyze.log'
                    elif self.pv_mode=='prob':
                        log_file = os.path.join(self.perl_file_name,'trainval_prob_analyze.log')#'/data2/lsy/nus_gen_data/trainval_proj_analyze.log'
                    elif self.pv_mode=='sanprob':
                        log_file = os.path.join(self.perl_file_name,'san_trainval_prob_analyze.log')
                    else:
                        print('error')
                    with open(log_file, 'r') as file:
                        for line in file:
                            parts = line.split(';')
                            token = parts[0].split(':')[-1]
                            cam_flag = parts[-2].split(':')[-1]
                            cam_flag = float(cam_flag)#get_bool(cam_flag)
                            if middle_perl!=0:
                                cam_flag = (cam_flag-middle_perl)*0.4+0.9
                            self.new_data_iou[name_i][token] = cam_flag
                elif name_i=='Magic':
                    if self.pv_mode == 'san':
                        log_file = os.path.join(self.magic_file_name,'trainval_san_analyze.log')#'/data2/lsy/magic_gen_data/trainval_san_analyze.log'
                    elif self.pv_mode == 'mask2former':
                        log_file = os.path.join(self.magic_file_name,'new_trainval_mask2former_analyze.log')#'/data2/lsy/magic_gen_data/trainval_mask2former_analyze.log'
                    elif self.pv_mode=='prob':
                        log_file = os.path.join(self.magic_file_name,'trainval_prob_analyze.log')#'/data2/lsy/magic_gen_data/trainval_prob_analyze.log'
                    elif self.pv_mode == 'sanprob':
                        log_file = os.path.join(self.magic_file_name, 'san_trainval_prob_analyze.log')
                    else:
                        print('error')
                    with open(log_file, 'r') as file:
                        for line in file:
                            parts = line.split(';')
                            token = parts[0].split(':')[-1]
                            cam_flag = parts[-2].split(':')[-1]
                            cam_flag = float(cam_flag)#get_bool(cam_flag)
                            if middle_magic != 0:
                                cam_flag = (cam_flag - middle_magic) * 0.4 + 0.9
                            self.new_data_iou[name_i][token] = cam_flag
                elif name_i == 'StableDiff':
                    log_file = os.path.join(self.SD_file_name, 'trainval_prob_analyze.log')  # '/data2/lsy/magic_gen_data/trainval_san_analyze.log'
                    with open(log_file, 'r') as file:
                        for line in file:
                            parts = line.split(';')
                            token = parts[0].split(':')[-1]
                            cam_flag = parts[-2].split(':')[-1]
                            cam_flag = float(cam_flag)  # get_bool(cam_flag)
                            self.new_data_iou[name_i][token] = cam_flag
                elif name_i == 'BEVCon':
                    log_file = os.path.join(self.BEVCon_file_name, 'new_trainval_mask2former_analyze.log')  # '/data2/lsy/magic_gen_data/trainval_san_analyze.log'
                    with open(log_file, 'r') as file:
                        for line in file:
                            parts = line.split(';')
                            token = parts[0].split(':')[-1]
                            cam_flag = parts[-2].split(':')[-1]
                            cam_flag = float(cam_flag)  # get_bool(cam_flag)
                            self.new_data_iou[name_i][token] = cam_flag
        self.nusc_maps = get_nusc_maps(self.nusc.dataroot)
        self.scene2map = {}
        for rec in self.nusc.scene:
            log = self.nusc.get('log', rec['log_token'])
            self.scene2map[rec['name']] = log['location']
        self.scenes = self.get_scenes()

        self.new_scenes = self.get_scenes_new()

        self.data_type = data_type
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        h = grid_conf['xbound'][1] - grid_conf['xbound'][1]
        w = grid_conf['ybound'][1] - grid_conf['ybound'][1]
        self.patch = (w,h)
        self.aug_mode = data_aug_conf['Aug_mode']

        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.seq_split_num = 1
        self._set_sequence_group_flag()
        self.is_osm=is_osm


        print(self)
        print(len(self.scenes),len(self.new_scenes),self.__len__(),data_aug_conf['Aug_mode'],)
        print('pv_weight:',self.pv_mode,'pv_path:',self.pv_mask_path,'is_lidar',self.is_lidar)


    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.seq_split_num == -1:
            self.flag = np.arange(len(self.samples))
            return

        res = []
        single_num = []
        curr_sequence = -1
        lrg = len(self.ixes)
        for idx in range(len(self.ixes)):
            if self.is_source:
                if self.ixes[idx]['new_data']:
                    if idx < lrg - 1 and self.ixes[idx]['next'] != self.ixes[idx + 1]['token'] and self.ixes[idx]['prev'] == '':
                        single_num.append(idx)
                    else:
                        if self.ixes[idx]['prev'] == '':
                            # new data sequence
                            curr_sequence += 1
                        res.append(curr_sequence)

                else:
                    if self.ixes[idx]['prev'] == '':
                        # new sequence
                        curr_sequence += 1
                    res.append(curr_sequence)
            else:
                if self.ixes[idx]['prev'] == '':
                    # new sequence
                    curr_sequence += 1
                res.append(curr_sequence)

        if len(single_num) != 0:
            print('dele single num ', len(single_num), single_num)
            self.ixes = [x for i, x in enumerate(self.ixes) if i not in single_num]

        self.flag = np.array(res, dtype=np.int64)
        # print(len(self.ixes),len(self.flag))
        assert len(self.ixes) == len(self.flag)

    def load_map_data(self,dataroot, location):

        # Load the NuScenes map object
        nusc_map = NuScenesMap(dataroot, location)

        map_data = OrderedDict()
        for layer in STATIC_CLASSES:

            # Retrieve all data associated with the current layer
            records = getattr(nusc_map, layer)
            polygons = list()

            # Drivable area records can contain multiple polygons
            if layer == 'drivable_area':
                for record in records:

                    # Convert each entry in the record into a shapely object
                    for token in record['polygon_tokens']:
                        poly = nusc_map.extract_polygon(token)
                        if poly.is_valid:
                            polygons.append(poly)
            else:
                for record in records:

                    # Convert each entry in the record into a shapely object
                    poly = nusc_map.extract_polygon(record['polygon_token'])
                    if poly.is_valid:
                        polygons.append(poly)

            # Store as an R-Tree for fast intersection queries
            map_data[layer] = STRtree(polygons)

        return map_data

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        if self.domain_gap:
            split = {
                'boston': {'strain': 'boston', 'ttrain': 'boston_train', 'tval': 'boston_val'},
                'singapore': {'strain': 'singapore', 'ttrain': 'singapore_train', 'tval': 'singapore_val'},
                'singapore_day': {'strain': 'singapore_day', 'ttrain': 'singapore_day_train',
                                  'tval': 'singapore_day_val'},
                'singapore_day_train': {'strain': 'singapore_day_train', 'ttrain': None, 'tval': None},
                'day': {'strain': 'day', 'ttrain': None, 'tval': None},
                'night': {'strain': None, 'ttrain': 'night_train', 'tval': 'night_val'},
                'dry': {'strain': 'dry', 'ttrain': None, 'tval': None},
                'rain': {'strain': None, 'ttrain': 'rain_train', 'tval': 'rain_val'},
                'nuscenes': {'strain': 'train', 'ttrain': 'train', 'tval': 'val'},
                'lyft': {'strain': 'lyft_train', 'ttrain': 'lyft_train', 'tval': 'lyft_val'},
            }[self.domain][self.domain_type]
        else:
            if self.new_split:
                split = {
                    'v1.0-trainval': {True: 'new_train', False: 'new_val'},
                    'v1.0-mini': {True: 'new_mini_train', False: 'new_mini_val'},
                }[self.nusc.version][self.is_train]
            else:
                split = {
                    'v1.0-trainval': {True: 'train', False: 'val'},
                    'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
                }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]
        if self.filter_overlap:
            sys_scenes = split = {
                    'v1.0-trainval': {True: 'new_train', False: 'new_val'},
                    'v1.0-mini': {True: 'new_mini_train', False: 'new_mini_val'},
                }[self.nusc.version][False]

        return scenes

    def get_scenes_new(self):
        # filter by scene split
        if self.domain_gap:
            if self.is_source and self.is_train:
                split = {
                        'v1.0-trainval': {True: 'train', False: 'val'},
                        'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
                }[self.nusc.version][False]
            else:
                return []
        else:
            if self.is_source and self.is_train:
                split = {
                    'v1.0-trainval': {True: 'train', False: 'val'},
                    'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
                }[self.nusc.version][False]
            else:
                return []
        scenes = create_splits_scenes()[split]

        return scenes
    def prepro(self):
        samples_o = [samp for samp in self.nusc.sample]

        # # remove samples that aren't in this split
        # samples = [samp for samp in samples if
        #            self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        samples = []
        new_samples = []
        new_samples_2 = []
        new_samples_3 = []
        new_samples_4 = []
        for samp in samples_o:
            if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes:
                samp['new_data'] = False
                samp['ori'] = None
                samples.append(samp)
            if self.is_source and self.is_train:
                if self.nusc.get('scene', samp['scene_token'])['name'] in self.new_scenes:
                    # if self.select_data:
                    #     token = samp['token']
                    #     if token in self.new_data_iou[self.data_type[0]]:
                    #         cam_flag = self.new_data_iou[self.data_type[0]][token]
                    #         cam_flag=1*np.array(cam_flag)
                    #         if cam_flag.mean()<1:
                    #             continue
                    #         else:
                    #             new_samp = samp.copy()
                    #             new_samp['new_data'] = True
                    #             new_samp['ori'] = self.data_type[0]
                    #             new_samples.append(new_samp)
                    #         if len(self.data_type) == 2:
                    #             cam_flag = self.new_data_iou[self.data_type[1]][token]
                    #             cam_flag = 1 * np.array(cam_flag)
                    #             if cam_flag.mean() < 1:
                    #                 continue
                    #             else:
                    #                 new_samp = samp.copy()
                    #                 new_samp['new_data'] = True
                    #                 new_samp['ori'] = self.data_type[1]
                    #                 new_samples_2.append(new_samp)
                    if True:
                        new_samp = samp.copy()
                        new_samp['new_data'] = True
                        new_samp['ori'] = self.data_type[0]
                        new_samples.append(new_samp)
                        if len(self.data_type) > 1:
                            new_samp = samp.copy()
                            new_samp['new_data'] = True
                            new_samp['ori'] = self.data_type[1]
                            new_samples_2.append(new_samp)
                            if len(self.data_type) > 2:
                                new_samp = samp.copy()
                                new_samp['new_data'] = True
                                new_samp['ori'] = self.data_type[2]
                                new_samples_3.append(new_samp)
                                if len(self.data_type) > 3:
                                    new_samp = samp.copy()
                                    new_samp['new_data'] = True
                                    new_samp['ori'] = self.data_type[3]
                                    new_samples_4.append(new_samp)


        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        if self.is_source and self.is_train:
            print('ori_sampel:', len(samples))
            print('new_sampel:', self.data_type[0], len(new_samples))
            new_samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
            samples = samples + new_samples
            if len(self.data_type) > 1:
                print('new_sampel:', self.data_type[1], len(new_samples_2))
                samples = samples + new_samples_2
                if len(self.data_type) > 2:
                    print('new_sampel:', self.data_type[2], len(new_samples_3))
                    samples = samples + new_samples_3
                    if len(self.data_type) > 3:
                        print('new_sampel:', self.data_type[3], len(new_samples_4))
                        samples = samples + new_samples_4
        return samples

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def sample_augmentation_simple_strong(self):
        fH, fW =  self.data_aug_conf['final_dim']#128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)#（1，1）变换尺度
        resize_dims = (fW, fH)#（128,352）目标数据尺寸
        if self.is_train:
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]) :
                colorjt = True
            else:
                colorjt = False
        else:
            colorjt = False
        return resize, resize_dims,colorjt
    def sample_augmentation_simple(self):
        fH, fW =  self.data_aug_conf['final_dim']#128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)#（1，1）变换尺度
        resize_dims = (fW, fH)#（128,352）目标数据尺寸
        if self.is_train:
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]) :
                colorjt = True
            else:
                colorjt = False
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            colorjt = False
            rotate= 0

        return resize, resize_dims,colorjt,rotate

    def sample_augmentation_weak(self):

        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if self.data_aug_conf['rand_resize']:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                resize = max(fH/H, fW/W)
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)  # （1，1）变换尺度
            # resize_dims = (fW, fH)  # （128,352）目标数据尺寸
            # newW, newH = resize_dims
            # crop_h = int((1 - 0)*newH) - fH
            # crop_w = int(max(0, newW - fW) / 2)
            # crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate =  np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop,flip, rotate

    def sample_augmentation(self,is_new=False):

        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if self.data_aug_conf['rand_resize']:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                resize = max(fH/H, fW/W)
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            fre_change = False
            if self.data_aug_conf['fre_change'] and np.random.choice([0, 1]):
                fre_change = True
            if self.rot:
                rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
            else:
                rotate = 0
            if self.data_aug_conf['color_jitter']:
                colorjt = True
            else:
                colorjt = False
            if self.data_aug_conf['GaussianBlur'] and np.random.choice([0, 1]):
                gauss = np.random.uniform(*self.data_aug_conf['gaussion_c'])
            else:
                gauss = 0
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - 0) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            colorjt = False
            gauss = 0
            fre_change=False
        return resize, resize_dims, crop, flip, rotate, colorjt,gauss,fre_change

    def get_label(self,mask):
        mask = torch.tensor(np.array(mask))
        vi = np.array(mask)
        mask_per = mask == 1
        mask_car = torch.logical_and(mask < 10, mask > 1)
        mask_tra = mask == 10
        mask_road = torch.logical_or(mask == 138, mask == 136)
        # mask_road = torch.logical_or(mask_road,mask==133)
        # mask_road = torch.logical_or(mask_road, mask == 134)
        # mask_road = torch.logical_or(mask_road, mask == 129)
        #mask_sky = mask == 146
        mask_build = torch.logical_or(mask == 147, mask == 85)
        #mask_tree = mask == 158
        mask_all = torch.stack([mask_road,mask_car,mask_build,mask_tra,mask_per])#,mask_build,mask_tree
        mask_all = torch.cat([(~torch.any(mask_all, axis=0)).unsqueeze(0), mask_all])


        return mask_all*1.0
    def get_label_mask2former(self,mask):
        mask = torch.tensor(np.array(mask))
        vi = np.array(mask)
        mask_road = mask==1
        mask_car = torch.logical_and(mask < 20, mask > 14)
        mask_build = mask==3
        mask_tra = torch.logical_or(mask == 7, mask == 8)
        mask_per = mask==12
        mask_all = torch.stack([mask_road, mask_car, mask_build, mask_tra, mask_per])  # ,mask_build,mask_tree
        mask_all = torch.cat([(~torch.any(mask_all, axis=0)).unsqueeze(0), mask_all])
        return mask_all * 1.0
    def get_img_path(self,samp,is_new,ori_data=None,ori_data_re =None):
        if is_new:
            if ori_data == 'Perl':
                fname = samp['filename']
                names = os.path.split(fname)
                img_name = names[1].split('.')[0]
                img_name = img_name + '.' + 'png'
                imgname = os.path.join(self.perl_file_name, names[0], img_name)
                if self.pv_mask_path =='san':
                    path = os.path.join(self.perl_file_name,'new_nuscenes_semanti_mask')#'/data2/lsy/nus_gen_data/new_nuscenes_semanti_mask'
                    cam_path = os.path.join(path, samp['filename'][8:])
                elif self.pv_mask_path =='mask2former':
                    path =  os.path.join(self.perl_file_name,'new_nuscenes_semanti_mask_mask2former')#'/data2/lsy/nus_gen_data/new_nuscenes_semanti_mask_mask2former'
                    img_name = samp['filename'][8:].split('.')[0]
                    img_name = img_name + '.png'
                    cam_path = os.path.join(path,img_name )
                image_gray = Image.open(cam_path)
            elif ori_data == 'Magic':
                fname = samp['filename']
                names = os.path.split(fname)
                img_name = names[1].split('.')[0]
                img_name = img_name + '_gen_0.jpg'
                imgname = os.path.join(self.magic_file_name, names[0], img_name)
                if self.pv_mask_path =='san':
                    path = os.path.join(self.magic_file_name,'new_semantic_mask')#'/data2/lsy/magic_gen_data/new_semantic_mask'
                elif self.pv_mask_path == 'mask2former':
                    path= os.path.join(self.magic_file_name,'new_semantic_mask_mask2former')#'/data2/lsy/magic_gen_data/new_semantic_mask_mask2former'
                camfolder = os.path.split(names[0])
                cam_path = os.path.join(path, camfolder[1], img_name)
                image_gray = Image.open(cam_path)
            elif ori_data == 'PerMag':
                if ori_data_re == 'Perl':
                    fname = samp['filename']
                    names = os.path.split(fname)
                    img_name = names[1].split('.')[0]
                    img_name = img_name + '.' + 'png'
                    imgname = os.path.join(self.perl_file_name, names[0], img_name)
                    if self.pv_mask_path == 'san':
                        path = os.path.join(self.perl_file_name, 'new_nuscenes_semanti_mask')  # '/data2/lsy/nus_gen_data/new_nuscenes_semanti_mask'
                    elif self.pv_mask_path == 'mask2former':
                        path = os.path.join(self.perl_file_name, 'new_nuscenes_semanti_mask_mask2former')
                    cam_path = os.path.join(path, samp['filename'][8:])
                    image_gray = Image.open(cam_path)
                else:
                    fname = samp['filename']
                    names = os.path.split(fname)
                    img_name = names[1].split('.')[0]
                    img_name = img_name + '_gen_0.jpg'
                    imgname = os.path.join(self.magic_file_name, names[0], img_name)
                    if self.pv_mask_path == 'san':
                        path = os.path.join(self.magic_file_name, 'new_semantic_mask')  # '/data2/lsy/magic_gen_data/new_semantic_mask'
                    elif self.pv_mask_path == 'mask2former':
                        path = os.path.join(self.magic_file_name, 'new_semantic_mask_mask2former')
                    camfolder = os.path.split(names[0])
                    cam_path = os.path.join(path, camfolder[1], img_name)
                    image_gray = Image.open(cam_path)
            elif ori_data == 'Ori':
                imgname = os.path.join(self.nusc.dataroot, samp['filename'])
                if self.pv_mask_path == 'san':
                    path = '/data1/lsy/nuscenes_semanti_mask'
                elif self.pv_mask_path == 'mask2former':
                    path = '/data1/lsy/nuscenes_semanti_mask_mask2former'

                cam_path = os.path.join(path, samp['filename'][8:])
                image_gray = Image.open(cam_path)
            elif ori_data == 'StableDiff':
                fname = samp['filename']
                names = os.path.split(fname)
                img_name = names[1].split('.')[0]
                img_name = img_name + '.' + 'png'
                imgname = os.path.join('/data2/lsy/SD_gen_data', names[0], img_name)
                if self.pv_mask_path =='san':
                    path = '/data2/lsy/SD_gen_data/nuscenes_semanti_mask'
                    cam_path = os.path.join(path, samp['filename'][8:])
                elif self.pv_mask_path =='mask2former':
                    path = '/data2/lsy/SD_gen_data/nuscenes_semanti_mask_mask2former'
                    img_name = samp['filename'][8:].split('.')[0]
                    img_name = img_name + '.png'
                    cam_path = os.path.join(path,img_name )
                image_gray = Image.open(cam_path)
            elif ori_data == 'BEVCon':
                fname = samp['filename']
                names = os.path.split(fname)
                img_name = names[1].split('.')[0]
                img_name = img_name + '.' + 'png'
                imgname = os.path.join('/data2/lsy/BEVCon_gen_data', names[0], img_name)
                if self.pv_mask_path =='san':
                    path = '/data2/lsy/BEVCon_gen_data/nuscenes_semanti_mask'
                    cam_path = os.path.join(path, samp['filename'][8:])
                elif self.pv_mask_path =='mask2former':
                    path = '/data2/lsy/BEVCon_gen_data/nuscenes_semanti_mask_mask2former'
                    img_name = samp['filename'][8:].split('.')[0]
                    img_name = img_name + '.png'
                    cam_path = os.path.join(path,img_name )
                image_gray = Image.open(cam_path)

        else:
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            if self.pv_mask_path == 'san':
                path = '/data1/lsy/nuscenes_semanti_mask'
            elif self.pv_mask_path == 'mask2former':
                path = '/data1/lsy/nuscenes_semanti_mask_mask2former'

            cam_path = os.path.join(path, samp['filename'][8:])
            image_gray = Image.open(cam_path)
        return imgname,image_gray
    def get_image_data(self, rec, cams,prev_rec=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        dep_image=[]
        seg_mask = []
        is_new = rec['new_data']
        token_name = rec['token']
        c = 0
        ori_data = rec['ori']
        fre_change = False
        # if self.data_aug_conf['fre_change'] and np.random.choice([0, 1]) and self.is_train:
        #     fre_change = True
        if ori_data == 'PerMag':
            if np.random.choice([0, 1]):
                ori_data_re = 'Perl'
            else:
                ori_data_re = 'Magic'
        else:
            ori_data_re = None
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])

            imgname, image_gray = self.get_img_path(samp,is_new,ori_data,ori_data_re)
            img = Image.open(imgname)
            if self.data_aug_conf['fre_change'] and prev_rec is not None:
                prev_samp = self.nusc.get('sample_data', prev_rec['data'][cam])
                prev_imgname, _ = self.get_img_path(prev_samp, is_new, ori_data)
                prev_img = Image.open(prev_imgname)
            else:
                prev_img=None

            # if is_new and self.select_data and ori_data == 'PerMag':
            #     if token_name in self.new_data_iou[ori_data_re]:
            #         cam_flag = self.new_data_iou[ori_data_re][token_name][c]
            #         if not cam_flag:
            #             img = Image.new("RGB", (1600, 900), color=(0, 0, 0))  # image = Image.new("RGB", (width, height), color=(0, 0, 0))
            c+=1
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            if True:
                resize, resize_dims, crop, flip, rotate, colorjt,gauss,_ = self.sample_augmentation(is_new)
                #print(cam,fre_change)
                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,False,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate,
                                                           colorjt=colorjt,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf'],
                                                            fre_change=fre_change,
                                                           prev_img=prev_img,
                                                           )
                image_gray = np.array(image_gray) + 1
                image_gray = Image.fromarray(image_gray.astype(np.uint8))
                mask, _,_ = img_transform(image_gray, post_rot, post_tran,False,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate,
                                                           colorjt=False,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf'])



            if self.pv_mask_path == 'san':
                mask = self.get_label(mask)
            elif self.pv_mask_path == 'mask2former':
                mask = self.get_label_mask2former(mask)
            else:
                print('error')
            if self.data_aug_conf['Ncams']==6:
                seg_mask.append(mask)
                dep_image.append(normalize_img(img))
                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs.append(normalize_img(img))
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)


        return (torch.stack(dep_image),torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans),torch.stack(seg_mask),ori_data_re)

    def get_binimg(self, rec,mask):
        img = np.zeros((self.nx[0], self.nx[1]))

        egopose = self.nusc.get('ego_pose',
                                    self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])#ego2global_translation
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        for tok in rec['anns']:
                inst = self.nusc.get('sample_annotation', tok)

                if not inst['category_name'].split('.')[0] == 'vehicle':
                        continue
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
        img = torch.Tensor(img)


        return img

    def get_divider(self,rec,mask):
        img = np.zeros((self.nx[0], self.nx[1]))
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = []
        line_names = ['road_divider', 'lane_divider']
        lmap = get_local_map(self.nusc_maps[map_name], center,
                             50.0, poly_names, line_names)
        for name in poly_names:
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
        for name in line_names:
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                # valid_pts = np.logical_and((pts[:, 0] < 200), np.logical_and((pts[:, 0] >= 0),
                #             np.logical_and((pts[:, 1] >= 0), (pts[:, 1] < 200))))
                # img[pts[valid_pts, 0], pts[valid_pts, 1]] = 1.0
                # cv2.fillPoly(img, [pts], 1.0)
                cv2.polylines(img, [pts], isClosed=False, color=1.0, thickness=2)
        img = torch.Tensor(img)
        # img = torch.flip(img,dims=[1])
        mask.append(img)
        return mask

    def get_static(self,rec):

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = ['road_segment','lane', 'ped_crossing', 'walkway', 'stop_line','carpark_area']#
        line_names = []
        lmap = get_local_map(self.nusc_maps[map_name], center,
                             50, poly_names, line_names)
        mask=[]

        img = np.zeros((self.nx[0], self.nx[1]))
        for name in ['road_segment','lane']:
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
        mask.append(torch.Tensor(img))

        for name in ['ped_crossing', 'walkway', 'stop_line','carpark_area']:
            img = np.zeros((self.nx[0], self.nx[1]))
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
            mask.append(torch.Tensor(img))


        return mask

    def get_pose(self,sample,meta):
        sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        meta['ego2global_translation']=torch.tensor(pose_record['translation'])
        meta['ego2global_rotation'] = Quaternion(pose_record['rotation'])
        return meta
    def get_lidar_data_to_img(self, rec, post_rots, post_trans, cams, downsample, nsweeps):
        points_imgs,_,_ = get_lidar_data_to_img(self.nusc, rec, data_aug_conf=self.data_aug_conf,
                                            grid_conf=self.grid_conf, post_rots=post_rots, post_trans=post_trans,
                                            cams=cams, downsample=downsample, nsweeps=nsweeps, min_distance=2.2,)
        return torch.Tensor(points_imgs)



    def get_vector(self,rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])[
            'location']  # 此场景日志信息，采集地名字
        # ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])  # 车本身在世界坐标系位姿
        sd_rec = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])  # 车本身在世界坐标系位姿
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        lidar2ego[:3, 3] = cs_record['translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
        ego2global[:3, 3] = ego_pose['translation']

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        osm_vectors = self.osmvect.gen_vectorized_samples(location, lidar2global_translation,lidar2global_rotation)  # lidar2global_translation, lidar2global_rotation
        return osm_vectors



    def __getitem__(self, index):
        #print('dd',index)
        rec = self.ixes[index]
        if rec['prev'] == '':
            prev_rec=None
        else:
            prev_rec = self.ixes[index-1]
        cams = self.choose_cams()
        depimg,imgs, rots, trans, intrins, post_rots, post_trans,seg_mask,ori_data_re = self.get_image_data(rec, cams,prev_rec)
        if self.is_lidar:
            lidar = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)
        else:
            lidar=torch.tensor([0])

        mask = self.get_static(rec)
        gt = self.get_divider(rec,mask)
        gt = torch.stack(gt)

        car = self.get_binimg(rec,None)
        gt = torch.cat([gt,car.unsqueeze(0)],dim=0)

        meta=dict()
        scene_record = self.nusc.get('scene', rec['scene_token'])
        if rec['new_data']:

            meta['scene_name'] = rec['ori'] +'_'+ scene_record['name']
        else:
            meta['scene_name']= scene_record['name']
        meta = self.get_pose(rec,meta)
        if rec['new_data']:
            if ori_data_re is not None:

                meta['weight'] = 1 if self.new_data_iou[ori_data_re][rec['token']] >0.85 else self.new_data_iou[ori_data_re][rec['token']]
            else:
                if rec['ori']=='Ori':
                    meta['weight'] = 1.0
                else:
                    meta['weight'] =1 if self.new_data_iou[rec['ori']][rec['token']] >0.85 else self.new_data_iou[rec['ori']][rec['token']]
        else:
            meta['weight'] = 1.0
        meta['token'] = rec
        return imgs, rots, trans, intrins, post_rots, post_trans,lidar,gt,seg_mask,meta

    def __len__(self):

        return len(self.ixes)

class SemanticNuscData_Day(SemanticNuscData):

    def get_img_path(self,samp,is_new,ori_data=None,ori_data_re =None):
        if is_new:
            if ori_data == 'Perl':
                fname = samp['filename']
                names = os.path.split(fname)
                img_name = names[1].split('.')[0]
                img_name = img_name + '.' + 'png'
                imgname = os.path.join('/data2/lsy/nus_gen_night_data', names[0], img_name)
                if self.pv_mask_path =='san':
                    path = '/data2/lsy/nus_gen_night_data/nuscenes_semanti_mask'
                    cam_path = os.path.join(path, samp['filename'][8:])
                elif self.pv_mask_path =='mask2former':
                    path = '/data2/lsy/nus_gen_night_data/nuscenes_semanti_mask_mask2former'
                    img_name = samp['filename'][8:].split('.')[0]
                    img_name = img_name + '.png'
                    cam_path = os.path.join(path,img_name )
                image_gray = Image.open(cam_path)
            elif ori_data == 'Magic':
                fname = samp['filename']
                names = os.path.split(fname)
                img_name = names[1].split('.')[0]
                img_name = img_name + '_gen_0.jpg'
                imgname = os.path.join('/data2/lsy/magic_gen_night_data', names[0], img_name)
                if self.pv_mask_path =='san':
                    path = '/data2/lsy/magic_gen_night_data/nuscenes_semanti_mask'
                elif self.pv_mask_path == 'mask2former':
                    path= '/data2/lsy/magic_gen_night_data/nuscenes_semanti_mask_mask2former'
                camfolder = os.path.split(names[0])
                cam_path = os.path.join(path, camfolder[1], img_name)
                image_gray = Image.open(cam_path)
            elif ori_data == 'PerMag':
                if ori_data_re == 'Perl':
                    fname = samp['filename']
                    names = os.path.split(fname)
                    img_name = names[1].split('.')[0]
                    img_name = img_name + '.' + 'png'
                    imgname = os.path.join('/data2/lsy/nus_gen_night_data', names[0], img_name)
                    if self.pv_mask_path == 'san':
                        path = '/data2/lsy/nus_gen_night_data/nuscenes_semanti_mask'
                        cam_path = os.path.join(path, samp['filename'][8:])
                    elif self.pv_mask_path == 'mask2former':
                        path = '/data2/lsy/nus_gen_night_data/nuscenes_semanti_mask_mask2former'
                        img_name = samp['filename'][8:].split('.')[0]
                        img_name = img_name + '.png'
                        cam_path = os.path.join(path, img_name)
                    image_gray = Image.open(cam_path)
                else:
                    fname = samp['filename']
                    names = os.path.split(fname)
                    img_name = names[1].split('.')[0]
                    img_name = img_name + '_gen_0.jpg'
                    imgname = os.path.join('/data2/lsy/magic_gen_night_data', names[0], img_name)
                    if self.pv_mask_path == 'san':
                        path = '/data2/lsy/magic_gen_night_data/nuscenes_semanti_mask'
                    elif self.pv_mask_path == 'mask2former':
                        path = '/data2/lsy/magic_gen_night_data/nuscenes_semanti_mask_mask2former'
                    camfolder = os.path.split(names[0])
                    cam_path = os.path.join(path, camfolder[1], img_name)
                    image_gray = Image.open(cam_path)
            elif ori_data == 'Ori':
                imgname = os.path.join(self.nusc.dataroot, samp['filename'])
                if self.pv_mask_path == 'san':
                    path = '/data1/lsy/nuscenes_semanti_mask'
                elif self.pv_mask_path == 'mask2former':
                    path = '/data1/lsy/nuscenes_semanti_mask_mask2former'

                cam_path = os.path.join(path, samp['filename'][8:])
                image_gray = Image.open(cam_path)

        else:
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            if self.pv_mask_path == 'san':
                path = '/data1/lsy/nuscenes_semanti_mask'
            elif self.pv_mask_path == 'mask2former':
                path = '/data1/lsy/nuscenes_semanti_mask_mask2former'

            cam_path = os.path.join(path, samp['filename'][8:])
            image_gray = Image.open(cam_path)
        return imgname,image_gray

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        depimg,imgs, rots, trans, intrins, post_rots, post_trans,seg_mask,ori_data_re = self.get_image_data(rec, cams)
        if self.is_lidar:
            lidar = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)
        else:
            lidar=torch.tensor([0])

        mask = self.get_static(rec)
        gt = self.get_divider(rec,mask)
        gt = torch.stack(gt)

        car = self.get_binimg(rec,None)
        gt = torch.cat([gt,car.unsqueeze(0)],dim=0)

        meta=dict()
        scene_record = self.nusc.get('scene', rec['scene_token'])
        if rec['new_data']:
            # if self.select_data and ori_data_re is not None:
            #     if rec['token'] in self.new_data_iou[ori_data_re]:
            #         gt = self.get_local_gt(gt, rec, name=ori_data_re)
            meta['scene_name'] = rec['ori'] +'_'+ scene_record['name']
        else:
            meta['scene_name']= scene_record['name']
        meta = self.get_pose(rec,meta)
        if rec['new_data']:
            if ori_data_re is not None:

                meta['weight'] = 1 if self.new_data_iou[ori_data_re][rec['token']] >0.9 else self.new_data_iou[ori_data_re][rec['token']]
            else:
                if rec['ori']=='Ori':
                    meta['weight'] = 1.0
                else:
                    meta['weight'] =1 if self.new_data_iou[rec['ori']][rec['token']] >0.9 else self.new_data_iou[rec['ori']][rec['token']]
        else:
            meta['weight'] = 1.0

        return imgs, rots, trans, intrins, post_rots, post_trans,lidar,gt,seg_mask,meta

class SemanticNuscData_Dry(SemanticNuscData):

    def get_img_path(self,samp,is_new,ori_data=None,ori_data_re =None):
        if is_new:
            if ori_data == 'Perl':
                fname = samp['filename']
                names = os.path.split(fname)
                img_name = names[1].split('.')[0]
                img_name = img_name + '.' + 'png'
                imgname = os.path.join('/data2/lsy/nus_gen_rain_data', names[0], img_name)
                if self.pv_mask_path =='san':
                    path = '/data2/lsy/nus_gen_rain_data/nuscenes_semanti_mask'
                    cam_path = os.path.join(path, samp['filename'][8:])
                elif self.pv_mask_path =='mask2former':
                    path = '/data2/lsy/nus_gen_rain_data/nuscenes_semanti_mask_mask2former'
                    img_name = samp['filename'][8:].split('.')[0]
                    img_name = img_name + '.png'
                    cam_path = os.path.join(path,img_name )
                image_gray = Image.open(cam_path)
            elif ori_data == 'Magic':
                fname = samp['filename']
                names = os.path.split(fname)
                img_name = names[1].split('.')[0]
                img_name = img_name + '_gen_0.jpg'
                imgname = os.path.join('/data2/lsy/magic_gen_night_data', names[0], img_name)
                if self.pv_mask_path =='san':
                    path = '/data2/lsy/magic_gen_rain_data/nuscenes_semanti_mask'
                elif self.pv_mask_path == 'mask2former':
                    path= '/data2/lsy/magic_gen_rain_data/nuscenes_semanti_mask_mask2former'
                camfolder = os.path.split(names[0])
                cam_path = os.path.join(path, camfolder[1], img_name)
                image_gray = Image.open(cam_path)
            elif ori_data == 'PerMag':
                if ori_data_re == 'Perl':
                    fname = samp['filename']
                    names = os.path.split(fname)
                    img_name = names[1].split('.')[0]
                    img_name = img_name + '.' + 'png'
                    imgname = os.path.join('/data2/lsy/nus_gen_rain_data', names[0], img_name)
                    if self.pv_mask_path == 'san':
                        path = '/data2/lsy/nus_gen_rain_data/nuscenes_semanti_mask'
                        cam_path = os.path.join(path, samp['filename'][8:])
                    elif self.pv_mask_path == 'mask2former':
                        path = '/data2/lsy/nus_gen_rain_data/nuscenes_semanti_mask_mask2former'
                        img_name = samp['filename'][8:].split('.')[0]
                        img_name = img_name + '.png'
                        cam_path = os.path.join(path, img_name)
                    image_gray = Image.open(cam_path)
                else:
                    fname = samp['filename']
                    names = os.path.split(fname)
                    img_name = names[1].split('.')[0]
                    img_name = img_name + '_gen_0.jpg'
                    imgname = os.path.join('/data2/lsy/magic_gen_rain_data', names[0], img_name)
                    if self.pv_mask_path == 'san':
                        path = '/data2/lsy/magic_gen_rain_data/nuscenes_semanti_mask'
                    elif self.pv_mask_path == 'mask2former':
                        path = '/data2/lsy/magic_gen_rain_data/nuscenes_semanti_mask_mask2former'
                    camfolder = os.path.split(names[0])
                    cam_path = os.path.join(path, camfolder[1], img_name)
                    image_gray = Image.open(cam_path)
            elif ori_data == 'Ori':
                imgname = os.path.join(self.nusc.dataroot, samp['filename'])
                if self.pv_mask_path == 'san':
                    path = '/data1/lsy/nuscenes_semanti_mask'
                elif self.pv_mask_path == 'mask2former':
                    path = '/data1/lsy/nuscenes_semanti_mask_mask2former'

                cam_path = os.path.join(path, samp['filename'][8:])
                image_gray = Image.open(cam_path)

        else:
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            if self.pv_mask_path == 'san':
                path = '/data1/lsy/nuscenes_semanti_mask'
            elif self.pv_mask_path == 'mask2former':
                path = '/data1/lsy/nuscenes_semanti_mask_mask2former'

            cam_path = os.path.join(path, samp['filename'][8:])
            image_gray = Image.open(cam_path)
        return imgname,image_gray

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        depimg,imgs, rots, trans, intrins, post_rots, post_trans,seg_mask,ori_data_re = self.get_image_data(rec, cams)
        if self.is_lidar:
            lidar = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)
        else:
            lidar=torch.tensor([0])

        mask = self.get_static(rec)
        gt = self.get_divider(rec,mask)
        gt = torch.stack(gt)

        car = self.get_binimg(rec,None)
        gt = torch.cat([gt,car.unsqueeze(0)],dim=0)

        meta=dict()
        scene_record = self.nusc.get('scene', rec['scene_token'])
        if rec['new_data']:
            # if self.select_data and ori_data_re is not None:
            #     if rec['token'] in self.new_data_iou[ori_data_re]:
            #         gt = self.get_local_gt(gt, rec, name=ori_data_re)
            meta['scene_name'] = rec['ori'] +'_'+ scene_record['name']
        else:
            meta['scene_name']= scene_record['name']
        meta = self.get_pose(rec,meta)
        if rec['new_data']:
            if ori_data_re is not None:

                meta['weight'] = 1 if self.new_data_iou[ori_data_re][rec['token']] >1.0 else self.new_data_iou[ori_data_re][rec['token']]
            else:
                if rec['ori']=='Ori':
                    meta['weight'] = 1.0
                else:
                    meta['weight'] =1 if self.new_data_iou[rec['ori']][rec['token']] >1.0 else self.new_data_iou[rec['ori']][rec['token']]
        else:
            meta['weight'] = 1.0

        return imgs, rots, trans, intrins, post_rots, post_trans,lidar,gt,seg_mask,meta



class FlipSemanticNuscData(SemanticNuscData):
    def get_image_data(self, rec, cams,prev_rec=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        dep_image=[]
        post_rots_flip = []
        post_trans_flip = []
        imgs_flip = []
        seg_mask = []
        seg_mask_flip = []
        seg_mask_3 = []
        post_rots_3 = []
        post_trans_3 = []
        imgs_3 = []
        flip_g = []
        fre_change = False
        if self.data_aug_conf['fre_change'] and np.random.choice([0, 1]) and self.is_train:
            fre_change = True
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            path = '/data1/lsy/nuscenes_semanti_mask'
            cam_path = os.path.join(path, samp['filename'][8:])
            mask_o = Image.open(cam_path)

            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img_o = Image.open(imgname)
            if self.data_aug_conf['fre_change'] and prev_rec is not None:
                prev_samp = self.nusc.get('sample_data', prev_rec['data'][cam])
                prev_imgname = os.path.join(self.nusc.dataroot, prev_samp['filename'])
                prev_img = Image.open(prev_imgname)
            else:
                prev_img = None
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            if self.aug_mode == 'hard':
                mask_o = np.array(mask_o) + 1
                mask_o = Image.fromarray(mask_o.astype(np.uint8))
                resize, resize_dims, crop, flip, rotate, colorjt,guass,_ = self.sample_augmentation()
                img, post_rot2, post_tran2 = img_transform(img_o, post_rot, post_tran,guass,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate,
                                                           colorjt=colorjt,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf'],
                                                           fre_change=fre_change,prev_img=prev_img
                                                           )
                mask,_,_ = img_transform(mask_o, post_rot, post_tran,0,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate,
                                                           colorjt=False,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf']
                                                           )
                if self.pv_mask_path == 'san':
                    mask = self.get_label(mask)
                elif self.pv_mask_path == 'mask2former':
                    mask = self.get_label_mask2former(mask)
                seg_mask.append(mask)
                # img_3, post_rot3, post_tran3 = img_transform(img_o, post_rot, post_tran, guass,
                #                                            resize=resize,
                #                                            resize_dims=resize_dims,
                #                                            crop=crop,
                #                                            flip=flip,
                #                                            rotate=0,
                #                                            colorjt=colorjt,
                #                                            colorjt_conf=self.data_aug_conf['color_jitter_conf']
                #                                            )
                # mask_3, _, _ = img_transform(mask_o, post_rot, post_tran, 0,
                #                            resize=resize,
                #                            resize_dims=resize_dims,
                #                            crop=crop,
                #                            flip=flip,
                #                            rotate=0,
                #                            colorjt=False,
                #                            colorjt_conf=self.data_aug_conf['color_jitter_conf']
                #                            )
                # if self.pv_mask_path == 'san':
                #     mask_3 = self.get_label(mask_3)
                # elif self.pv_mask_path == 'mask2former':
                #     mask_3 = self.get_label_mask2former(mask_3)
                # seg_mask_3.append(mask_3)
                resize, resize_dims,_,_ = self.sample_augmentation_simple()
                img_flip, post_rot2_flip, post_tran2_flip = img_transform_simple(img_o, resize, resize_dims)
                mask_o, _, _ = img_transform_simple(mask_o, resize, resize_dims)
                if self.pv_mask_path == 'san':
                    mask_o = self.get_label(mask_o)
                elif self.pv_mask_path == 'mask2former':
                    mask_o = self.get_label_mask2former(mask_o)
                seg_mask_flip.append(mask_o)


            flip_g.append(1*flip)


            dep_image.append(normalize_img(img))
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            post_tran_f = torch.zeros(3)
            post_rot_f = torch.eye(3)
            post_tran_f[:2] = post_tran2_flip
            post_rot_f[:2, :2] = post_rot2_flip

            # post_tran_3 = torch.zeros(3)
            # post_rot_3 = torch.eye(3)
            # post_tran_3[:2] = post_tran3
            # post_rot_3[:2, :2] = post_rot3

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            imgs_flip.append(normalize_img(img_flip))
            post_trans_flip.append(post_tran_f)
            post_rots_flip.append(post_rot_f)

            # imgs_3.append(normalize_img(img_3))
            # post_trans_3.append(post_tran_3)
            # post_rots_3.append(post_rot_3)

        return (torch.stack(dep_image),torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans),torch.stack(imgs_flip),torch.stack(post_trans_flip),torch.stack(post_rots_flip),
                torch.stack(seg_mask),torch.stack(seg_mask_flip),
                )

    def get_inv_gt(self,mask):
        mask_road = mask[0]
        mask_inv = mask_road-mask
        mask_inv[0]=1-mask[0]
        mask_inv[2] = mask_road
        return mask_inv

    def __getitem__(self, index):
        rec = self.ixes[index]
        if rec['prev'] =='':
            prev_rec=None
        else:
            prev_rec = self.ixes[index-1]
        cams = self.choose_cams()
        depimg,imgs, rots, trans, intrins, post_rots, post_trans,imgs_flip,post_trans_flip,post_rots_flip,seg_mask,seg_mask_o,= self.get_image_data(rec, cams,prev_rec=prev_rec)
        lidar = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)

        mask = self.get_static(rec)
        gt = self.get_divider(rec, mask)
        gt = torch.stack(gt)
        #gt = torch.cat([(~torch.any(gt, axis=0)).unsqueeze(0), gt])
        car = self.get_binimg(rec, None)
        gt = torch.cat([gt,car.unsqueeze(0)], dim=0)
        meta = dict()
        scene_record = self.nusc.get('scene', rec['scene_token'])
        meta['scene_name'] = scene_record['name']
        meta = self.get_pose(rec, meta)
        return imgs, rots, trans, intrins, post_rots, post_trans,lidar,gt,imgs_flip,post_rots_flip,post_trans_flip,seg_mask,seg_mask_o,meta

def collate_wrapper(batch):
    if len(batch[0])==10:
        imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s, = [],[],[],[],[],[],[],[],[],
        img_meta = [dict() for i in range(len(batch))]
        batch_i = 0
        for img, rot, tran, intrin, post_rot, post_tran, lidar, binimgs, seg_mask,meta in batch:
            imgs.append(img)
            trans.append(tran)
            rots.append(rot)
            intrins.append(intrin)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            lidars.append(lidar)
            binimgs_s.append(binimgs)
            seg_mask_s.append(seg_mask)
            img_meta[batch_i]['scene_name']=meta['scene_name']
            img_meta[batch_i]['weight'] = meta['weight']
            img_meta[batch_i]['ego2global_translation'] = meta['ego2global_translation']
            img_meta[batch_i]['ego2global_rotation'] = meta['ego2global_rotation']
            batch_i += 1
        imgs = torch.stack(imgs)
        trans = torch.stack(trans)
        rots = torch.stack(rots)
        intrins = torch.stack(intrins)
        post_trans = torch.stack(post_trans)
        post_rots = torch.stack(post_rots)
        lidars = torch.stack(lidars)
        binimgs_s = torch.stack(binimgs_s)
        seg_mask_s = torch.stack(seg_mask_s)
        return imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s,img_meta
    if len(batch[0])==11:
        imgs_o,imgs, rots, trans, intrins, post_rots, post_trans, binimgs_s, seg_mask_s,osm_s = [],[],[],[],[],[],[],[],[],[]
        img_meta = [dict() for i in range(len(batch))]
        batch_i = 0
        for img_o,img, rot, tran, intrin, post_rot, post_tran,  binimgs, seg_mask,meta,osm in batch:
            imgs_o.append(img_o)
            imgs.append(img)
            trans.append(tran)
            rots.append(rot)
            intrins.append(intrin)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            binimgs_s.append(binimgs)
            seg_mask_s.append(seg_mask)
            osm_s.append(osm)
            img_meta[batch_i]['scene_name']=meta['scene_name']
            img_meta[batch_i]['weight'] = meta['weight']
            img_meta[batch_i]['ego2global_translation'] = meta['ego2global_translation']
            img_meta[batch_i]['ego2global_rotation'] = meta['ego2global_rotation']
            batch_i += 1
        imgs = torch.stack(imgs)
        imgs_o = torch.stack(imgs_o)
        trans = torch.stack(trans)
        rots = torch.stack(rots)
        intrins = torch.stack(intrins)
        post_trans = torch.stack(post_trans)
        post_rots = torch.stack(post_rots)
        binimgs_s = torch.stack(binimgs_s)
        seg_mask_s = torch.stack(seg_mask_s)
        osm_s = torch.stack(osm_s)
        return imgs_o,imgs, rots, trans, intrins, post_rots, post_trans,  binimgs_s, seg_mask_s,img_meta,osm_s

def collate_wrapper_flip(batch):

    imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s,  = [],[],[],[],[],[],[],[],
    img_oris, un_post_rots_oris, un_post_trans_oris, seg_mask_ts, seg_mask_os=[],[],[],[],[]
    img_meta = [dict() for i in range(len(batch))]
    batch_i = 0
    for img, rot, tran, intrin, post_rot, post_tran, lidar, binimgs,\
            img_ori, un_post_rots_ori, un_post_trans_ori, seg_mask_t, seg_mask_o,meta in batch:
        imgs.append(img)
        trans.append(tran)
        rots.append(rot)
        intrins.append(intrin)
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        lidars.append(lidar)
        binimgs_s.append(binimgs)
        img_oris.append(img_ori)
        un_post_trans_oris.append(un_post_trans_ori)
        un_post_rots_oris.append(un_post_rots_ori)
        seg_mask_ts.append(seg_mask_t)
        seg_mask_os.append(seg_mask_o)

        img_meta[batch_i]['scene_name']=meta['scene_name']
        img_meta[batch_i]['ego2global_translation'] = meta['ego2global_translation']
        img_meta[batch_i]['ego2global_rotation'] = meta['ego2global_rotation']
        batch_i += 1
    imgs = torch.stack(imgs)
    trans = torch.stack(trans)
    rots = torch.stack(rots)
    intrins = torch.stack(intrins)
    post_trans = torch.stack(post_trans)
    post_rots = torch.stack(post_rots)
    lidars = torch.stack(lidars)
    binimgs_s = torch.stack(binimgs_s)
    img_oris = torch.stack(img_oris)
    un_post_rots_oris = torch.stack(un_post_rots_oris)
    un_post_trans_oris = torch.stack(un_post_trans_oris)
    seg_mask_ts = torch.stack(seg_mask_ts)
    seg_mask_os = torch.stack(seg_mask_os)
    return imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, img_oris,un_post_rots_oris,un_post_trans_oris,seg_mask_ts,seg_mask_os,img_meta

import torch
import torch.nn.functional as F
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode
if __name__ == '__main__':

    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],  # [-30.0, 00.0, 0.15],#
        'ybound': [-50.0, 50.0, 0.5],  #
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 6,'H':900,'W':1600,
                     'up_scale': 4,'rand_resize': False,'resize_lim': (0.20, 0.235),
                     'rand_flip': False,'rot_lim':(-5.4, 5.4),'bot_pct_lim': (0.0, 0.22),
                     'color_jitter': False, 'color_jitter_conf': [0.2, 0.2, 0.2, 0.1],
                     'GaussianBlur': False, 'gaussion_c': (0, 2),
                      'final_dim': (128, 352),'Aug_mode':'hard',#'simple',#
                     'fre_change': True, 'is_lidar':False,
                     }
    source_name_list = ['boston', 'singapore', 'day', 'dry']
    target_name_list = ['singapore', 'boston', 'night', 'rain']
    n = 0
    source_name=source_name_list[n]
    target_name=target_name_list[n]
    data = SemanticNuscData(version='v1.0-mini',#'v1.0-trainval',#
                    dataroot='/data0/lsy/dataset/nuScenes/v1.0/',is_train=False,data_aug_conf=data_aug_conf,grid_conf=grid_conf,nsweeps=4,
                            domain_gap=False,domain=source_name, domain_type='strain',is_source=True,select_data=True,data_type=['Perl'],is_osm=True)
    l = data.__len__()
    for i in range(0,2):
        print(i)
        if i>-1:
            imgs,_ , _, _, _, _, _, binimgs_t, _, meta_prev = data[i]
            visual(binimgs_t[:6]>0.5,classes=['road_segment'])#
            s = torch.sum(binimgs_t,dim=(1,2))
            print(s)
            #vis_img(imgs)


    # n=150
    # _, _, _, _, _, _, _, binimgs_t, _, meta_prev = data[n]
    # visual(binimgs_t[:6]>0.5)
    # imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s, meta = data[n+3]
    # visual(binimgs_s[:6] > 0.5)
    # translation_prev = meta_prev['ego2global_translation']
    # rotation_prev = meta_prev['ego2global_rotation']
    # translation_curr = meta['ego2global_translation']
    # rotation_curr = meta['ego2global_rotation']
    # de = translation_curr - translation_prev
    # patch_angle = quaternion_yaw(rotation_curr) / np.pi * 180
    # rot = patch_angle / 180 * np.pi
    # patch_angle_pre = quaternion_yaw(rotation_prev) / np.pi * 180
    # delta_translation_global = torch.tensor(
    #     [de[0], de[1]]).unsqueeze(1)  # .permute(1, 0)#2 1
    # curr_transform_matrix = torch.tensor([[np.cos(-rot), -np.sin(-rot)],
    #                                       [np.sin(-rot), np.cos(-rot)]])  # (2, 2) 世界坐标转换为当前帧自车坐标
    #
    # delta_translation_curr = torch.matmul(
    #     curr_transform_matrix.float(),
    #     delta_translation_global.float()).squeeze()  # (bs, 2, 1) -> (bs, 2) = (bs, 2, 2）* (bs, 2, 1) (真实坐标)
    # delta_translation_curr = delta_translation_curr // 0.5  # (bs, 2) = (bs, 2) / (1, 2) (格子数)
    # delta_translation_curr = delta_translation_curr.round().tolist()
    #
    # theta = patch_angle-patch_angle_pre
    # tran = [-delta_translation_curr[1], -delta_translation_curr[0]]  # nuscenes是x为h，y为w
    # pre_warp = affine(binimgs_t,  # 必须是(c, h, w)形式的tensor
    #                   angle=theta,
    #                   # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
    #                   translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
    #                   scale=1,  # 浮点数，中心缩放尺度
    #                   shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
    #                   interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
    #                   fill=[0.0])  # 先旋转再平移
    #
    # # warped_feat = F.grid_sample(binimgs_t.unsqueeze(0),
    # #                             prev_coord.unsqueeze(0),
    # #                             padding_mode='zeros', align_corners=False).squeeze(0)
    # print(pre_warp.shape)
    # visual(pre_warp[:6]>0.5)
