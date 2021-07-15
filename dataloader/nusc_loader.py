import os
import random
import numpy as np
import torch
import PIL
from PIL import Image
from scipy import interpolate
from torchvision import transforms as T
from torch.utils.data import Dataset
from dataloader import nusc_utils
# import nusc_utils
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
random.seed(1984)


def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')
        

def read_list(list_path):
    print('read list from {}'.format(list_path))
    _list = []
    with open(list_path) as f:
        for line in f:
            _list.append(line.split('\n')[0])
            
    print(len(_list))
    return _list


def get_scene_token_list(nusc, split):
    scene_token_list = []
    for _scene_name in split:
        scene_token = nusc_utils.getSceneToken(nusc, _scene_name)
        scene_token_list.append(scene_token)
    return scene_token_list


def resize_depth(depth):
    depth = np.array(depth)
    re_depth = np.zeros((450,800))
    pts = np.where(depth!=0)
    re_depth[(pts[0][:]/2).astype(np.int), (pts[1][:]/2).astype(np.int)] = depth[pts[0][:], pts[1][:]]
    
    return re_depth


def get_samples(nusc, scene_token_list, cam_channels):
    samples = []
    for scene_token in scene_token_list:
        scene = nusc.get('scene',scene_token)
        sample_token = scene['first_sample_token']
        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            # get image file path
            for cam in cam_channels:
                camera_token = sample['data'][cam]
                lidar_token = sample['data']['LIDAR_TOP']
                desc = nusc_utils.getDescription(nusc, camera_token)
                RGB_path = os.path.join(nusc.dataroot, nusc_utils.getFileName(nusc, camera_token))
                sparse_path = os.path.join(nusc.dataroot, 'samples', cam.replace('CAM','SPARSE'), RGB_path.split('/')[-1].replace('jpg','png'))
                dense_path = os.path.join(nusc.dataroot, 'samples', cam.replace('CAM','DENSE'), RGB_path.split('/')[-1].replace('jpg','png'))
                radar_path = os.path.join(nusc.dataroot, 'samples', cam.replace('CAM','RADAR_EX5'), RGB_path.split('/')[-1].replace('jpg','png'))
                samples.append({
                    'scene': scene,
                    'scene_token': scene_token,
                    'sample_token': sample_token,
                    'desc': desc,
                    'channel': cam,
                    'camera_token': camera_token,
                    'lidar_token': lidar_token,
                    'RGB_path': RGB_path,
                    'sparse_path': sparse_path,
                    'dense_path': dense_path,
                    'radar_path': radar_path,
                })
            sample_token = sample['next']
        
    return samples


def load_data(sample):
    
    rgb = pil_loader(sample['RGB_path'], rgb=True)
    sparse = pil_loader(sample['sparse_path'], rgb=False)
    dense = pil_loader(sample['dense_path'], rgb=False)
    radar = pil_loader(sample['radar_path'], rgb=False)

    rgb = rgb.resize((800,450))
    radar = resize_depth(radar)

    rgb = np.array(rgb).astype(np.float32) / 255.
    sparse = np.array(sparse).astype(np.float32) / 256.
    dense = np.array(dense).astype(np.float32) / 256.
    radar = np.array(radar).astype(np.float32) / 256.

    return {'RGB': rgb,
            'SPARSE': sparse,
            'DENSE': dense,
            'RADAR': radar,
           }
    

def train_preprocess(image, slidar, dlidar, radar):
    
    # Random flipping
    do_flip = random.random()
    if do_flip > 0.5:
        image = (image[:, ::-1, :]).copy()
        slidar = (slidar[:, ::-1]).copy()
        dlidar = (dlidar[:, ::-1]).copy()
        radar = (radar[:, ::-1]).copy()

    # Random gamma, brightness, color augmentation
    do_augment = random.random()
    if do_augment > 0.5:
        image = augment_image(image)

    return image, slidar, dlidar, radar

def augment_image(image):
    
    # gamma augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness augmentation
    brightness = random.uniform(0.9, 1.1)   
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)

    return image_aug


class NuScenesLoader(Dataset):

    def __init__(self, 
                 scene_token_list='./list/nusc/train_scene.txt', 
                 data_root='/datasets/nuscenes/v1.0-trainval',
                 cam_channels=['CAM_FRONT'],
                 mode='train', size=(350, 800), nsweeps=5):
        super(NuScenesLoader, self).__init__()
        
        self.mode = mode
        self.size = size
        self.nsweeps = nsweeps
        
        self.scene_token_list = read_list(scene_token_list)
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
        self.samples = get_samples(self.nusc, self.scene_token_list, cam_channels)
        
        print('mode: {} with {} samples'.format(mode, len(self.samples)))


    def __len__(self):
        return len(self.samples)


    def train_procedure(self, data):
        
        # crop sky out
        rgb = data['RGB'][100::,:]
        sparse = data['SPARSE'][100::,:]
        dense = data['DENSE'][100::,:]
        radar = data['RADAR'][100::,:]
            
        # data augmentation
        rgb, sparse, dense, radar = train_preprocess(rgb, sparse, dense, radar)
        
        # cvt to pytorch tensor
        rgb = T.ToTensor()(rgb)
        rgb = T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(rgb)
        sparse = T.ToTensor()(sparse)
        dense = T.ToTensor()(dense)
        radar = T.ToTensor()(radar)

        return {'RGB': rgb,
                'SPARSE': sparse,
                'DENSE': dense,
                'RADAR': radar}
    
    def val_procedure(self, data):
        
        # crop sky out
        rgb = data['RGB'][100::,:]
        sparse = data['SPARSE'][100::,:]
        dense = data['DENSE'][100::,:]
        radar = data['RADAR'][100::,:]
    
        # cvt to pytorch tensor
        rgb = T.ToTensor()(rgb)
        rgb = T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(rgb)
        sparse = T.ToTensor()(sparse)
        dense = T.ToTensor()(dense)
        radar = T.ToTensor()(radar)

        return {'RGB': rgb,
                'SPARSE': sparse,
                'DENSE': dense,
                'RADAR': radar}

    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        data = load_data(sample)
#         pred_path = data['PRED_PATH']
        
        if self.mode == 'train':
            data = self.train_procedure(data)
        elif self.mode == 'val':
            data = self.val_procedure(data)
            
            
        return {'RGB': data['RGB'],
                'SPARSE_PATH': sample['sparse_path'],
#                 'PRED_PATH': pred_path,
                'DESC': sample['desc'],
                'SPARSE': data['SPARSE'],
                'DENSE': data['DENSE'],
                'RADAR': data['RADAR']}
  
