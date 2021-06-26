import os
# import mkl
import time
import scipy
import skimage
from skimage import color
from nuscenes.nuscenes import NuScenes
from scipy.interpolate import LinearNDInterpolator
import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

# from scipy_aliases import spsolve
# from scipy.sparse.linalg import spsolve

from pypardiso import spsolve

import multiprocessing as mp
from dataloader import nusc_utils


#
# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	#print ('Solving system..')

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	#print ('Done.')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
	return output

def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')
        
def readPathFiles(file_path, root_dir):
    im_gt_paths = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            cam_token = line.split()[0]
            im_path = os.path.join(root_dir, line.split()[1])
            gt_path = os.path.join(root_dir, line.split()[2])
            dense_path = os.path.join(root_dir, line.split()[3])

            im_gt_paths.append((cam_token, im_path, gt_path, dense_path))

    return im_gt_paths

def resize_depth(depth):
    depth = np.array(depth)
    re_depth = np.zeros((450,800))
    pts = np.where(depth!=0)
    re_depth[(pts[0][:]/2).astype(np.int), (pts[1][:]/2).astype(np.int)] = depth[pts[0][:], pts[1][:]]
    
    return re_depth

def loadRGB(sample, root_dir):
    im_path = os.path.join(root_dir, sample['RGB_path'])
    image = pil_loader(im_path, rgb=True)
    image = image.resize((800,450))
    image = np.array(image).astype(np.float32) / 255.
    return image

def loadSparseDepth(nusc, sample):
    cam_token = sample['camera_token']
    lidar_token = sample['lidar_token']
    points, coloring, im = nusc.explorer.map_pointcloud_to_image(pointsensor_token=lidar_token,camera_token=cam_token)
    sparse_depth = np.zeros((450,800))
    sparse_depth[(points[1][:]/2).astype(np.int), (points[0][:]/2).astype(np.int)] = coloring
    return sparse_depth


def saveDepth(depth, path):
    depth = depth*256
    depth = depth.astype(np.int32)
    depth = Image.fromarray(depth)
    depth = depth.convert('I')
    depth.save(path)    

def getDescription(nusc, camera_token):
    data = nusc.get('sample_data', camera_token)
    sample = nusc.get('sample', data['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    return scene['description'].lower()

def readList(filepath):
    print('readList from {}'.format(filepath))
    _list = []
    with open(filepath) as f:
        for line in f:
            _list.append(line.split('\n')[0])
            
    print(len(_list))
    return _list


def getFileName(nusc, camera_token):
    return nusc.get('sample_data', camera_token)['filename']


def checkIfDone(sample):
    sparse_path = os.path.join(DATA_ROOT, sample['sparse_path'])
    dense_path = os.path.join(DATA_ROOT, sample['dense_path'])
    return os.path.isfile(dense_path) and os.path.isfile(sparse_path)


def getRelSize(camera_token, d, w=0.5, h=1.5):
    cam = nusc.get('sample_data', camera_token)
    # get camera_intrinsic value
    camera_intrinsic = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])['camera_intrinsic']
    f_u = camera_intrinsic[0][0]
    f_v = camera_intrinsic[1][1]
    c_u = camera_intrinsic[0][2]
    c_v = camera_intrinsic[1][2]
    
    v = (h*f_v)/d
    u = (w*f_u)/d    
    
    return int(u),int(v)


def extendHeight(depth, camera_token, h0=0.25, h1=1.5, reverse=False):
    '''
    input depth and camera_token
    return heighht-extended radar depth
    '''
    Y,X,D = nusc_utils.getIndexValue(depth)
    sort_index = D.argsort()
    if reverse:
        sort_index = sort_index[::-1]
    
    ex_depth = np.zeros(depth.shape)
    for _index in sort_index:
        x = X[_index]
        y = Y[_index]
        d = D[_index]
        
        _,v1 = getRelSize(camera_token, d, 0, h1)
        _,v0 = getRelSize(camera_token, d, 0, h0)
        dv1 = int(v1)
        dv0 = int(v0)

        ex_depth[y-dv1:y+dv0,x] = d
        
    return ex_depth


def getHeightExtendRadarDepth(nusc, sample, nsweeps=5):
    sample_token = sample['sample_token']
    camera_token = sample['camera_token']
    lidar_token = sample['lidar_token']
    
    # radar_pc in lidar_top coord
    radar_pc, times = nusc_utils.read_radar(nusc, sample_token, nsweeps=nsweeps)

    _pc = LidarPointCloud(radar_pc.T.copy())
    _depth = nusc_utils.velo2depth(nusc, _pc, camera_token=camera_token, pointsensor_token=lidar_token)
    _depth_ex = extendHeight(_depth,camera_token)

    return _depth_ex

def process_data(idx):
    root_dir = '/datasets/nuscenes/v1.0-trainval'
    _sample = camera_samples[idx]
    image = loadRGB(_sample, root_dir)
    sparse_path = os.path.join(root_dir, _sample['sparse_path'])
    dense_path = os.path.join(root_dir, _sample['dense_path'])
    ex_radar_path = os.path.join(root_dir, _sample['ex_radar_path'])
    
    sparse_depth = loadSparseDepth(nusc, _sample)
    dense_depth = fill_depth_colorization(imgRgb=image, imgDepthInput=sparse_depth, alpha=1)
    ex_radar = getHeightExtendRadarDepth(nusc, _sample, nsweeps=5)
    
    saveDepth(sparse_depth, sparse_path)
    saveDepth(dense_depth, dense_path)
    saveDepth(ex_radar, ex_radar_path)


# load nusc 
DATA_ROOT = '/datasets/nuscenes/v1.0-trainval'
nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT, verbose=True)

CAM_CHANNELS = ['CAM_FRONT']
train = readList('./list/nusc/train_scene.txt')
val = readList('./list/nusc/val_scene.txt')
trainval = train + val
print('{} scenes in trainval'.format(len(trainval)))

# get all camera samples in given CAM_CHANNELS
camera_samples = []
for scene_token in trainval:
    scene = nusc.get('scene',scene_token)
    sample_token = scene['first_sample_token']
    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        # get image file path
        for cam in CAM_CHANNELS:
            camera_token = sample['data'][cam]
            lidar_token = sample['data']['LIDAR_TOP']
            RGB_path = getFileName(nusc, camera_token)
            sparse_path = os.path.join('samples', cam.replace('CAM','SPARSE'), RGB_path.split('/')[-1].replace('jpg','png'))
            dense_path = os.path.join('samples', cam.replace('CAM','DENSE'), RGB_path.split('/')[-1].replace('jpg','png'))
            ex_radar_path = os.path.join('samples', cam.replace('CAM','RADAR_EX5'), RGB_path.split('/')[-1].replace('jpg','png'))
            _sample = {
                'scene': scene,
                'scene_token': scene_token,
                'sample_token': sample_token,
                'channel': cam,
                'camera_token': camera_token,
                'lidar_token': lidar_token,
                'RGB_path': RGB_path,
                'sparse_path': sparse_path,
                'dense_path': dense_path,
                'ex_radar_path': ex_radar_path,
            }
#             if not checkIfDone(_sample):
#                 camera_samples.append(_sample)
            camera_samples.append(_sample)
                
                # the CAM_FRONT shoud be handled afterwards...
        sample_token = sample['next']

print(len(camera_samples))

#     cpu_count = mp.cpu_count()
cpu_count = 25

print('Number of processors: {}, and we are taking {}'.format(mp.cpu_count(), cpu_count))

pool = mp.Pool(cpu_count)
for _ in tqdm(pool.imap_unordered(process_data, range(len(camera_samples))), total=len(camera_samples)):
    pass
pool.close()
pool.join()
