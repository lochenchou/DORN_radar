from datetime import datetime
import os
import math
import time
import torch
import torch.nn as nn
import numpy as np
import utils
import random
from tqdm import tqdm

from metrics import AverageMeter, Result, compute_errors
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from utils import PolynomialLRDecay
from dataloader.nusc_loader import NuScenesLoader
from nuscenes.nuscenes import NuScenes
from loss import OrdinalRegressionLoss


def filter_radar_delta(trg, src, delta=1.25):
    delta = float(delta)
    
    # calculate upper and lower bound
    mask_lb = np.array(src.cpu()) > (np.array(trg.cpu()/delta))
    mask_ub = np.array(src.cpu()) < (np.array(trg.cpu()*delta))
    
    return mask_lb & mask_ub


# set arguments
BATCH_SIZE = 3
WORKERS = 3
SEED = 1984
SIZE = (350,800)
NSWEEPS = 5

ORD_NUM = 80
GAMMA = 0.3
ALPHA = 1
BETA = 80

# set random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# create output dir,
# _now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
output_dir = os.path.join('./result','dorn_radar'.format(NSWEEPS, SIZE[0], SIZE[1]))
test_dir = os.path.join(output_dir, 'self_val')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    
print('OUTPUT_DIR = {}'.format(output_dir))

# init dataloader
MODE = 'val'
DATA_ROOT = '/datasets/nuscenes/v1.0-trainval'
SCENE_VERSION = 'v1.0-trainval'
SCENE_TOKEN_LIST = './list/nusc/val_scene.txt'
CAM_CHANNELS=['CAM_FRONT']

nusc =  NuScenes(version=SCENE_VERSION, dataroot=DATA_ROOT, verbose=True)
test_set = NuScenesLoader(scene_token_list=SCENE_TOKEN_LIST,
                          data_root=DATA_ROOT, 
                          cam_channels=CAM_CHANNELS, 
                          mode=MODE, 
                          nsweeps=NSWEEPS,
                          scene_version=SCENE_VERSION,
                          nusc=nusc)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

# create model
CHECKPOINT = os.path.join('./pretrained_weight/','dorn_radar.pth.tar')
model = torch.load(CHECKPOINT,map_location="cpu")


print('GPU number: {}'.format(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if GPU number > 1, then use multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)
model.eval()


# loss function
ord_loss = OrdinalRegressionLoss(ord_num=ORD_NUM, beta=BETA)
         

avg80_sparse = AverageMeter()
avg80_dense = AverageMeter()


end = time.time()
skip =int(len(test_loader)/10)
img_list = []

pbar = tqdm(total=len(test_loader))

for i, data in enumerate(test_loader):
    _rgb, _sparse_depth, _dense_depth = data['RGB'].to(device), data['SPARSE'].to(device), data['DENSE'].to(device)
    _radar_depth = data['RADAR'].to(device)
    torch.cuda.synchronize()
    data_time = time.time() - end

    # compute output
    end = time.time()
    
    # stage1 prediction
    with torch.no_grad():
        _pred_prob, _pred_label = model(_rgb, _radar_depth)
    pred_depth = utils.label2depth_sid(_pred_label, K=ORD_NUM, alpha=1.0, beta=BETA, gamma=GAMMA)  

    # delta2 filtering based on stage1 prediction
    mask = filter_radar_delta(pred_depth, _radar_depth, delta=1.5625)
    _radar_depth = _radar_depth.cpu() * mask
    _radar_depth = _radar_depth.to(device)
    
    # stage 2 prediction
    with torch.no_grad():
        _pred_prob, _pred_label = model(_rgb, _radar_depth)

        loss = ord_loss(_pred_prob, _dense_depth)

    torch.cuda.synchronize()
    gpu_time = time.time() - end

    pred_depth = utils.label2depth_sid(_pred_label, K=ORD_NUM, alpha=1.0, beta=BETA, gamma=GAMMA)
    s_abs_rel, s_sq_rel, s_rmse, s_rmse_log, s_a1, s_a2, s_a3 = compute_errors(_sparse_depth, pred_depth.to(device))
    d_abs_rel, d_sq_rel, d_rmse, d_rmse_log, d_a1, d_a2, d_a3 = compute_errors(_dense_depth, pred_depth.to(device))

    # measure accuracy and record loss
    result80_sparse = Result()
    result80_sparse.evaluate(pred_depth, _sparse_depth.data, cap=80)
    avg80_sparse.update(result80_sparse, gpu_time, data_time, _rgb.size(0))

    result80_dense = Result()
    result80_dense.evaluate(pred_depth, _dense_depth.data, cap=80)
    avg80_dense.update(result80_dense, gpu_time, data_time, _rgb.size(0))

    end = time.time()

    # save images for visualization 
    if i == 0:
        img_merge = utils.merge_into_row_with_radar(_rgb, _radar_depth, _dense_depth, pred_depth)
    elif (i < 8 * skip) and (i % skip == 0):
        row = utils.merge_into_row_with_radar(_rgb, _radar_depth, _dense_depth, pred_depth)
        img_merge = utils.add_row(img_merge, row)
    elif i == 8 * skip:
        filename = os.path.join(test_dir,'test.png')
        print('save validation figures at {}'.format(filename))
        utils.save_image(img_merge, filename)

    # update progress bar and show loss
    pbar.set_postfix(ORD_LOSS='{:.2f}||DENSE||RMSE={:.2f},delta={:.2f}/{:.2f}|||SPARSE||RMSE={:.2f},delta={:.2f}/{:.2f}|'.format(loss,d_rmse,d_a1,d_a2,s_rmse,s_a1,s_a2))
    pbar.update(1)

    i = i+1

print('\n**** EVALUATE WITH SPARSE DEPTH ****\n'
      '\n**** CAP=80 ****\n'
      'RMSE={average.rmse:.3f}\n'
      'RMSE_log={average.rmse_log:.3f}\n'
      'AbsRel={average.absrel:.3f}\n'
      'SqRel={average.squared_rel:.3f}\n'
      'SILog={average.silog:.3f}\n'
      'Delta1={average.delta1:.3f}\n'
      'Delta2={average.delta2:.3f}\n'
      'Delta3={average.delta3:.3f}\n'
      'iRMSE={average.irmse:.3f}\n'
      'iMAE={average.imae:.3f}\n'
      't_GPU={average.gpu_time:.3f}\n'.format(
    average=avg80_sparse.average()))

textfile = open(os.path.join(test_dir,"test_result.txt"), "a")
textfile.write('\n**** EVALUATE WITH SPARSE DEPTH ****\n'
      '\n**** CAP=80 ****\n'
      'RMSE={average.rmse:.3f}\n'
      'RMSE_log={average.rmse_log:.3f}\n'
      'AbsRel={average.absrel:.3f}\n'
      'SqRel={average.squared_rel:.3f}\n'
      'SILog={average.silog:.3f}\n'
      'Delta1={average.delta1:.3f}\n'
      'Delta2={average.delta2:.3f}\n'
      'Delta3={average.delta3:.3f}\n'
      'iRMSE={average.irmse:.3f}\n'
      'iMAE={average.imae:.3f}\n'
      't_GPU={average.gpu_time:.3f}\n'.format(
    average=avg80_sparse.average()))
textfile.close()
