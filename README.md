# Depth Estimation from Monocular Images and Sparse Radar using Deep Ordinal Regression Network

Official implementation of "Depth Estimation from Monocular Images and Sparse Radar using Deep Ordinal Regression Network" (https://arxiv.org/abs/2107.07596), an accepted paper of ICIP2021.



## Dependency

Please check `Dockerfile` for environment settings and python packages


## Usage

### Generate interpolation dense lidar depthmap

Use `gen_interpolation.py` to generate sparse depth, dense depth and height-extended radar depth for training and evaluation. Generated depths will be saved in the same data_root directory of nuScenes sample, and please remember to modify the dir path in the .py. The number of CPU cores is set to 25 for faster data generating. 

pypardiso is highly recommended, otherwise one could have similar result via spsolve in scipy but with much slower speed.
Interpolation code is from https://gist.github.com/ialhashim/be6235489a9c43c6d240e8331836586a 

### Download pretrained resnet101 weights

Please download the pretrained ResNet101 weight file from http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth and modify the weight path in `model/resnet.py`.

### Train baseline model and proposed model on nuScenes

After modifying some paths in `train_nusc.py` and `trian_nusc_radar.py`, directly call the .py files for training the baseline model and the proposed model on nuScenes.

The code for evaluation and the pretrained weight file to meet the evaluation results in the paper will be updated soon.

The train_scene and val_scene txt files in ./list/nusc/ are the train/val splits I used in the paper. The reason why I used my own splits instead of official train/val splits was just because I didn't realize there are official ones at the time I conducted the experiments.


