# Depth Estimation from Monocular Images and Sparse Radar using Deep Ordinal Regression Network

Official implementation of "Depth Estimation from Monocular Images and Sparse Radar using Deep Ordinal Regression Network", an accepted paper of ICIP2021.



## Dependency

Please check `Dockerfile` for environment settings and python packages


## Usage

### Generate interpolation dense lidar depthmap

Use `gen_interpolation.py` to generate sparse depth, dense depth and height-extended radar depth for training and evaluation. Generated depths will be saved in the same data_root directory of nuScenes sample, and please remember to modify the dir path in the .py. The number of CPU cores is set to 25 for faster data generating. 

pypardiso is highly recommended, otherwise one could have similar result via spsolve in scipy but with much slower speed.
Interpolation code is from https://gist.github.com/ialhashim/be6235489a9c43c6d240e8331836586a 

### Download pretrained resnet101 weights

Please download the pretrained ResNet101 weight file and modify the weight path in `model/resnet.py`.

### Train baseline model and proposed model on nuScenes

After modifying some paths in `train_nusc.py` and `trian_nusc_radar.py`, directly call the .py files for training the baseline model and the proposed model on nuScenes.

The code for evaluation and the pretrained weight file to meet the evaluation results in the paper will be updated soon.





