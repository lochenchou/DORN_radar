import os
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from scipy import interpolate
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from scipy.interpolate import LinearNDInterpolator
import cv2 as cv2

        

def lin_interp(shape, points, depth):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij = points
    f = LinearNDInterpolator(ij, depth, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    interp_depth = f(IJ).reshape(shape)
    return interp_depth.T


def getSensorToken(nusc, camera_token, sensor_channel):
    # return sensor_token in the same sample of given camera_token
    data = nusc.get('sample_data', camera_token)
    sample = nusc.get('sample', data['sample_token'])
    sensor_token = sample['data'][sensor_channel]
    return sensor_token


def getSceneToken(nusc, scene_name):
    for scene in nusc.scene:
        if scene['name'] == scene_name:
            return scene['token']


def getFileName(nusc, camera_token):
    return nusc.get('sample_data', camera_token)['filename']


def getDescription(nusc, camera_token):
    data = nusc.get('sample_data', camera_token)
    sample = nusc.get('sample', data['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    return scene['description'].lower()


def getData(nusc, token):
    data = nusc.get('sample_data', token)
    return data


def getIndexValue(array):
    i,j = np.where(array>0)
    v = array[i,j]
    
    return i,j,v


def get_gray_image(nusc, camera_token):
    cam = nusc.get('sample_data', camera_token)
    im = Image.open(os.path.join(nusc.dataroot, cam['filename']))
    im  = np.asarray(im)
    Ig = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return Ig


def show_image(img_array, isGray=False, figsize=(15,10)):
    plt.figure(figsize=figsize)
    if isGray:
        plt.imshow(img_array, cmap="gray")
    else:
        plt.imshow(img_array)
    plt.axis('off');
    

def velo2depth(nusc, pc, camera_token, pointsensor_token, min_dist = 1.0):
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    
#     _pc = LidarPointCloud(pc.points)

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    im = Image.open(os.path.join(nusc.dataroot, cam['filename']))
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    depthmap = np.zeros((im.size[1],im.size[0]))
    depthmap[points[1,:].astype(np.int), points[0,:].astype(np.int)] = coloring
    
    return depthmap


def depth2velo(nusc, depthmap, camera_token, pointsensor_token):
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    
    # form depthmap into points
    rows, cols = depthmap.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depthmap])
    points = points.reshape((3, -1))
    points = points[:,points[2,:]>0]
    
    # get camera_intrinsic value
    camera_intrinsic = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])['camera_intrinsic']
    f_u = camera_intrinsic[0][0]
    f_v = camera_intrinsic[1][1]
    c_u = camera_intrinsic[0][2]
    c_v = camera_intrinsic[1][2]
    
    # according to pseudo-lidar paper
    uv_depth = points.T
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u 
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    
    _pc = LidarPointCloud(np.vstack([pts_3d_rect.T,np.zeros((1,pts_3d_rect.T.shape[1]))]))
#     print(_pc.points.shape)
    # inverse steps from velo2depth...
    # Fourth step: transform from camera into the ego.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    _pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    _pc.translate(np.array(cs_record['translation']))

    # Third step: transform from the ego vehicle frame for the timestamp of the image into global.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    _pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    _pc.translate(np.array(poserecord['translation']))

    # Second step: transform from the global frame to ego_sensor.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    _pc.translate(-np.array(poserecord['translation']))
    _pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # First step: transform the ego vehicle frame for the timestamp of the sweep to the pointcloud.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    _pc.translate(-np.array(cs_record['translation']))
    _pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    return _pc


def read_multisweep(nusc,
                    pointsensor_token: str,
                    nsweeps: int = 5):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>). 
    
    !!! return in lidar_top coordinate (ref_chan)
    """
    
    sd_record = nusc.get('sample_data', pointsensor_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    
    ref_chan = 'LIDAR_TOP'
    ref_record = nusc.get('sample_data', sample_rec['data'][ref_chan])
    
    sensor_modality = sd_record['sensor_modality']
    
    if sensor_modality == 'lidar':
        # Get aggregated lidar point cloud in lidar frame.
        pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
    else:
        # Get aggregated radar point cloud in reference frame.
        # The point cloud is transformed to the reference frame for visualization purposes.
        pc, times = RadarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)
    
    return pc.points[:3, :], times


def read_radar(nusc, sample_token, nsweeps=5):
    sample = nusc.get('sample', sample_token)
    pc, times = read_multisweep(nusc, sample["data"]["RADAR_FRONT"], nsweeps=nsweeps)
    for radar_chan in ['RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']:
        _pc, _times = read_multisweep(nusc, sample["data"][radar_chan])

        pc = np.concatenate((pc, _pc), axis=1)
        times = np.concatenate((times, _times), axis=1)
        
    ex_pc = 255 * np.ones((4, pc.shape[1]))
    ex_pc[:3,:] = pc
           
    return ex_pc.T.astype('float32'), times.T.astype('float32')

