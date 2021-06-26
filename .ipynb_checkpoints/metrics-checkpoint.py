import torch
import math
import numpy as np

lg_e_10 = math.log(10)

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / lg_e_10

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    valid_mask = gt>0
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    
    valid_mask = pred>0
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    
    thresh = torch.max((gt / pred), (pred / gt))
    d1 = float((thresh < 1.25).float().mean())
    d2 = float((thresh < 1.25 ** 2).float().mean())
    d3 = float((thresh < 1.25 ** 3).float().mean())
        
    rmse = (gt - pred) ** 2
    rmse = math.sqrt(rmse.mean())
    
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = math.sqrt(rmse_log.mean())
    
    abs_rel = ((gt - pred).abs() / gt).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean()

    return abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

class Result(object):
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.rmse_log = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0  # Scale invariant logarithmic error [log(m)*100]
        self.photometric = 0

    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.lg10 = np.inf
        self.rmse_log = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, lg10, rmse_log, \
            delta1, delta2, delta3, gpu_time, data_time, silog, photometric=0):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.rmse_log = rmse_log
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.data_time = data_time
        self.gpu_time = gpu_time
        self.silog = silog
        self.photometric = photometric

    def evaluate(self, output, target, photometric=0, cap=None):
    
        if cap != None:
            output = torch.clamp(output, max=cap)
            target = torch.clamp(target, max=cap)
            
        valid_mask = output>0.1
        output = output[valid_mask]
        target = target[valid_mask]

        valid_mask = target > 0.1
        output = output[valid_mask]
        target = target[valid_mask]

        # convert from meters to mm
#         output_mm = 1e3 * output
#         target_mm = 1e3 * target
        output_mm = output
        target_mm = target

        abs_diff = (output_mm - target_mm).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float(((abs_diff**2 / target_mm)).mean())

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25**2).float().mean())
        self.delta3 = float((maxRatio < 1.25**3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        # silog uses meters
        err_log = torch.log(target) - torch.log(output)
        self.rmse_log =  math.sqrt((err_log**2).mean())       
        self.silog = math.sqrt((err_log ** 2).mean() - (err_log.mean() ** 2)) * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output)**(-1)
        inv_target_km = (1e-3 * target)**(-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        self.photometric = float(photometric)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_rmse_log = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_photometric = 0
        self.sum_silog = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_rmse_log += n * result.rmse_log
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time
        self.sum_silog += n * result.silog
        self.sum_photometric += n * result.photometric

    def average(self):
        avg = Result()
        if self.count > 0:
            avg.update(
                self.sum_irmse / self.count, self.sum_imae / self.count,
                self.sum_mse / self.count, self.sum_rmse / self.count,
                self.sum_mae / self.count, self.sum_absrel / self.count,
                self.sum_squared_rel / self.count, self.sum_lg10 / self.count, self.sum_rmse_log / self.count,
                self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count, 
                self.sum_gpu_time / self.count,
                self.sum_data_time / self.count, self.sum_silog / self.count,
                self.sum_photometric / self.count)
        return avg
