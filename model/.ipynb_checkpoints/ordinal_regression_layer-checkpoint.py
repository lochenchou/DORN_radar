import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :input x: shape = (N,C,H,W), C = 2*ord_num (2*K)
        :return: ord prob is the label probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size() # (N, 2K, H, W)
        ord_num = C // 2
        
        label_0 = x[:, 0::2, :, :].clone().view(N, 1, ord_num, H, W)  # (N, 1, K, H, W)
        label_1 = x[:, 1::2, :, :].clone().view(N, 1, ord_num, H, W)  # (N, 1, K, H, W)

        label = torch.cat((label_0, label_1), dim=1) # (N, 2, K, H, W)
        label = torch.clamp(label, min=1e-8, max=1e8)  # prevent nans

        label_ord = torch.nn.functional.softmax(label, dim=1)
        prob = label_ord[:,1,:,:,:].clone() # label_ord is the output softmax probability of this model
        return prob
