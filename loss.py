import numpy as np
import torch
import torch.nn.functional as F

class OrdinalRegressionLoss(torch.nn.Module):

    def __init__(self, ord_num, beta, discretization="SID"):
        super(OrdinalRegressionLoss, self).__init__()
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, depth):
        N, _, H, W = depth.shape

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(depth.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(depth) / np.log(self.beta)
        else:
            label = self.ord_num * (depth - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(depth.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask < label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        return ord_c0, ord_c1

    def forward(self, prob, depth):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        N, C, H, W = prob.shape
        valid_mask = depth > 0.
        ord_c0, ord_c1 = self._create_ord_label(depth)
        logP = torch.log(torch.clamp(prob, min=1e-8))
        log1_P = torch.log(torch.clamp(1 - prob, min=1e-8))
        entropy = torch.sum(ord_c1*logP, dim=1) + torch.sum(ord_c0*log1_P, dim=1) # eq. (2)
        
        valid_mask = torch.squeeze(valid_mask, 1)
        loss = - entropy[valid_mask].mean()
        return loss