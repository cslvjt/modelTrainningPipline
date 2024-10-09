import torch
import torch.nn as nn
import torch.nn.functional as F

def L1loss(pred, gt, weight = 1.0, reduction="mean"):
    assert reduction in ["mean", "sum", "none"]
    loss_func = nn.L1Loss(reduction=reduction)
    return weight * loss_func(pred,gt)

def FFTloss(pred, gt, weight = 1.0, reduction="mean"):
    assert reduction in ["mean", "sum", "none"]
    pred_fft = torch.fft.rfft2(pred, norm="backward")
    gt_fft = torch.fft.rfft2(gt, norm="backward")
    loss_func = nn.L1Loss(reduction=reduction)
    return weight * loss_func(pred_fft,gt_fft)