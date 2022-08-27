import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

def ce_loss(pred, target):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(np.array(pred), dtype=torch.float32)

    if not isinstance(target, torch.Tensor):
        target = torch.tensor(np.array(target), dtype=torch.int64)

    CEloss = CrossEntropyLoss()
    return CEloss(pred, target)

def mse_loss(pred, target):
    return (pred - target).pow(2).mean()

def binary_ce_loss(pred, target):
    pred_squeeze = pred[:, 1].squeeze()
    return F.binary_cross_entropy(pred_squeeze, target.to(torch.float))
