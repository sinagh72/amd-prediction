import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        cce_loss = F.cross_entropy(input=y_pred, target=y_true)
        loss = self.alpha * (1 - torch.exp(-cce_loss)) ** self.gamma * cce_loss
        return loss
