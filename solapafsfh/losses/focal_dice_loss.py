import torch
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )
        self.focal_loss = FocalLoss(gamma=gamma)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.alpha * dice + (1 - self.alpha) * focal
