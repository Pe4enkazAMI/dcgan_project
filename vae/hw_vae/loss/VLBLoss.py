from typing import Optional
import torch
from torch import Tensor 
import torch.nn as nn
import torch.nn.functional as F


class VLBLoss(nn.Module):
    def __init__(self, kld_weight=1) -> None:
        super().__init__()
        self.kld_weight = kld_weight

    
    def forward(self, decoded_sample, input, mu, log_sigma, *args, **kwargs):
        rec_loss = F.mse_loss(decoded_sample, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim = 1), dim = 0)
        loss = rec_loss + kld_loss * self.kld_weight
        return {
            "VLBLoss": loss,
            "ReconstructionLoss": rec_loss.detach(),
            "KLDLoss": -kld_loss.detach(),
        }