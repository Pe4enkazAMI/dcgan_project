from typing import Optional
import torch
from torch import Tensor 
import torch.nn as nn
import torch.nn.functional as F


class VLBLoss(nn.Module):
    def __init__(self, kld_weight=2) -> None:
        super().__init__()
        self.kld_weight = kld_weight

    
    def forward(self, decoded_sample, input, mu, log_sigma, *args, **kwargs):
        rec_loss = 0.5 * F.mse_loss(decoded_sample, input)
        std = torch.sqrt(torch.exp(log_sigma))

        kld_loss = 0.5 * torch.sum(std.pow(2) + mu.pow(2) - 1. - torch.log(std.pow(2)))
        loss = rec_loss + kld_loss * self.kld_weight
        return {
            "VLBLoss": loss,
            "ReconstructionLoss": rec_loss.detach(),
            "KLDLoss": -kld_loss.detach(),
        }