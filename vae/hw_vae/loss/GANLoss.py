from typing import Optional
import torch
from torch import Tensor 
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.BCELoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 size_average=None, 
                 reduce=None, 
                 reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)