import torch
from torch import nn 
import torch.nn.functional as F


class VEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims) -> None:
        super().__init__()
        self.hidden_dims  = hidden_dims

        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)