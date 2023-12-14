import torch
from torch import nn 
import torch.nn.functional as F


class VDecoder(nn.Module):
    def __init__(self, hidden_dims) -> None:
        super().__init__()
        
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i], 
                                       out_channels=hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )
        self.decoder = nn.Sequential(*modules)
    def forward(self, x):
        return self.decoder(x)