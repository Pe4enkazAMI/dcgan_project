import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        modules = []
        exp_factor = 1
        for i in range(num_layers):
            modules.append(
                nn.Conv2d(in_channels=out_channels * exp_factor,
                          out_channels=out_channels * exp_factor * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                          )
            )
            modules.append(
                nn.BatchNorm2d(out_channels * exp_factor * 2)
            )
            modules.append(
                nn.LeakyReLU(0.2, inplace=True)
            )
            exp_factor *= 2

        self.body = nn.Sequential(*modules)
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * (2 ** self.num_layers), 
                      out_channels=1, 
                      kernel_size=4, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.body(x)
        x = self.out_proj(x)
        return x