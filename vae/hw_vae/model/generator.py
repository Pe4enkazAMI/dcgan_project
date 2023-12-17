import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, pic_channels, num_layers=4):
        super().__init__()
        self.nz = in_channels
        self.ngf = out_channels
        self.nc = pic_channels

        modules = []
        exp_factor = 2 ** (num_layers - 1)
        for i in range(num_layers):
            modules.append(nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels= out_channels * exp_factor,
                kernel_size=4,
                stride=1 if i == 0 else 2,
                padding=0 if i == 0 else 1,
                bias=False
            ))
            modules.append(
                nn.BatchNorm2d(out_channels * exp_factor)
            )
            modules.append(nn.ReLU(inplace=True))
            in_channels = exp_factor * out_channels
            exp_factor = exp_factor // 2
        self.body = nn.Sequential(*modules)

        self.out_proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels, 
                               out_channels=pic_channels, 
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            #nn.Tanh() # pixels should be in [-1, 1]
        )

    def forward(self, input):
        input = self.body(input)
        input = self.out_proj(input)
        return input