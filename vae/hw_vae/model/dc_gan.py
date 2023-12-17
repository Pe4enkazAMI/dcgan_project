import torch.nn as nn
from .discriminator import Discriminator
from.generator import Generator

class DCGAN(nn.Module):
    def __init__(self, 
                 in_channels=128,
                 out_channels=64, 
                 pic_channels=3, 
                 disc_out_dim=64, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = Generator(in_channels, out_channels, pic_channels)
        self.discriminator = Discriminator(pic_channels, disc_out_dim)

    def forward(self, x):
        return x
    
    def generate(self, input):
        return self.generator(input)
    
    def discriminate(self, input):
        return self.discriminator(input)