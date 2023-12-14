import torch
from torch import nn 
import torch.nn.functional as F
from .encoder import VEncoder
from .decoder import VDecoder
import numpy as np

class vVAE(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 in_channels=3,
                 hidden_dim=None, 
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        if self.hidden_dim is None:
            self.hidden_dim = [32, 64, 128, 256, 512]

        self.encoder = VEncoder(in_channels=in_channels, hidden_dims=self.hidden_dim)

        self.fc_mu_head = nn.Linear(self.hidden_dim[-1] * 4, self.latent_dim)
        self.fc_sigma_head = nn.Linear(self.hidden_dim[-1] * 4, self.latent_dim)

        self.pre_decoder_layer = nn.Linear(latent_dim, self.hidden_dim[-1]*4)

        self.decoder = VDecoder(hidden_dims=self.hidden_dim)

        self.out_proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.hidden_dim[-1],
                               out_channels=self.hidden_dim[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.hidden_dim[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1),
            nn.Tanh() # <---- pixels should be in [-1, 1]
        )        
    
    def encode(self, x):
        lat_x = self.encoder(x)
        lat_x = torch.flatten(lat_x, start_dim=1)
        mu = self.fc_mu_head(lat_x)
        log_sigma = self.fc_sigma_head(lat_x)
        return mu, log_sigma
    
    def decode(self, z):
        out = self.pre_decoder_layer(z)
        out = out.view(-1, 512, 2, 2)
        out = self.decoder(out)
        out = self.out_proj(out)
        return out
    
    def sample_latent(self, mu, log_sigma):
        eps = torch.rand_like(mu)
        sigma = torch.exp(0.5 * log_sigma)

        return mu + sigma * eps

    def forward(self, image):
        mu, log_sigma = self.encode(image)
        z = self.sample_latent(mu, log_sigma)
        return {
            "decoded_sample": self.decode(z),
            "input": image,
            "mu": mu,
            "log_sigma": log_sigma
        }
    
    def generate(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    


