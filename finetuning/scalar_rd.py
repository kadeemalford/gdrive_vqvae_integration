"""Native residual transforms with scalar quantization and a factorized logistic prior."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from finetuning.latent_geometry import apply_geometry


class ScalarRDCodec(nn.Module):
    VERSION = 'native-scalar-factorized-v1'

    def __init__(self, base_channels=32, latent_channels=8, geometry='spatial-4x'):
        super().__init__()
        from vqvae.train_vqvae import Encoder, Decoder
        self.encoder = Encoder(1, base_channels, latent_channels, 4)
        self.decoder = Decoder(1, base_channels, latent_channels, 4)
        self.downsample_levels = 4
        self.log_scale = nn.Parameter(torch.zeros(latent_channels))
        self.config = {'architecture': self.VERSION, 'base_channels': base_channels,
                       'latent_channels': latent_channels, 'geometry': geometry}
        apply_geometry(self, geometry)

    @property
    def scales(self):
        return F.softplus(self.log_scale) + .05

    def rate_bits(self, values):
        scale = self.scales[None, :, None, None]
        upper = (.5 - values) / scale
        lower = (-.5 - values) / scale
        log_mass = upper - F.softplus(upper) - F.softplus(lower) + torch.log(-torch.expm1(-1 / scale))
        return -log_mass / math.log(2)

    def analysis(self, tensor):
        h, w = tensor.shape[-2:]
        return self.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))

    def synthesis(self, symbols, shape):
        return self.decoder(symbols)[..., :shape[0], :shape[1]]

    def forward(self, tensor):
        values = self.analysis(tensor)
        if self.training:
            quantized = values + torch.empty_like(values).uniform_(-.5, .5)
        else:
            quantized = torch.round(values)
        reconstruction = self.synthesis(quantized, tensor.shape[-2:])
        rate = self.rate_bits(quantized).sum() / tensor.numel()
        return reconstruction, rate, quantized
