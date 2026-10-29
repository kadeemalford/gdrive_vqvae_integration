"""FEBUS-oriented, factorized multiscale VQ-VAE without uncoded decoder skips."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MultiscaleAxisBlock(nn.Module):
    """Mix local space and two temporal scales before pointwise feature fusion."""
    def __init__(self, channels, dilation=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.temporal_local = nn.Conv2d(channels, channels, (1, 9), padding=(0, 4), groups=channels)
        self.temporal_context = nn.Conv2d(channels, channels, (1, 9), padding=(0, 4*dilation),
                                          dilation=(1, dilation), groups=channels)
        self.spatial_local = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0), groups=channels)
        self.fusion = nn.Sequential(nn.Conv2d(3*channels, 4*channels, 1), nn.SiLU(),
                                    nn.Conv2d(4*channels, channels, 1))

    def forward(self, x):
        if self.training and x.requires_grad:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        z = F.silu(self.norm(x))
        features = torch.cat((self.temporal_local(z), self.temporal_context(z), self.spatial_local(z)), dim=1)
        return x+self.fusion(features)


class AxisDown(nn.Module):
    def __init__(self, channels, next_channels, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.temporal = nn.Conv2d(channels, next_channels, (1, 4), stride=(1, 2), padding=(0, 1))
        self.spatial = nn.Conv2d(next_channels, next_channels, (4, 1), stride=(2, 1), padding=(1, 0))
        self.block = MultiscaleAxisBlock(next_channels, dilation)

    def forward(self, x):
        return self.block(self.spatial(self.temporal(F.silu(self.norm(x)))))


class AxisUp(nn.Module):
    def __init__(self, channels, next_channels, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.spatial = nn.ConvTranspose2d(channels, channels, (4, 1), stride=(2, 1), padding=(1, 0))
        self.temporal = nn.ConvTranspose2d(channels, next_channels, (1, 4), stride=(1, 2), padding=(0, 1))
        self.block = MultiscaleAxisBlock(next_channels, dilation)

    def forward(self, x):
        return self.block(self.temporal(self.spatial(F.silu(self.norm(x)))))


class AxisEncoder(nn.Module):
    def __init__(self, base, embedding_dim, levels):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, base, (1, 9), padding=(0, 4)), nn.SiLU(),
                                  nn.Conv2d(base, base, (3, 1), padding=(1, 0)))
        channels = base
        self.stages = nn.ModuleList()
        for index in range(levels):
            next_channels = min(channels*2, base*8)
            self.stages.append(AxisDown(channels, next_channels, 2**(index % 3)))
            channels = next_channels
        self.context = MultiscaleAxisBlock(channels, 8)
        self.projection = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, embedding_dim, 1))

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.projection(self.context(x))


class AxisDecoder(nn.Module):
    def __init__(self, base, embedding_dim, levels):
        super().__init__()
        channels = min(base*2**levels, base*8)
        self.projection = nn.Sequential(nn.Conv2d(embedding_dim, channels, 1), MultiscaleAxisBlock(channels, 8))
        self.stages = nn.ModuleList()
        for index in range(levels):
            next_channels = max(channels//2, base)
            self.stages.append(AxisUp(channels, next_channels, 2**((levels-index-1) % 3)))
            channels = next_channels
        self.output = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, x):
        x = self.projection(x)
        for stage in self.stages:
            x = stage(x)
        return self.output(x)


class AnisotropicVQVAE(nn.Module):
    """Native-compatible index geometry; independent reduction/reconstruction axes."""
    def __init__(self, base_channels, embedding_dim, num_embeddings, downsample_levels, commitment_cost):
        super().__init__()
        from vqvae.train_vqvae import VectorQuantizer
        self.downsample_levels = downsample_levels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.encoder = AxisEncoder(base_channels, embedding_dim, downsample_levels)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = AxisDecoder(base_channels, embedding_dim, downsample_levels)

    def configure_latent_geometry(self, spatial_changes, temporal_changes):
        levels = self.downsample_levels
        for i, stage in enumerate(self.encoder.stages):
            stage.spatial.stride = (1 if i >= levels-spatial_changes else 2, 1)
            stage.temporal.stride = (1, 1 if i >= levels-temporal_changes else 2)
        for i, stage in enumerate(self.decoder.stages):
            stage.spatial.stride = (1 if i < spatial_changes else 2, 1)
            stage.temporal.stride = (1, 1 if i < temporal_changes else 2)

    def forward(self, x):
        height, width = x.shape[-2:]
        multiple = 2**self.downsample_levels
        x = F.pad(x, (0, (-width) % multiple, 0, (-height) % multiple))
        z = self.encoder(x)
        quantized, loss, indices, perplexity = self.quantizer(z)
        return self.decoder(quantized)[..., :height, :width], loss, indices, perplexity
