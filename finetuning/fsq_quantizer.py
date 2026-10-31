"""Finite scalar quantization with 8x8x32 states and an exact discrete decoder.

This scientific reconstruction variant uses scalar rounding and straight-through
gradients, not a learned VQ codebook or an adversarial/perceptual objective.
"""
import torch
from torch import nn


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.num_embeddings = 2048
        self.embedding_dim = embedding_dim
        self.project_in = nn.Linear(embedding_dim, 3)
        self.project_out = nn.Linear(3, embedding_dim)
        self.register_buffer('levels_minus_one', torch.tensor([7., 7., 31.]))

    def embedding(self, indices):
        """Decode the 11-bit mixed-radix state without encoder features."""
        indices = indices.long()
        digits = torch.stack((indices % 8, (indices // 8) % 8, indices // 64), dim=-1)
        coordinates = 2 * digits.to(self.project_out.weight.dtype) / self.levels_minus_one - 1
        return self.project_out(coordinates)

    def forward(self, z):
        batch, channels, height, width = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, channels)
        bounded = torch.sigmoid(self.project_in(flat)) * self.levels_minus_one
        rounded = bounded.round()
        digits = rounded.long()
        indices = digits[:, 0] + 8*digits[:, 1] + 64*digits[:, 2]
        if self.training:
            quantized = bounded + (rounded-bounded).detach()
            vectors = self.project_out(2*quantized/self.levels_minus_one-1)
        else:
            vectors = self.embedding(indices)
        vectors = vectors.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        probabilities = torch.bincount(indices, minlength=2048).to(z.dtype) / indices.numel()
        perplexity = torch.exp(-(probabilities * (probabilities+1e-10).log()).sum())
        return vectors, z.sum()*0., indices, perplexity
