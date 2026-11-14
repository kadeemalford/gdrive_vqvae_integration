"""Experimental decoder-only lookup: fixed analysis IDs, independent synthesis vectors."""
import torch
import torch.nn.functional as F


class SynthesisLookup(torch.nn.Module):
    def __init__(self, analysis_weight):
        super().__init__()
        if (analysis_weight.ndim != 2 or not analysis_weight.is_floating_point()
                or min(analysis_weight.shape) < 1 or not torch.isfinite(analysis_weight).all()):
            raise ValueError('Finite two-dimensional analysis codebook required')
        # Never alias the encoder's assignment dictionary or its gradient storage.
        self.weight = torch.nn.Parameter(analysis_weight.detach().clone())

    def forward(self, indices, latent_shape):
        if len(latent_shape) != 2 or any(int(v) != v or v < 1 for v in latent_shape):
            raise ValueError('Positive two-dimensional latent grid required')
        h, w = map(int, latent_shape)
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint16)
        if indices.numel() != h * w or indices.dtype not in integer_types:
            raise ValueError('One integer code per latent position required')
        indices = indices.reshape(-1).long()
        if indices.device != self.weight.device or indices.min() < 0 or indices.max() >= len(self.weight):
            raise ValueError('Invalid code IDs or device mismatch')
        return F.embedding(indices, self.weight).view(1, h, w, self.weight.shape[1]).permute(0, 3, 1, 2).contiguous()
