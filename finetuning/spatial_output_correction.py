"""Identity-initialized correction using decoder outputs, never original neighbors."""
import torch
from torch import nn
import torch.nn.functional as F


class SpatialOutputCorrection(nn.Module):
    def __init__(self, spatial=True, hidden=16):
        super().__init__()
        self.spatial = spatial
        self.hidden = hidden
        self.net = nn.Sequential(nn.Conv2d(7, hidden, 1), nn.SiLU(),
                                 nn.Conv2d(hidden, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, decoded):
        if decoded.ndim != 4 or decoded.shape[1] != 1:
            raise ValueError('Expected batch x 1 x channels x samples')
        h = decoded.shape[2]
        if self.spatial:
            padded = F.pad(decoded, (0, 0, 2, 2), mode='replicate')
            neighbors = [padded[:, :, offset:offset + h] for offset in range(5)]
        else:
            neighbors = [decoded] * 5
        position = torch.arange(h, device=decoded.device, dtype=decoded.dtype)
        left = (position / 8).clamp(max=1).view(1, 1, h, 1).expand_as(decoded)
        right = ((h - 1 - position) / 8).clamp(max=1).view(1, 1, h, 1).expand_as(decoded)
        return torch.cat([*neighbors, left, right], dim=1)

    def forward(self, decoded):
        return decoded + self.net(self.features(decoded))
