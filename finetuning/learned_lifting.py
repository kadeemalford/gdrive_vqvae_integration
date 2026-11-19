"""Local invertible lifting with explicit coefficient streams and scalar RD.

Odd trailing samples remain in the low band. Empty details are not generated.
Temporal lifting is fixed; only spatial predict/update residuals are learned.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class LocalResidual(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.first = nn.Conv2d(1, width, (3, 5), padding=(1, 2))
        self.last = nn.Conv2d(width, 1, (3, 5), padding=(1, 2))
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, x):
        return .25 * torch.tanh(self.last(F.silu(self.first(x))))


class LiftingCodec(nn.Module):
    VERSION = "spatial-lifting-scalar-v1"

    def __init__(self, learned=True, spatial_levels=5, temporal_levels=3, width=8):
        super().__init__()
        self.config = {"learned": learned, "spatial_levels": spatial_levels,
                       "temporal_levels": temporal_levels, "width": width}
        self.spatial_levels = spatial_levels
        self.temporal_levels = temporal_levels
        self.predict = nn.ModuleList([LocalResidual(width) for _ in range(spatial_levels)])
        self.update = nn.ModuleList([LocalResidual(width) for _ in range(spatial_levels)])
        if not learned:
            for parameter in list(self.predict.parameters()) + list(self.update.parameters()):
                parameter.requires_grad_(False)
        self.learned = learned
        bands = spatial_levels + temporal_levels + 1
        self.log_step = nn.Parameter(torch.full((bands,), math.log(4.)))
        self.log_scale = nn.Parameter(torch.zeros(bands))

    @property
    def steps(self):
        return self.log_step.clamp(-7, 7).exp()

    def schedule(self, shape):
        result = []
        h, w = shape
        for level in range(self.spatial_levels):
            if h < 2:
                break
            result.append((2, level, h))
            h = (h + 1) // 2
        for level in range(self.temporal_levels):
            if w < 2:
                break
            result.append((3, self.spatial_levels + level, w))
            w = (w + 1) // 2
        return result

    @staticmethod
    def extend(tensor, length, axis):
        if tensor.shape[axis] == length:
            return tensor
        # Zero detail on an unpaired endpoint: it is carried only by the low band.
        pad = (0, length - tensor.shape[3], 0, 0) if axis == 3 else (0, 0, 0, length - tensor.shape[2])
        return F.pad(tensor, pad)

    def analysis(self, tensor):
        low = tensor
        bands = []
        for axis, level, length in self.schedule(tensor.shape[-2:]):
            even = low.index_select(axis, torch.arange(0, length, 2, device=low.device))
            odd = low.index_select(axis, torch.arange(1, length, 2, device=low.device))
            n = odd.shape[axis]
            prediction = self.predict[level](even) if self.learned and axis == 2 else 0.
            detail = odd - (even + prediction).narrow(axis, 0, n)
            extended = self.extend(detail, even.shape[axis], axis)
            correction = self.update[level](extended) if self.learned and axis == 2 else 0.
            low = (even + .5 * extended + correction) * math.sqrt(2)
            bands.append(detail / math.sqrt(2))
        return bands + [low]

    def synthesis(self, bands, shape):
        schedule = self.schedule(shape)
        if len(bands) != len(schedule) + 1:
            raise ValueError("Coefficient band count does not match geometry")
        low = bands[-1]
        for detail, (axis, level, length) in reversed(list(zip(bands[:-1], schedule))):
            detail = detail * math.sqrt(2)
            scaled = low / math.sqrt(2)
            extended = self.extend(detail, scaled.shape[axis], axis)
            correction = self.update[level](extended) if self.learned and axis == 2 else 0.
            even = scaled - .5 * extended - correction
            prediction = self.predict[level](even) if self.learned and axis == 2 else 0.
            odd = detail + (even + prediction).narrow(axis, 0, detail.shape[axis])
            merged_shape = list(even.shape)
            merged_shape[axis] = length
            merged = torch.empty(merged_shape, device=low.device, dtype=low.dtype)
            if axis == 2:
                merged[:, :, ::2, :] = even
                merged[:, :, 1::2, :] = odd
            else:
                merged[:, :, :, ::2] = even
                merged[:, :, :, 1::2] = odd
            low = merged
        return low

    def band_indices(self, shape):
        return [level for _, level, _ in self.schedule(shape)] + [len(self.log_step) - 1]

    def forward(self, tensor, hard=False):
        bands = self.analysis(tensor)
        quantized, bits = [], tensor.new_zeros(())
        for band, index in zip(bands, self.band_indices(tensor.shape[-2:])):
            scaled = band / self.steps[index]
            if hard or not self.training:
                symbols = scaled.round()
            else:
                symbols = scaled + (scaled.round() - scaled).detach()
            quantized.append(symbols * self.steps[index])
            # Smooth factorized logistic mass, measured separately from real bytes.
            prior_scale = F.softplus(self.log_scale[index]) + .05
            upper, lower = (.5 - symbols) / prior_scale, (-.5 - symbols) / prior_scale
            log_mass = upper - F.softplus(upper) - F.softplus(lower) + torch.log(
                -torch.expm1(-1 / prior_scale))
            bits = bits - log_mass.sum() / math.log(2)
        return self.synthesis(quantized, tensor.shape[-2:]), bits / tensor.numel()
