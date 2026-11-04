"""Frozen-decoder VQ training primitives and paired within-configuration sampling."""
from bisect import bisect_right
from collections import defaultdict
import random

import torch
import torch.nn.functional as F


class PairedTileSampler:
    """Identical configuration draws; only the within-configuration CDF changes."""
    def __init__(self, rows, policy, seed):
        if policy not in ('window', 'configuration-cell'):
            raise ValueError('Unknown tile sampling policy')
        self.rows = sorted(rows, key=lambda row: row['sample_path'])
        if not self.rows or any(row['split'] != 'train' for row in self.rows):
            raise ValueError('Nonempty train-only sampling rows required')
        self.policy = policy
        self.rng = random.Random(seed)
        self.groups = defaultdict(list)
        self.cdfs = {}
        for row in self.rows:
            self.groups[self.key(row)].append(row)
        for key, values in self.groups.items():
            cumulative = []
            total = 0
            for row in values:
                cells = int(row['channels']) * int(row['time_samples'])
                if cells < 1:
                    raise ValueError('Positive tile cell counts required')
                total += cells if policy == 'configuration-cell' else 1
                cumulative.append(total)
            self.cdfs[key] = cumulative

    @staticmethod
    def key(row):
        return row['family'], row['configuration']

    def draw(self):
        group = self.key(self.rows[self.rng.randrange(len(self.rows))])
        cumulative = self.cdfs[group]
        offset = self.rng.random() * cumulative[-1]
        index = min(bisect_right(cumulative, offset), len(cumulative) - 1)
        return self.groups[group][index]


def squared_distances(z, embedding):
    flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    return flat.square().sum(1, keepdim=True) - 2 * flat @ embedding.T + embedding.square().sum(1)


def hard_rate_with_soft_gradient(distances, costs, temperature, cells):
    if temperature <= 0 or cells < 1:
        raise ValueError('Positive temperature and retained-cell count required')
    indices = distances.argmin(1)
    soft = (torch.softmax(-distances / temperature, dim=1) * costs).sum() / cells
    hard = costs[indices].sum() / cells
    rate = hard.detach() + (soft - soft.detach())
    return rate, indices


def frozen_forward(model, tensor, costs, temperature, differentiable_rate=True):
    h, w = tensor.shape[-2:]
    multiple = 2 ** model.downsample_levels
    z = model.encoder(F.pad(tensor, (0, (-w) % multiple, 0, (-h) % multiple)))
    embedding = model.quantizer.embedding.weight
    distances = squared_distances(z, embedding)
    if differentiable_rate:
        rate, indices = hard_rate_with_soft_gradient(distances, costs, temperature, tensor.numel())
    else:
        indices = distances.argmin(1)
        rate = costs[indices].sum() / tensor.numel()
    exact = F.embedding(indices, embedding).view(z.shape[0], *z.shape[-2:], z.shape[1]).permute(0, 3, 1, 2).contiguous()
    # Preserve the historical commitment gradient; codebook loss is constant when frozen.
    vq = F.mse_loss(exact, z.detach()) + float(getattr(model.quantizer, 'commitment_cost', .25)) * F.mse_loss(z, exact.detach())
    quantized = z + (exact - z).detach() if model.encoder.training else exact
    predicted = model.decoder(quantized)[..., :h, :w]
    return predicted, vq, rate, indices, tuple(z.shape[-2:]), distances


def configure_encoder_only(model):
    model.eval()
    parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('encoder.'))
        if parameter.requires_grad:
            parameters.append(parameter)
    if not parameters:
        raise ValueError('No trainable encoder parameters')
    return parameters
