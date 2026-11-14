"""Identity-initialized synthesis residuals using only decoded tokens and known shape."""
import math

import torch


KINDS = ('local', 'shape', 'moments', 'axial')


def known_shape_features(q, original_shape):
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 128 or len(original_shape) != 2:
        raise ValueError('One 128-feature compact latent frame and original shape required')
    h, w = map(int, original_shape)
    lh, lw = q.shape[-2:]
    # Inverse spatial-4x decoder canvas, including its two stride-1 transpose layers.
    canvas = ((lh + 2) * 4, lw * 16)
    if min(h, w) < 1 or h > canvas[0] or w > canvas[1]:
        raise ValueError('Original shape exceeds compact decoder canvas')
    rows = torch.linspace(-1, 1, lh, device=q.device, dtype=q.dtype)
    columns = torch.linspace(-1, 1, lw, device=q.device, dtype=q.dtype)
    yy, xx = torch.meshgrid(rows, columns, indexing='ij')
    frequencies = (2. ** torch.arange(8, device=q.device, dtype=q.dtype)) * math.pi
    position = torch.cat([torch.sin(yy[None] * frequencies[:, None, None]),
                          torch.cos(yy[None] * frequencies[:, None, None]),
                          torch.sin(xx[None] * frequencies[:, None, None]),
                          torch.cos(xx[None] * frequencies[:, None, None])], dim=0)[None] * .1
    fractions = q.new_tensor([h / canvas[0], w / canvas[1]])[None, :, None, None].expand(1, 2, lh, lw)
    return position, fractions


class AxisBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(32)
        self.attention = torch.nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        self.norm2 = torch.nn.LayerNorm(32)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.SiLU(), torch.nn.Linear(128, 32))

    def forward(self, sequence):
        normalized = self.norm1(sequence)
        sequence = sequence + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return sequence + self.ffn(self.norm2(sequence))


class LatentContextAdapter(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError('Unknown latent context control')
        self.kind = kind
        if kind == 'axial':
            self.input = torch.nn.Conv2d(130, 32, 1)
            self.rows, self.columns = AxisBlock(), AxisBlock()
            self.output = torch.nn.Conv2d(32, 128, 1)
        else:
            inputs, hidden = {'local': (128, 132), 'shape': (162, 116), 'moments': (418, 62)}[kind]
            self.input = torch.nn.Sequential(torch.nn.Conv2d(inputs, hidden, 1), torch.nn.SiLU())
            self.output = torch.nn.Conv2d(hidden, 128, 1)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, q, original_shape):
        position, fractions = known_shape_features(q, original_shape)
        if self.kind == 'axial':
            x = self.input(torch.cat([q, fractions], dim=1)) + position
            b, c, h, w = x.shape
            sequence = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
            x = self.rows(sequence).reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()
            sequence = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
            x = self.columns(sequence).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        else:
            features = [q]
            if self.kind != 'local':
                features += [position, fractions]
            if self.kind == 'moments':
                mean = q.mean(dim=(-2, -1), keepdim=True)
                std = torch.sqrt((q - mean).square().mean(dim=(-2, -1), keepdim=True) + 1e-8)
                features += [mean.expand_as(q), std.expand_as(q)]
            x = self.input(torch.cat(features, dim=1))
        return q + self.output(x)
