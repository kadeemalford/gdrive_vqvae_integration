"""Frozen VQ reference and off-manifold continuous-latent diagnostic hooks."""
import math

import torch


class QuantizationProbe:
    def __init__(self, continuous=False):
        self.continuous = continuous
        self.latest = None

    def __call__(self, module, inputs, output):
        latent = inputs[0]
        quantized, vq_loss, indices, perplexity = output
        if (latent.dtype != torch.float32 or latent.shape != quantized.shape
                or not torch.isfinite(latent).all() or not torch.isfinite(quantized).all()):
            raise ValueError("Invalid latent tensors")
        positions = latent.shape[0] * latent.shape[2] * latent.shape[3]
        if indices.numel() != positions:
            raise ValueError("Latent/index count mismatch")
        a, b = latent.double(), quantized.double()
        energy = float(a.square().sum().item())
        error = float((a - b).square().sum().item())
        self.latest = {"relative_quantization_error": math.sqrt(error / max(energy, 1e-24)),
                       "encoder_latent_rms": math.sqrt(energy / a.numel()),
                       "quantized_latent_rms": math.sqrt(float(b.square().mean().item())),
                       "reference_vq_loss": float(vq_loss.item()),
                       "latent_positions": positions, "latent_dimensions": int(latent.shape[1]),
                       "reference_used_codes": int(indices.unique().numel())}
        if self.continuous:
            # Reference indices remain for geometry checking only, not transmitted payload.
            return latent, torch.zeros_like(vq_loss), indices, perplexity
        return output


def payload_bits(positions, dimensions, codebook_size, continuous):
    if min(positions, dimensions) < 1 or codebook_size < 2:
        raise ValueError("Invalid latent/rate dimensions")
    return positions * (dimensions * 32 if continuous else math.ceil(math.log2(codebook_size)))
