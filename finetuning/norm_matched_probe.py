"""Single deterministic latent-energy control, not output gain calibration."""
import math

import torch

try:
    from .quantization_probe import QuantizationProbe
except ImportError:
    from quantization_probe import QuantizationProbe


class NormMatchedProbe(QuantizationProbe):
    def __init__(self):
        super().__init__(continuous=True)

    def __call__(self, module, inputs, output):
        latent, loss, indices, perplexity = super().__call__(module, inputs, output)
        source = self.latest["encoder_latent_rms"]
        target = self.latest["quantized_latent_rms"]
        if source == 0:
            if target != 0:
                raise ValueError("Cannot match nonzero quantized energy from zero latent")
            scale = 1.
        else:
            scale = target / source
        if not math.isfinite(scale):
            raise ValueError("Invalid latent norm scale")
        transmitted = latent * scale
        if not torch.isfinite(transmitted).all():
            raise ValueError("Nonfinite norm-matched latents")
        self.latest.update(norm_matching_scale=scale,
                           transmitted_latent_rms=float(transmitted.double().square().mean().sqrt().item()))
        return transmitted, loss, indices, perplexity
