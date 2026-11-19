"""Zero-initialized local-statistics residual, preserving native GN at origin."""
import torch
from torch import nn

from finetuning.model_factory import new_model

VERSION = "local-internal-normalization-v1"


class LocalResidualGroupNorm(nn.GroupNorm):
    def __init__(self, original, enabled):
        if not original.affine:
            raise ValueError("Expected native affine GroupNorm")
        super().__init__(original.num_groups, original.num_channels, original.eps, original.affine,
                         device=original.weight.device, dtype=original.weight.dtype)
        self.load_state_dict(original.state_dict())
        self.gate = nn.Parameter(torch.zeros((), device=self.weight.device), requires_grad=enabled)
        self.enabled = enabled

    def local_values(self, x):
        batch, channels, height, width = x.shape
        grouped = x.reshape(batch, self.num_groups, channels // self.num_groups, height, width)
        mean = grouped.mean(2, keepdim=True)
        variance = grouped.var(2, keepdim=True, unbiased=False)
        local = ((grouped - mean) * torch.rsqrt(variance + self.eps)).reshape_as(x)
        if self.affine:
            local = local * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return local

    def forward(self, x):
        global_values = super().forward(x)
        if not self.enabled:
            return global_values
        local = self.local_values(x)
        return global_values + .5 * torch.tanh(self.gate) * (local - global_values)


def replace_norms(module, enabled):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, LocalResidualGroupNorm(child, enabled))
            count += 1
        else:
            count += replace_norms(child, enabled)
    return count


def initialize(source, enabled, device):
    config = {**source["config"], "exact_eval_quantization": True, "activation_checkpointing": True}
    if config.get("latent_geometry") != "spatial4-temporal4":
        raise ValueError("Local-statistics contrast expects the registered dense source")
    model = new_model(config, device)
    model.load_state_dict(source["model_state_dict"])
    count = replace_norms(model, enabled)
    if not count:
        raise ValueError("No native GroupNorm operators to contrast")
    return model, {"architecture": VERSION, "base_config": config, "local_residual_enabled": enabled,
                   "norm_layers": count}


def build(config, state, device):
    if config.get("architecture") != VERSION:
        raise ValueError("Unknown local-statistics checkpoint schema")
    model = new_model(config["base_config"], device)
    if replace_norms(model, config["local_residual_enabled"]) != config["norm_layers"]:
        raise ValueError("Local normalization topology differs from serialized schema")
    model.load_state_dict(state, strict=True)
    return model.eval()
