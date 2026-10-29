"""Versioned rate-allocation geometries; weight shapes do not encode strides/padding."""
import math

import torch
import torch.nn.functional as F

POLICIES = ("standard", "temporal-2x", "temporal-4x", "balanced-2x", "spatial-4x", "spatial4-temporal2", "spatial4-temporal4")
CHANGES = {"standard": (0, 0), "temporal-2x": (0, 1), "temporal-4x": (0, 2),
           "balanced-2x": (1, 1), "spatial-4x": (2, 0),
           "spatial4-temporal2": (2, 1), "spatial4-temporal4": (2, 2)}


def spatial_padding(policy, levels):
    spatial, temporal = CHANGES[policy]
    if levels < max(spatial, temporal):
        raise ValueError("Unsupported geometry/downsample levels")
    # Each stride-1 kernel-4 loses one cell. Compensating bottom padding
    # preserves spatial grid density without removing any original channel.
    return spatial * 2 ** (levels - spatial) if spatial else 0


def checkpoint_geometry(checkpoint):
    state = checkpoint.get("finetune", {})
    config = checkpoint.get("config", {})
    a = state.get("latent_geometry")
    b = config.get("latent_geometry")
    if a is not None and b is not None and a != b:
        raise ValueError("Conflicting checkpoint latent geometry metadata")
    policy = a or b or "standard"
    if policy not in POLICIES:
        raise ValueError("Unknown checkpoint latent geometry")
    return policy


def require_standard_geometry(checkpoint):
    if checkpoint_geometry(checkpoint) != "standard":
        raise ValueError("This historical tool assumes standard latent geometry; use a stride-aware evaluator/loader")


def apply_geometry(model, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown latent geometry")
    previous = getattr(model, "latent_geometry", "standard")
    hook = getattr(model, "_geometry_padding_hook", None)
    if hook is not None:
        hook.remove()
        model._geometry_padding_hook = None
    if policy == "standard" and previous == "standard":
        model.latent_geometry = policy
        return model
    if not all(hasattr(model, key) for key in ("encoder", "decoder", "downsample_levels")):
        raise ValueError("Temporal geometry requires native encoder/decoder")
    if hasattr(model, "configure_latent_geometry"):
        spatial, temporal = CHANGES[policy]
        if model.downsample_levels < max(spatial, temporal):
            raise ValueError("Unsupported geometry/downsample levels")
        model.configure_latent_geometry(spatial, temporal)
        extra = spatial_padding(policy, model.downsample_levels)
        if extra:
            model._geometry_padding_hook = model.encoder.register_forward_pre_hook(
                lambda module, inputs: (F.pad(inputs[0], (0, 0, 0, extra)),))
        model.latent_geometry = policy
        return model
    down = [m for m in model.encoder.modules() if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (4, 4) and m.stride[0] in (1, 2)]
    up = [m for m in model.decoder.modules() if isinstance(m, torch.nn.ConvTranspose2d) and m.kernel_size == (4, 4) and m.stride[0] in (1, 2)]
    spatial, temporal = CHANGES[policy]
    if len(down) != model.downsample_levels or len(up) != model.downsample_levels or len(down) < max(spatial, temporal):
        raise ValueError("Unsupported stride topology")
    if any(m.padding != (1, 1) or m.dilation != (1, 1) for m in [*down, *up]):
        raise ValueError("Unsupported convolution padding/dilation")
    if any(m.output_padding != (0, 0) for m in up):
        raise ValueError("Unsupported transpose output padding")
    for index, layer in enumerate(down):
        layer.stride = (1 if index >= len(down) - spatial else 2,
                        1 if index >= len(down) - temporal else 2)
    for index, layer in enumerate(up):
        layer.stride = (1 if index < spatial else 2, 1 if index < temporal else 2)
    extra = spatial_padding(policy, model.downsample_levels)
    if extra:
        def pad_encoder(module, inputs):
            return (F.pad(inputs[0], (0, 0, 0, extra)),)
        model._geometry_padding_hook = model.encoder.register_forward_pre_hook(pad_encoder)
    model.latent_geometry = policy
    return model


def latent_positions(model, channels, samples):
    levels = model.downsample_levels
    multiple = 2 ** levels
    policy = getattr(model, "latent_geometry", "standard")
    spatial = math.ceil(channels / multiple) * multiple + spatial_padding(policy, levels)
    temporal = math.ceil(samples / multiple) * multiple
    spatial_changes, temporal_changes = CHANGES[policy]
    if min(channels, samples) < 1:
        raise ValueError("Invalid shape/geometry")
    for level in range(levels):
        spatial_stride = 1 if level >= levels - spatial_changes else 2
        temporal_stride = 1 if level >= levels - temporal_changes else 2
        spatial = (spatial - 2) // spatial_stride + 1
        temporal = (temporal - 2) // temporal_stride + 1
    return spatial * temporal
