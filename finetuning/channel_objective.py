"""Optional bounded inverse-variance channel emphasis on ordinary normalized cells."""
import torch


CHANNEL_WEIGHT_CAP = 4.0


def bounded_channel_mse(prediction, target, valid):
    if (prediction.shape != target.shape or valid.shape != target.shape or target.ndim != 4
            or target.shape[:2] != (1, 1) or valid.dtype != torch.bool):
        raise ValueError("Channel objective requires matching [1, 1, channels, time] tensors and boolean mask")
    mask = valid[0, 0]
    counts = mask.sum(dim=-1)
    used = counts > 0
    if not used.any():
        raise ValueError("No ordinary cells for channel objective")
    x = torch.where(mask, target[0, 0].detach(), 0.)
    y = torch.where(mask, prediction[0, 0], 0.)
    channel_mean = x.sum(dim=-1) / counts.clamp_min(1)
    centered = torch.where(mask, x - channel_mean[:, None], 0.)
    channel_variance = centered.square().sum(dim=-1) / counts.clamp_min(1)
    global_mean = x.sum() / counts.sum()
    global_variance = torch.where(mask, x - global_mean, 0.).square().sum() / counts.sum()
    if global_variance <= 1e-12:
        weights = torch.ones_like(channel_variance)
    else:
        weights = (global_variance / channel_variance.clamp_min(global_variance / CHANNEL_WEIGHT_CAP)).clamp(
            min=1 / CHANNEL_WEIGHT_CAP, max=CHANNEL_WEIGHT_CAP)
    channel_error = (y - x).square().sum(dim=-1) / counts.clamp_min(1)
    return (weights.detach()[used] * channel_error[used]).mean()
