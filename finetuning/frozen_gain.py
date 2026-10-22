"""Train-only equal-window least-squares gain, folded into a linear decoder head."""
import math

import numpy as np
import torch


def window_moments(target, prediction, valid):
    if target.shape != prediction.shape or valid.shape != target.shape or not valid.any():
        raise ValueError("Incompatible or empty ordinary region")
    x = np.asarray(target[valid], dtype=np.float64)
    y = np.asarray(prediction[valid], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite ordinary calibration data")
    return {"xy": float(np.mean(x * y)), "yy": float(np.mean(y * y)), "xx": float(np.mean(x * x))}


def fitted_gain(totals):
    if not all(math.isfinite(totals[k]) for k in ("xy", "yy", "xx")) or totals["yy"] <= 1e-12:
        raise ValueError("Invalid calibration moments")
    gain = float(np.float32(totals["xy"] / totals["yy"]))
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Positive finite gain required")
    return gain


def fold_gain(model, gain):
    """Change only the last affine convolution, with no added parameters."""
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Positive finite gain required")
    if not isinstance(model.decoder.net, torch.nn.Sequential) or not isinstance(model.decoder.net[-1], torch.nn.Conv2d):
        raise ValueError("A final linear Conv2d decoder head is required")
    head = model.decoder.net[-1]
    with torch.no_grad():
        head.weight.mul_(gain)
        if head.bias is not None:
            head.bias.mul_(gain)
        if any(not torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("Nonfinite calibrated head")
