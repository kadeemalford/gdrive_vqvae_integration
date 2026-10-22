"""Conservative, unit-adaptive dynamic-range escape; not a physical validity label."""
import numpy as np

try:
    from .reversible_normalization import encode, DEFAULT_THRESHOLD
except ImportError:
    from reversible_normalization import encode, DEFAULT_THRESHOLD


def robust_threshold(raw):
    raw = np.asarray(raw)
    if raw.dtype != np.dtype("float32") or raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError("Finite nonempty 2D float32 required")
    magnitude = np.abs(raw.astype(np.float64)).reshape(-1)
    reference = float(np.quantile(magnitude, .99))
    if reference == 0:
        # Do not route every legitimate sparse impulse into escape just because
        # the mostly-zero background makes p99 zero.
        positive = magnitude[magnitude > 0]
        reference = float(np.quantile(positive, .99)) if len(positive) else 0.
    if reference == 0:
        return DEFAULT_THRESHOLD
    return min(DEFAULT_THRESHOLD, max(1e-30, reference * 1000.))


def encode_robust(raw):
    return encode(raw, threshold=robust_threshold(raw))
