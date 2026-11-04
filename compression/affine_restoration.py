"""Versioned adaptive affine restoration; ordinary input normalization stays private.

FAS1 reuses the byte layout, not the semantics, of the FNE1 side header:
the two float64 fields are output offset/scale rather than input mean/std.
Escapes, shape and threshold remain unchanged. A descriptor must declare FAS1.
"""
import numpy as np

from finetuning.reversible_normalization import HEADER, MAGIC, parse_side_stream, decode
from compression.numeric_restoration import (parse_input_side, NUMERIC_MAGIC, AFFINE_NUMERIC_MAGIC,
    COMPRESSED_MAGIC, AFFINE_COMPRESSED_MAGIC, NUMERIC_MAGICS, restore_numeric)

AFFINE_MAGIC = b'FAS1'


def affine_side(side, gain, offset):
    shape, mean, scale, _, _ = parse_input_side(side)
    if not np.isfinite([gain, offset]).all() or gain <= 0:
        raise ValueError('Finite positive affine gain and finite offset required')
    fields = list(HEADER.unpack_from(side))
    fields[0] = (AFFINE_NUMERIC_MAGIC if side[:4] == NUMERIC_MAGIC else
                 AFFINE_COMPRESSED_MAGIC if side[:4] == COMPRESSED_MAGIC else AFFINE_MAGIC)
    fields[3] = mean + scale * offset
    fields[4] = scale * gain
    if not np.isfinite(fields[3:5]).all() or fields[4] <= 0:
        raise ValueError('Affine output scale/offset overflow')
    result = HEADER.pack(*fields) + side[HEADER.size:]
    assert len(result) == len(side)
    return result


def restore(values, side):
    if side[:4] in NUMERIC_MAGICS:
        return restore_numeric(values, side)
    if side[:4] == AFFINE_MAGIC:
        # Reuse the verified escape/overflow implementation under explicit semantics.
        side = MAGIC + side[4:]
    return decode(values, side)


def candidates(target, prediction, valid, side, baseline_rms=None):
    """Two fixed encoder-available choices; no fit on validation across windows.

    Per-window coefficients are encoded, not shared weights learned on validation.
    OLS minimizes ordinary squared error. Unit-RMS is a separate constrained probe.
    Neither is silently called optimal for the mixed MSE/L1 or physical metrics.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if target.shape != prediction.shape or target.shape != valid.shape or target.ndim != 2 or not valid.any():
        raise ValueError('Matching nonempty 2D arrays and ordinary mask required')
    x, y = target[valid], prediction[valid]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Ordinary affine fitting values must be finite')
    xm, ym = float(x.mean()), float(y.mean())
    xc, yc = x - xm, y - ym
    xp, yp = float(xc @ xc), float(yc @ yc)
    if xp <= 1e-20 or yp <= 1e-20:
        return []
    gains = [('ols', float(xc @ yc) / yp), ('unit_rms', float(np.sqrt(xp / yp)))]
    if baseline_rms is not None:
        if not np.isfinite(baseline_rms) or baseline_rms < 0:
            raise ValueError('Finite nonnegative source RMS ratio required')
        distance = abs(baseline_rms - 1.)
        lo, hi = max(0., 1. - distance), 1. + distance
        target_power = float(x @ x)
        gain_lo = np.sqrt(max(0., lo * lo * target_power - len(x) * xm * xm) / yp)
        gain_hi = np.sqrt(max(0., hi * hi * target_power - len(x) * xm * xm) / yp)
        # Move inside the computed feasible interval to tolerate float32 output
        # rounding; the actual, unchanged physical guard is still mandatory.
        margin = min(.25 * (gain_hi - gain_lo),
                     32 * np.finfo(np.float32).eps * max(1., gain_lo, gain_hi))
        bounded = float(np.clip(gains[0][1], gain_lo + margin, gain_hi - margin))
        gains.append(('bounded_ols', bounded))
    results = []
    for name, gain in gains:
        offset = xm - gain * ym
        if not np.isfinite([gain, offset]).all() or gain <= 0:
            continue
        result = affine_side(side, gain, offset)
        results.append({'kind': name, 'gain': gain, 'offset': offset, 'side': result})
    return results
