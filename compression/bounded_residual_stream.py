"""Float32 residual correction with a transmitted absolute bound and exact escapes."""
import hashlib
import zlib

import numpy as np

from compression.adaptive_coefficient_stream import inflate
from compression.predictive_coefficient_stream import pack_predictive, unpack_predictive

MAX_CELLS = 16 * 1024 * 1024


def base_digest(base):
    return hashlib.sha256(np.asarray(base, dtype="<f4").tobytes()).hexdigest()


def validate_base(base):
    base = np.asarray(base)
    if base.dtype != np.float32 or base.ndim != 2 or not 0 < base.size <= MAX_CELLS:
        raise ValueError("Expected bounded nonempty 2D float32 base")
    return base


def reconstruct(base, coefficients, epsilon):
    origin = np.where(np.isfinite(base), base, 0).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return (origin + coefficients.astype(np.float64) * (2 * epsilon)).astype(np.float32)


def encode(raw, base, epsilon):
    base = validate_base(base)
    raw = validate_base(raw)
    epsilon = float(epsilon)
    if raw.shape != base.shape or not np.isfinite(epsilon) or not 0 < epsilon <= np.finfo(np.float64).max / 2:
        raise ValueError("Invalid residual shape or absolute tolerance")
    origin = np.where(np.isfinite(base), base, 0).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        quotient = np.rint((raw.astype(np.float64) - origin) / (2 * epsilon))
    usable = np.isfinite(quotient) & (quotient >= -(2**31)) & (quotient <= 2**31 - 1)
    coefficients = np.where(usable, quotient, 0).astype(np.int32)
    candidate = reconstruct(base, coefficients, epsilon)
    with np.errstate(invalid="ignore"):
        ordinary = (np.isfinite(raw) & np.isfinite(candidate)
                    & (np.abs(raw.astype(np.float64) - candidate.astype(np.float64)) <= epsilon))
    escaped = ~ordinary | (raw == 0)
    positions = np.flatnonzero(escaped).astype("<u8")
    bits = raw.ravel().view(np.uint32)[positions].astype("<u4")
    coefficients.ravel()[positions] = 0
    side = zlib.compress(positions.tobytes() + bits.tobytes(), 6)
    descriptor = {"kind": "absolute-bounded-float32-v1", "epsilon": epsilon,
                  "base_sha256": base_digest(base), "escapes": len(positions)}
    return pack_predictive([coefficients], descriptor, side)


def decode(frame, base):
    base = validate_base(base)
    arrays, descriptor, side = unpack_predictive(frame)
    if not isinstance(descriptor, dict):
        raise ValueError("Invalid bounded residual descriptor")
    epsilon, count = descriptor.get("epsilon"), descriptor.get("escapes")
    if (descriptor.get("kind") != "absolute-bounded-float32-v1"
            or type(epsilon) not in (int, float) or not np.isfinite(epsilon)
            or not 0 < epsilon <= np.finfo(np.float64).max / 2
            or type(count) is not int or not 0 <= count <= base.size
            or len(arrays) != 1 or arrays[0].shape != base.shape
            or descriptor.get("base_sha256") != base_digest(base)):
        raise ValueError("Invalid bounded residual contract or base")
    payload = inflate(side, count * 12)
    positions = np.frombuffer(payload, "<u8", count=count)
    bits = np.frombuffer(payload, "<u4", count=count, offset=count * 8)
    if count and (positions[-1] >= base.size or np.any(positions[1:] <= positions[:-1])):
        raise ValueError("Invalid bounded residual escape positions")
    result = reconstruct(base, arrays[0], epsilon)
    result.ravel().view(np.uint32)[positions] = bits
    return result


def audit(raw, restored, epsilon):
    finite = np.isfinite(raw)
    error = np.abs(raw[finite].astype(np.float64) - restored[finite].astype(np.float64))
    exact = ~finite | (raw == 0)
    if (not np.all(np.isfinite(restored[finite])) or np.any(error > epsilon)
            or not np.array_equal(raw.view(np.uint32)[exact], restored.view(np.uint32)[exact])):
        raise AssertionError("Physical absolute error or exact special-value contract violated")
    return {"maximum_absolute_error": float(error.max(initial=0)),
            "finite_cells": int(finite.sum()), "exact_special_cells": int(exact.sum())}
