"""Versioned float32 escape stream; no clipping or mutation of source arrays.

Only escaped cells are lossless. Ordinary cells still incur normalization and
model error. This is preprocessing/side information, not a complete DAS codec.
"""
from dataclasses import dataclass
import struct

import numpy as np

HEADER = struct.Struct("<4sIIdddI")
MAGIC = b"FNE1"
DEFAULT_THRESHOLD = 1e20


@dataclass
class EncodedWindow:
    values: np.ndarray
    valid: np.ndarray
    side_stream: bytes


def encode(data, threshold=DEFAULT_THRESHOLD):
    raw = np.asarray(data)
    if raw.dtype != np.dtype("float32") or raw.ndim != 2 or not raw.size:
        raise ValueError("Expected a nonempty 2D native float32 window")
    if not np.isfinite(raw).all() or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Finite data and positive finite threshold required")
    valid = np.abs(raw.astype(np.float64)) <= threshold
    if not valid.any():
        raise ValueError("No ordinary cells remain to normalize")
    ordinary = raw[valid].astype(np.float64)
    mean, scale = float(ordinary.mean()), float(ordinary.std()) + 1e-6
    values = np.zeros(raw.shape, dtype=np.float32)
    values[valid] = ((ordinary - mean) / scale).astype(np.float32)
    indices = np.flatnonzero(~valid).astype("<u4")
    if raw.size > np.iinfo(np.uint32).max:
        raise ValueError("Window exceeds format index limit")
    escaped = raw.ravel()[indices].astype("<f4")
    stream = (HEADER.pack(MAGIC, *raw.shape, mean, scale, threshold, len(indices))
              + indices.tobytes() + escaped.tobytes())
    return EncodedWindow(values, valid, stream)


def parse_side_stream(stream):
    if len(stream) < HEADER.size:
        raise ValueError("Truncated side stream")
    magic, channels, samples, mean, scale, threshold, count = HEADER.unpack_from(stream)
    if (magic != MAGIC or not channels or not samples or not np.isfinite([mean, scale, threshold]).all()
            or scale <= 0 or threshold <= 0 or len(stream) != HEADER.size + 8 * count):
        raise ValueError("Invalid side stream header/length")
    indices = np.frombuffer(stream, dtype="<u4", count=count, offset=HEADER.size)
    escaped = np.frombuffer(stream, dtype="<f4", count=count, offset=HEADER.size + 4 * count)
    if (np.any(indices >= channels * samples) or (count > 1 and np.any(indices[1:] <= indices[:-1]))
            or not np.isfinite(escaped).all() or np.any(np.abs(escaped.astype(np.float64)) <= threshold)):
        raise ValueError("Invalid escape entries")
    return (channels, samples), mean, scale, indices, escaped


def decode(values, side_stream):
    shape, mean, scale, indices, escaped = parse_side_stream(side_stream)
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("Decoded model window has invalid geometry/values")
    # Restore ordinary region in float64 before final float32 conversion.
    restored64 = values.astype(np.float64) * scale + mean
    restored64.ravel()[indices] = escaped
    if np.any(np.abs(restored64) > np.finfo(np.float32).max):
        raise ValueError("Ordinary model reconstruction overflows float32")
    restored = restored64.astype(np.float32)
    restored.ravel()[indices] = escaped
    return restored
