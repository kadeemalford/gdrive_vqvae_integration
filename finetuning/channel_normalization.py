"""Experimental channel-wise normalization with explicit float32 side parameters.

Escapes are lossless; ordinary reconstruction is lossy. This is not a complete
codec. Its additional side information must be charged in rate comparisons.
"""
import struct

import numpy as np

try:
    from .reversible_normalization import EncodedWindow, DEFAULT_THRESHOLD
except ImportError:
    from reversible_normalization import EncodedWindow, DEFAULT_THRESHOLD

HEADER = struct.Struct("<4sIIdI")
MAGIC = b"FNC1"


def encode_channel(data, threshold=DEFAULT_THRESHOLD):
    raw = np.asarray(data)
    if raw.dtype != np.dtype("float32") or raw.ndim != 2 or not raw.size:
        raise ValueError("Expected nonempty 2D native float32 window")
    if not np.isfinite(raw).all() or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Finite data and positive finite threshold required")
    if raw.size > np.iinfo(np.uint32).max:
        raise ValueError("Window exceeds index format")
    values64 = raw.astype(np.float64)
    valid = np.abs(values64) <= threshold
    if not valid.any():
        raise ValueError("No ordinary cells")
    counts = valid.sum(axis=1, keepdims=True)
    safe_counts = np.maximum(counts, 1)
    means = np.where(valid, values64, 0).sum(axis=1, keepdims=True) / safe_counts
    variance = np.where(valid, (values64 - means) ** 2, 0).sum(axis=1, keepdims=True) / safe_counts
    # Use the transmitted parameters, not hidden float64 parameters, for normalization.
    parameters = np.concatenate((means, np.sqrt(variance) + 1e-6), axis=1).astype("<f4")
    transmitted_mean = parameters[:, 0:1].astype(np.float64)
    transmitted_scale = parameters[:, 1:2].astype(np.float64)
    normalized = np.where(valid, (values64 - transmitted_mean) / transmitted_scale, 0).astype(np.float32)
    indices = np.flatnonzero(~valid).astype("<u4")
    stream = (HEADER.pack(MAGIC, *raw.shape, threshold, len(indices)) + parameters.tobytes()
              + indices.tobytes() + raw.ravel()[indices].astype("<f4").tobytes())
    return EncodedWindow(normalized, valid, stream)


def parse_channel_stream(stream):
    if len(stream) < HEADER.size:
        raise ValueError("Truncated channel stream")
    magic, channels, samples, threshold, count = HEADER.unpack_from(stream)
    if (magic != MAGIC or not channels or not samples or not np.isfinite(threshold) or threshold <= 0
            or len(stream) != HEADER.size + 8 * channels + 8 * count):
        raise ValueError("Invalid channel stream header/length")
    parameters = np.frombuffer(stream, dtype="<f4", count=2 * channels, offset=HEADER.size).reshape(channels, 2)
    offset = HEADER.size + 8 * channels
    indices = np.frombuffer(stream, dtype="<u4", count=count, offset=offset)
    escaped = np.frombuffer(stream, dtype="<f4", count=count, offset=offset + 4 * count)
    if (not np.isfinite(parameters).all() or np.any(parameters[:, 1] <= 0)
            or np.any(indices >= channels * samples) or (count > 1 and np.any(indices[1:] <= indices[:-1]))
            or not np.isfinite(escaped).all() or np.any(np.abs(escaped.astype(np.float64)) <= threshold)):
        raise ValueError("Invalid channel parameters/escapes")
    return (channels, samples), parameters, indices, escaped


def decode_channel(values, stream):
    shape, parameters, indices, escaped = parse_channel_stream(stream)
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("Invalid model reconstruction")
    restored64 = values.astype(np.float64) * parameters[:, 1:2].astype(np.float64) + parameters[:, 0:1].astype(np.float64)
    restored64.ravel()[indices] = escaped
    if np.any(np.abs(restored64) > np.finfo(np.float32).max):
        raise ValueError("Ordinary reconstruction overflows float32")
    restored = restored64.astype(np.float32)
    restored.ravel()[indices] = escaped
    return restored
