"""Explicit FNX1/FAX1 side contracts preserving nonfinite float32 bit patterns.

This extends the research codec, never the original finite-only FNE1 parser.
Ordinary finite values are normalized; nonfinite/finite-extreme cells are exact
uint32 escapes and are charged in the transmitted side. No imputation or clipping.
"""
import numpy as np
import zlib

from finetuning.reversible_normalization import HEADER, EncodedWindow, parse_side_stream
from finetuning.robust_escape import encode_robust, robust_threshold

NUMERIC_MAGIC = b'FNX1'
AFFINE_NUMERIC_MAGIC = b'FAX1'
COMPRESSED_MAGIC = b'FNZ1'
AFFINE_COMPRESSED_MAGIC = b'FAZ1'
NUMERIC_MAGICS = (NUMERIC_MAGIC, AFFINE_NUMERIC_MAGIC, COMPRESSED_MAGIC, AFFINE_COMPRESSED_MAGIC)


def parse_numeric_side(side):
    if len(side) < HEADER.size:
        raise ValueError('Truncated numeric side')
    magic, h, w, mean, scale, threshold, count = HEADER.unpack_from(side)
    if (magic not in NUMERIC_MAGICS or not h or not w or
            not np.isfinite([mean, scale, threshold]).all() or scale <= 0 or threshold <= 0 or
            count > h * w or 8 * count > 64 * 1024 * 1024):
        raise ValueError('Invalid numeric side header/length')
    if magic in (COMPRESSED_MAGIC, AFFINE_COMPRESSED_MAGIC):
        try:
            codec = zlib.decompressobj()
            body = codec.decompress(side[HEADER.size:], 8 * count + 1)
        except zlib.error as error:
            raise ValueError('Corrupt compressed numeric side') from error
        if len(body) != 8 * count or not codec.eof or codec.unused_data or codec.unconsumed_tail:
            raise ValueError('Invalid compressed numeric side length/termination')
        delta = np.frombuffer(body, dtype='<u4', count=count)
        cumulative = np.cumsum(delta, dtype=np.uint64)
        if np.any(cumulative >= h * w):
            raise ValueError('Numeric delta index overflow')
        indices = cumulative.astype('<u4')
        bits = np.frombuffer(body, dtype='<u4', count=count, offset=4 * count)
    else:
        if len(side) != HEADER.size + 8 * count:
            raise ValueError('Invalid raw numeric side length')
        indices = np.frombuffer(side, dtype='<u4', count=count, offset=HEADER.size)
        bits = np.frombuffer(side, dtype='<u4', count=count, offset=HEADER.size + 4 * count)
    if np.any(indices >= h * w) or (count > 1 and np.any(indices[1:] <= indices[:-1])):
        raise ValueError('Invalid numeric escape indices')
    values = bits.view('<f4')
    finite = np.isfinite(values)
    if np.any(np.abs(values[finite].astype(np.float64)) <= threshold):
        raise ValueError('Finite numeric escape does not exceed threshold')
    return (h, w), mean, scale, indices, bits


def parse_input_side(side):
    if side[:4] in (NUMERIC_MAGIC, COMPRESSED_MAGIC):
        shape, mean, scale, indices, bits = parse_numeric_side(side)
        return shape, mean, scale, indices, bits.view('<f4')
    return parse_side_stream(side)


def encode_numeric(raw):
    raw = np.asarray(raw)
    if raw.dtype != np.dtype('float32') or raw.ndim != 2 or not raw.size:
        raise ValueError('Expected nonempty native float32 2D data')
    finite = np.isfinite(raw)
    if finite.all():
        return encode_robust(raw)
    if not finite.any():
        raise ValueError('All-nonfinite frame has no ordinary target; separate exact-only contract required')
    threshold = robust_threshold(raw[finite].reshape(1, -1))
    valid = finite & (np.abs(raw.astype(np.float64)) <= threshold)
    if not valid.any() or raw.size > np.iinfo(np.uint32).max:
        raise ValueError('Numeric frame has no ordinary target or exceeds index limit')
    ordinary = raw[valid].astype(np.float64)
    mean, scale = float(ordinary.mean()), float(ordinary.std()) + 1e-6
    values = np.zeros(raw.shape, dtype=np.float32)
    values[valid] = ((ordinary - mean) / scale).astype(np.float32)
    indices = np.flatnonzero(~valid).astype('<u4')
    bits = raw.reshape(-1).view(np.uint32)[indices].astype('<u4')
    side = HEADER.pack(NUMERIC_MAGIC, *raw.shape, mean, scale, threshold, len(indices)) + indices.tobytes() + bits.tobytes()
    delta = np.diff(np.concatenate([np.array([0], dtype=np.uint64), indices.astype(np.uint64)])).astype('<u4')
    compressed = zlib.compress(delta.tobytes() + bits.tobytes(), level=6)
    if len(compressed) < 8 * len(indices):
        side = HEADER.pack(COMPRESSED_MAGIC, *raw.shape, mean, scale, threshold, len(indices)) + compressed
    return EncodedWindow(values, valid, side)


def restore_numeric(values, side):
    shape, mean, scale, indices, bits = parse_numeric_side(side)
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError('Invalid numeric reconstruction')
    valid = np.ones(values.size, dtype=bool)
    valid[indices] = False
    ordinary = values.reshape(-1)[valid].astype(np.float64) * scale + mean
    if not np.isfinite(ordinary).all() or np.any(np.abs(ordinary) > np.finfo(np.float32).max):
        raise ValueError('Numeric ordinary restoration overflows float32')
    result = np.zeros(shape, dtype=np.float32)
    result.reshape(-1)[valid] = ordinary.astype(np.float32)
    result.reshape(-1).view(np.uint32)[indices] = bits
    return result
