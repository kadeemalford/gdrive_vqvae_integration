"""Experimental FLR1: transmitted 8-bit spatial/temporal low-rank factors."""
import struct
import zlib

import numpy as np

from compression.affine_restoration import restore
from compression.numeric_restoration import parse_input_side

MAGIC = b'FLR1'
HEADER = struct.Struct('<4sBBHIIdIII')
MAX_RANK = 64
MAX_CELLS = 16 * 1024 * 1024


def _frame(shape, rank, gain, side, payload):
    head = HEADER.pack(MAGIC, 1, 8, rank, *shape, gain, len(side), len(payload), 0)
    crc = zlib.crc32(head + side + payload)
    return head[:-4] + struct.pack('<I', crc) + side + payload


def factorize(packet):
    values = np.asarray(packet.values)
    if (values.dtype != np.float32 or values.ndim != 2 or not values.size
            or values.size > MAX_CELLS or not np.isfinite(values).all()):
        raise ValueError('Expected finite protected float32 matrix')
    u, singular, vt = np.linalg.svd(values.astype(np.float64), full_matrices=False)
    return u[:, :MAX_RANK], singular[:MAX_RANK], vt[:MAX_RANK]


def compress_factors(packet, u, singular, vt, rank):
    shape = packet.values.shape
    if (len(shape) != 2 or min(shape) < 1 or np.prod(shape) > MAX_CELLS
            or not isinstance(rank, (int, np.integer)) or not 0 <= rank <= min(MAX_RANK, *shape)
            or u.shape[0] != shape[0] or vt.shape[1] != shape[1]
            or u.shape[1] < rank or vt.shape[0] < rank or len(singular) < rank):
        raise ValueError('Invalid factor geometry/rank')
    side = packet.side_stream
    if side[:4] not in (b'FNE1', b'FNX1', b'FNZ1') or parse_input_side(side)[0] != shape:
        raise ValueError('Low-rank codec requires matching input-normalization side')
    left, right = u[:, :rank], singular[:rank, None] * vt[:rank]
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Nonfinite factors')
    if rank:
        a = np.maximum(np.abs(left).max(axis=0) / 127., np.finfo(np.float32).tiny).astype('<f4')
        b = np.maximum(np.abs(right).max(axis=1) / 127., np.finfo(np.float32).tiny).astype('<f4')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Factor scales overflow')
        qa = np.clip(np.rint(left / a), -127, 127).astype(np.int8)
        qb = np.clip(np.rint(right / b[:, None]), -127, 127).astype(np.int8)
        quantized = np.concatenate((qa.T, qb), axis=1).view(np.uint8)
        delta = quantized.copy()
        delta[:, 1:] = quantized[:, 1:] - quantized[:, :-1]
        scales = a.tobytes() + b.tobytes()
    else:
        quantized = delta = np.empty((0, sum(shape)), dtype=np.uint8)
        scales = b''
    payloads = [bytes([mode]) + scales + zlib.compress(values.tobytes(), level=6)
                for mode, values in enumerate((quantized, delta))]
    payload = min(payloads, key=len)
    return _frame(shape, rank, 1., side, payload)


def with_gain(frame, gain):
    # Validate the original before changing a transmitted output coefficient.
    decode_normalized(frame)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Finite positive low-rank gain required')
    fields = list(HEADER.unpack_from(frame))
    fields[6] = float(gain)
    fields[-1] = 0
    head = HEADER.pack(*fields)
    crc = zlib.crc32(head + frame[HEADER.size:])
    return head[:-4] + struct.pack('<I', crc) + frame[HEADER.size:]


def decode_normalized(frame):
    if len(frame) < HEADER.size:
        raise ValueError('Truncated low-rank frame')
    fields = list(HEADER.unpack_from(frame))
    magic, version, bits, rank, h, w, gain, side_size, payload_size, crc = fields
    if (magic != MAGIC or version != 1 or bits != 8 or min(h, w) < 1 or h * w > MAX_CELLS
            or rank > min(MAX_RANK, h, w) or not np.isfinite(gain) or gain <= 0
            or side_size < 4 or payload_size < 1 + 8 * rank
            or HEADER.size + side_size + payload_size != len(frame)):
        raise ValueError('Invalid low-rank header/lengths')
    fields[-1] = 0
    if zlib.crc32(HEADER.pack(*fields) + frame[HEADER.size:]) != crc:
        raise ValueError('Low-rank frame CRC mismatch')
    side = frame[HEADER.size:HEADER.size + side_size]
    if side[:4] not in (b'FNE1', b'FNX1', b'FNZ1') or parse_input_side(side)[0] != (h, w):
        raise ValueError('Invalid low-rank input side semantics/shape')
    body = frame[HEADER.size + side_size:]
    mode = body[0]
    if mode not in (0, 1):
        raise ValueError('Invalid factor predictor')
    scales = np.frombuffer(body, dtype='<f4', count=2 * rank, offset=1)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Invalid factor scales')
    expected = rank * (h + w)
    try:
        codec = zlib.decompressobj()
        data = codec.decompress(body[1 + 8 * rank:], expected + 1)
    except zlib.error as error:
        raise ValueError('Corrupt low-rank payload') from error
    if len(data) != expected or not codec.eof or codec.unused_data or codec.unconsumed_tail:
        raise ValueError('Invalid factor decompression length/termination')
    if rank:
        integers = np.frombuffer(data, dtype=np.uint8).reshape(rank, h + w)
        if mode:
            integers = integers.cumsum(axis=1, dtype=np.uint64).astype(np.uint8)
        integers = integers.view(np.int8)
        if np.any(integers == -128):
            raise ValueError('Noncanonical signed factor value')
        left = integers[:, :h].T.astype(np.float64) * scales[:rank]
        right = integers[:, h:].astype(np.float64) * scales[rank:, None]
        with np.errstate(over='ignore', invalid='ignore'):
            values = ((left @ right) * gain).astype(np.float32)
    else:
        values = np.zeros((h, w), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Low-rank output overflows float32')
    return values, side, {'rank': rank, 'factor_bits': bits, 'gain': gain, 'factor_predictor': mode}


def decompress(frame):
    values, side, _ = decode_normalized(frame)
    return restore(values, side)
