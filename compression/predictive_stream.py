"""Experimental FPI1: bounded scalar quantization, integer differences and byte planes.

The bound applies to the protected float32 normalized target, not an operational
physical tolerance. Exact escapes and normalization are transmitted in the side.
"""
import struct
import zlib

import numpy as np

from compression.affine_restoration import restore
from compression.numeric_restoration import parse_input_side, encode_numeric

MAGIC = b'FPI1'
HEADER = struct.Struct('<4sBBIIdIII')
MAX_CELLS = 16 * 1024 * 1024
BIN_LIMIT = (1 << 29) - 1
PREDICTORS = ('identity', 'temporal', 'lorenzo2d')


def _residuals(grid, method):
    output = grid.copy()
    if method in (1, 2):
        output[:, 1:] -= grid[:, :-1]
    if method == 2:
        previous = output.copy()
        output[1:] -= previous[:-1]
    return output


def compress_packet(packet, epsilon):
    values = np.asarray(packet.values)
    valid = np.asarray(packet.valid)
    if (values.dtype != np.float32 or values.ndim != 2 or not values.size or values.size > MAX_CELLS
            or valid.shape != values.shape or valid.dtype != bool or not valid.any()
            or not np.isfinite(values).all() or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Expected finite protected float32 target and positive normalized bound')
    side = packet.side_stream
    if side[:4] not in (b'FNE1', b'FNX1', b'FNZ1'):
        raise ValueError('Predictive codec requires INPUT-normalization side, not affine output side')
    shape, _, _, escaped, _ = parse_input_side(side)
    expected_valid = np.ones(values.size, dtype=bool)
    expected_valid[escaped] = False
    if shape != values.shape or not np.array_equal(expected_valid.reshape(shape), valid):
        raise ValueError('Input side, ordinary mask and target geometry differ')
    step = 2. * float(epsilon)
    if not np.isfinite(step) or step <= 0 or float(np.abs(values[valid]).max()) / BIN_LIMIT > step:
        raise ValueError('Requested bound exceeds supported integer precision')
    # Invalid cells are exact escapes, never approximated or imputed at the receiver.
    grid = np.zeros(values.shape, dtype=np.int64)
    grid[valid] = np.rint(values[valid].astype(np.float64) / step).astype(np.int64)
    candidates = []
    for method in range(3):
        residual = _residuals(grid, method)
        if residual.min() < np.iinfo(np.int32).min or residual.max() > np.iinfo(np.int32).max:
            raise ValueError('Predictive integer overflow')
        planes = residual.astype('<i4').view(np.uint8).reshape(-1, 4).T.copy().tobytes()
        payload = zlib.compress(planes, level=6)
        header = HEADER.pack(MAGIC, 1, method, *shape, step, len(side), len(payload), 0)
        crc = zlib.crc32(header + side + payload)
        frame = header[:-4] + struct.pack('<I', crc) + side + payload
        candidates.append(frame)
    frame = min(candidates, key=lambda f: (len(f), f[5]))
    return frame, {'frame_bytes': len(frame), 'normalized_epsilon': float(epsilon),
                   'predictor': PREDICTORS[frame[5]], 'predictor_bytes': [len(f) for f in candidates]}


def compress(raw, epsilon):
    return compress_packet(encode_numeric(np.asarray(raw)), epsilon)


def decode_normalized(frame):
    if len(frame) < HEADER.size:
        raise ValueError('Truncated predictive frame')
    fields = list(HEADER.unpack_from(frame))
    magic, version, method, h, w, step, side_size, payload_size, crc = fields
    if (magic != MAGIC or version != 1 or method not in (0, 1, 2) or min(h, w) < 1
            or h * w > MAX_CELLS or not np.isfinite(step) or step <= 0
            or side_size < 4 or payload_size < 1 or HEADER.size + side_size + payload_size != len(frame)):
        raise ValueError('Invalid predictive header or lengths')
    fields[-1] = 0
    if zlib.crc32(HEADER.pack(*fields) + frame[HEADER.size:]) != crc:
        raise ValueError('Predictive frame CRC mismatch')
    side = frame[HEADER.size:HEADER.size + side_size]
    if side[:4] not in (b'FNE1', b'FNX1', b'FNZ1'):
        raise ValueError('Invalid predictive side semantics')
    shape, _, _, escaped, _ = parse_input_side(side)
    if shape != (h, w):
        raise ValueError('Predictive side geometry mismatch')
    expected = h * w * 4
    try:
        codec = zlib.decompressobj()
        data = codec.decompress(frame[HEADER.size + side_size:], expected + 1)
    except zlib.error as error:
        raise ValueError('Corrupt predictive payload') from error
    if len(data) != expected or not codec.eof or codec.unconsumed_tail or codec.unused_data:
        raise ValueError('Invalid predictive decompression length/termination')
    packed = np.frombuffer(data, dtype=np.uint8).reshape(4, -1).T.copy()
    grid = packed.view('<i4').reshape(h, w).astype(np.int64)
    if method == 2:
        grid = grid.cumsum(axis=0, dtype=np.int64)
    if method in (1, 2):
        grid = grid.cumsum(axis=1, dtype=np.int64)
    if np.abs(grid).max() > BIN_LIMIT or np.any(grid.reshape(-1)[escaped] != 0):
        raise ValueError('Invalid reconstructed integer grid or escape locations')
    with np.errstate(over='ignore', invalid='ignore'):
        normalized = (grid.astype(np.float64) * step).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError('Predictive reconstruction overflows float32')
    return normalized, side, {'normalized_epsilon': step / 2, 'predictor': PREDICTORS[method]}


def decompress(frame):
    values, side, _ = decode_normalized(frame)
    return restore(values, side)
