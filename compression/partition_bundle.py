"""FPB1: full-row-coverage bundles of neural frames and exact nonfinite fragments."""
import struct
import zlib

import numpy as np

HEADER = struct.Struct('<4sIII')
ENTRY = struct.Struct('<BIII')
CRC = struct.Struct('<I')
MAGIC = b'FPB1'
MAX_CELLS = 16 * 1024 * 1024


def exact_nonfinite(raw):
    raw = np.asarray(raw)
    if raw.dtype != np.dtype('float32') or raw.ndim != 2 or not raw.size or np.isfinite(raw).any():
        raise ValueError('Exact fragments may contain only nonfinite float32 cells')
    return zlib.compress(raw.view(np.uint32).astype('<u4').tobytes(), level=6)


def pack_bundle(shape, fragments):
    h, w = shape
    if min(shape) < 1 or h * w > MAX_CELLS or not fragments or len(fragments) > min(h, 1024):
        raise ValueError('Invalid bundle shape or fragment count')
    body = bytearray(HEADER.pack(MAGIC, h, w, len(fragments)))
    cursor = 0
    for kind, start, rows, payload in fragments:
        if kind not in (0, 1) or start != cursor or rows < 1 or start + rows > h or not payload:
            raise ValueError('Fragments must cover all rows once in order')
        body.extend(ENTRY.pack(kind, start, rows, len(payload)))
        body.extend(payload)
        cursor += rows
    if cursor != h:
        raise ValueError('Incomplete bundle coverage')
    return bytes(body) + CRC.pack(zlib.crc32(body))


def unpack_bundle(bundle):
    if len(bundle) < HEADER.size + CRC.size or zlib.crc32(bundle[:-4]) != CRC.unpack(bundle[-4:])[0]:
        raise ValueError('Truncated/corrupt bundle')
    magic, h, w, count = HEADER.unpack_from(bundle)
    if magic != MAGIC or min(h, w) < 1 or h * w > MAX_CELLS or not 0 < count <= min(h, 1024):
        raise ValueError('Invalid bundle header')
    position, cursor, fragments = HEADER.size, 0, []
    for _ in range(count):
        if position + ENTRY.size > len(bundle) - 4:
            raise ValueError('Truncated bundle entry')
        kind, start, rows, size = ENTRY.unpack_from(bundle, position)
        position += ENTRY.size
        if kind not in (0, 1) or start != cursor or not rows or start + rows > h or not size or position + size > len(bundle) - 4:
            raise ValueError('Invalid bundle coverage/length')
        fragments.append((kind, start, rows, bundle[position:position + size]))
        position += size
        cursor += rows
    if position != len(bundle) - 4 or cursor != h:
        raise ValueError('Incomplete or trailing bundle data')
    return (h, w), fragments


def decode_bundle(bundle, stream, model, device):
    from finetuning.affine_codec_study import decode_affine_frame
    shape, fragments = unpack_bundle(bundle)
    result = np.empty(shape, dtype=np.float32)
    for kind, start, rows, payload in fragments:
        expected_shape = rows, shape[1]
        if kind == 0:
            decoded = decode_affine_frame(payload, stream, model, device)
            if decoded.shape != expected_shape:
                raise ValueError('Neural fragment geometry mismatch')
        else:
            expected = rows * shape[1] * 4
            try:
                codec = zlib.decompressobj()
                raw = codec.decompress(payload, expected + 1)
            except zlib.error as error:
                raise ValueError('Corrupt exact fragment') from error
            if len(raw) != expected or not codec.eof or codec.unused_data or codec.unconsumed_tail:
                raise ValueError('Invalid exact fragment length/termination')
            decoded = np.frombuffer(raw, dtype='<f4').reshape(expected_shape)
            if np.isfinite(decoded).any():
                raise ValueError('Exact-only fragments cannot bypass finite neural targets')
        result.view(np.uint32)[start:start + rows] = decoded.view(np.uint32)
    return result
