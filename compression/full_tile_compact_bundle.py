"""FTC1 diagnostic container: deterministic wrapper elision, intact inner frames.

Installed receiver/registry are external common installation. Geometry is opaque
and charged once. Every tile has an absolute offset, length, mode and CRC; SHA256
protects the index and complete object. No neural decoder or weights are loaded.
"""
import hashlib
import json
import struct
import zlib

import numpy as np

from compression.partition_bundle import unpack_bundle

HEADER = struct.Struct("<4sII32s")
ENTRY = struct.Struct("<QIBI")
OVERLAY = struct.Struct("<4sIII")
FAK = struct.Struct("<4sBBHIIIII32sII")
MAX_CELLS = 16 * 1024 * 1024


def inflate(data, limit):
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(data, limit + 1)
    except zlib.error as error:
        raise ValueError("Invalid compressed stream") from error
    if len(raw) > limit or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Invalid compressed termination/length")
    return raw


def literal(frame):
    """Validate the exact unbound FullTileVQWire.decode literal grammar."""
    from compression.adaptive_coefficient_stream import unpack_any
    arrays, metadata, side = unpack_any(frame)
    shape = metadata.get("shape")
    if (metadata.get("kind") != "literal-float32-v1" or side or len(arrays) != 1
            or arrays[0].shape != (1,) or not isinstance(shape, list) or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)
            or shape[0] * shape[1] > MAX_CELLS):
        raise ValueError("Invalid full-tile literal")
    return tuple(shape), int(arrays[0].view(np.uint32)[0])


def frame_shape(frame):
    if frame[:4] in (b"FRC1", b"FRC2"):
        return literal(frame)[0]
    if len(frame) < FAK.size:
        raise ValueError("Truncated FAK1")
    fields = list(FAK.unpack_from(frame))
    magic, version, method, k, h, w, lh, lw, ns, identity, npayload, crc = fields
    if (magic != b"FAK1" or version != 1 or method not in (0, 1)
            or k != 2048 or min(h, w, lh, lw) < 1 or h * w > MAX_CELLS
            or lh * lw > 1024 * 1024 or npayload < 4
            or FAK.size + ns + npayload != len(frame)):
        raise ValueError("Invalid FAK1 grammar")
    fields[-1] = 0
    if zlib.crc32(FAK.pack(*fields) + frame[FAK.size:]) != crc:
        raise ValueError("FAK1 CRC mismatch")
    payload = frame[FAK.size + ns + 4:]
    if method == 0:
        nbits = lh * lw * 11
        if len(payload) != (nbits + 7) // 8:
            raise ValueError("Invalid fixed symbols length")
        bits = np.unpackbits(np.frombuffer(payload, np.uint8))
        if bits[nbits:].any():
            raise ValueError("Invalid fixed symbols padding")
        ids = (bits[:nbits].reshape(-1, 11) @ (1 << np.arange(10, -1, -1))).astype("<u2")
        if zlib.crc32(ids.tobytes()) != struct.unpack_from("<I", frame, FAK.size + ns)[0]:
            raise ValueError("Fixed symbols CRC mismatch")
    elif not payload or len(payload) % 4:
        raise ValueError("Invalid range symbols length")
    return h, w


def overlay(data, shape):
    if len(data) < OVERLAY.size:
        raise ValueError("Truncated overlay")
    magic, size, count, crc = OVERLAY.unpack_from(data)
    body = data[OVERLAY.size:]
    cells = shape[0] * shape[1]
    if (magic != b"FZC1" or not 0 < size < len(body) or count > cells
            or zlib.crc32(body) != crc):
        raise ValueError("Invalid overlay")
    raw = inflate(body[size:], count * 8)
    if len(raw) != count * 8:
        raise ValueError("Invalid overlay count")
    positions = np.frombuffer(raw[:count * 4], "<u4")
    bits = np.frombuffer(raw[count * 4:], "<u4")
    if (np.any(positions >= cells) or np.any(positions[1:] <= positions[:-1])
            or np.any(bits & np.uint32(0x7fffffff))):
        raise ValueError("Invalid overlay positions/bits")
    return body[:size], positions, bits


def compact_tile(bundle):
    """Mode 0 is byte-identical fallback; mode 1 is a bare intact frame."""
    shape, fragments = unpack_bundle(bundle)
    if len(fragments) != 1 or fragments[0][:3] != (0, 0, shape[0]):
        return 0, bundle
    payload = fragments[0][3]
    if payload[:4] != b"FZC1":
        return 0, bundle
    frame, positions, bits = overlay(payload, shape)
    if frame[:4] not in (b"FAK1", b"FRC1", b"FRC2"):
        return 0, bundle
    if frame_shape(frame) != shape:
        raise ValueError("Inner/outer geometry mismatch")
    if not len(positions):
        return 1, frame
    if frame[:4] in (b"FRC1", b"FRC2") and np.all(bits == literal(frame)[1]):
        return 1, frame
    return 0, bundle


def pack(bundles, geometry):
    if not bundles or len(bundles) > 1_000_000:
        raise ValueError("Invalid tile count")
    chunks = [compact_tile(bundle) for bundle in bundles]
    cursor = HEADER.size + len(chunks) * ENTRY.size + len(geometry)
    index = bytearray()
    for mode, payload in chunks:
        index.extend(ENTRY.pack(cursor, len(payload), mode, zlib.crc32(payload)))
        cursor += len(payload)
    body = (HEADER.pack(b"FTC1", len(chunks), len(geometry), hashlib.sha256(index).digest())
            + index + geometry + b"".join(payload for _, payload in chunks))
    return body + hashlib.sha256(body).digest()


class Reader:
    """Index validation is independent of the packing/selection path."""
    def __init__(self, data):
        if len(data) < HEADER.size + 32 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
            raise ValueError("Object SHA256 mismatch")
        magic, count, ng, expected = HEADER.unpack_from(data)
        if magic != b"FTC1" or not 0 < count <= 1_000_000:
            raise ValueError("Invalid object header")
        end = HEADER.size + count * ENTRY.size
        if end + ng > len(data) - 32:
            raise ValueError("Truncated index/geometry")
        index = data[HEADER.size:end]
        if hashlib.sha256(index).digest() != expected:
            raise ValueError("Index SHA256 mismatch")
        self.entries = list(ENTRY.iter_unpack(index))
        cursor = end + ng
        for offset, size, mode, crc in self.entries:
            if offset != cursor or not size or mode not in (0, 1) or offset + size > len(data) - 32:
                raise ValueError("Invalid tile offsets/coverage")
            cursor += size
        if cursor != len(data) - 32:
            raise ValueError("Trailing object bytes")
        self.geometry = data[end:end + ng]
        self.data = data

    def tile(self, index):
        offset, size, mode, crc = self.entries[index]
        payload = self.data[offset:offset + size]
        if zlib.crc32(payload) != crc:
            raise ValueError("Tile CRC mismatch")
        shape = frame_shape(payload) if mode else unpack_bundle(payload)[0]
        return mode, shape, payload


def geometry_rows(geometry):
    rows = json.loads(inflate(geometry, 16 * 1024 * 1024))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Invalid geometry")
    return rows
