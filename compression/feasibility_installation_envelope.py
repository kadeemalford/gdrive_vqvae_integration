"""Exact-byte installation envelope; Python 3 stdlib with lzma is preinstalled."""
import hashlib
import hmac
import lzma
import struct


HEADER = struct.Struct("<4sBQQ32s")
MAGIC = b"FIE1"
TRAILER_BYTES = 32
OVERHEAD = HEADER.size + TRAILER_BYTES
PRESET = 6


def pack(original):
    """Try exactly one XZ/LZMA2 preset; raw wins ties and expansion."""
    compressed = lzma.compress(original, format=lzma.FORMAT_XZ,
                               check=lzma.CHECK_CRC64, preset=PRESET)
    flag = int(len(compressed) < len(original))
    payload = compressed if flag else original
    header = HEADER.pack(MAGIC, flag, len(original), len(payload),
                         hashlib.sha256(original).digest())
    body = header + payload
    return body + hashlib.sha256(body).digest()


def unpack(envelope):
    """Reject corruption, trailing bytes, unknown modes and incomplete XZ."""
    if len(envelope) < OVERHEAD:
        raise ValueError("Truncated installation envelope")
    body, checksum = envelope[:-TRAILER_BYTES], envelope[-TRAILER_BYTES:]
    if not hmac.compare_digest(hashlib.sha256(body).digest(), checksum):
        raise ValueError("Installation envelope integrity mismatch")
    magic, flag, original_size, payload_size, digest = HEADER.unpack_from(body)
    if magic != MAGIC or flag not in (0, 1):
        raise ValueError("Unknown installation envelope format")
    payload = body[HEADER.size:]
    if len(payload) != payload_size:
        raise ValueError("Installation envelope payload length mismatch")
    if flag:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
        try:
            original = decoder.decompress(payload, max_length=original_size + 1)
        except lzma.LZMAError as exc:
            raise ValueError("Invalid installation XZ payload") from exc
        if not decoder.eof or decoder.unused_data:
            raise ValueError("Incomplete or trailing installation XZ payload")
    else:
        original = payload
    if (len(original) != original_size
            or not hmac.compare_digest(hashlib.sha256(original).digest(), digest)):
        raise ValueError("Original installation integrity mismatch")
    return original
