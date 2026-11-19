"""Checksummed, bounded integer-plane frames for prospective research codecs.

Zlib is the initial real coder, not a claim of optimal entropy coding.
All reconstruction parameters and the normalization side are inside the frame.
Shared neural weights are identified by hash and accounted separately.
"""
import json
import struct
import zlib

import numpy as np

HEADER = struct.Struct("<4sIIII")
MAGIC = b"FRC1"
MAX_BYTES = 64 * 1024 * 1024


def pack(arrays, descriptor, side):
    shapes = []
    chunks = []
    for array in arrays:
        array = np.asarray(array)
        if array.dtype != np.int32 or not array.size:
            raise ValueError("Nonempty int32 coefficients required")
        shapes.append(list(array.shape))
        chunks.append(array.astype("<i4").view(np.uint8).reshape(-1, 4).T.copy().tobytes())
    metadata = json.dumps({"shapes": shapes, "codec": descriptor},
                          sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    raw = b"".join(chunks)
    if len(raw) > MAX_BYTES or len(metadata) + len(side) > MAX_BYTES:
        raise ValueError("Frame exceeds allocation limit")
    payload = zlib.compress(raw, 6)
    body = metadata + side + payload
    return HEADER.pack(MAGIC, len(metadata), len(side), len(payload), zlib.crc32(body)) + body


def unpack(frame):
    if len(frame) < HEADER.size:
        raise ValueError("Truncated coefficient frame")
    magic, nm, ns, npayload, crc = HEADER.unpack_from(frame)
    body = frame[HEADER.size:]
    if magic != MAGIC or nm + ns + npayload != len(body) or nm + ns > MAX_BYTES:
        raise ValueError("Invalid coefficient frame")
    if zlib.crc32(body) != crc:
        raise ValueError("Coefficient frame checksum mismatch")
    metadata = json.loads(body[:nm])
    shapes = metadata["shapes"]
    size = 0
    for shape in shapes:
        if not shape or len(shape) > 4 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("Invalid coefficient shape")
        cells = 1
        for n in shape:
            cells *= n
        size += cells * 4
    if size > MAX_BYTES:
        raise ValueError("Coefficient allocation limit")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(body[nm + ns:], size + 1)
    except zlib.error as error:
        raise ValueError("Invalid compressed coefficients") from error
    if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Invalid coefficient payload length")
    offset = 0
    arrays = []
    for shape in shapes:
        count = int(np.prod(shape))
        planes = np.frombuffer(raw, np.uint8, count=count * 4, offset=offset)
        arrays.append(planes.reshape(4, -1).T.copy().view("<i4").reshape(shape))
        offset += count * 4
    return arrays, metadata["codec"], body[nm:nm + ns]


def quantize(values, step):
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Positive quantization step required")
    symbols = np.rint(np.asarray(values, np.float64) / step)
    if not np.isfinite(symbols).all() or np.abs(symbols).max(initial=0) >= 2 ** 30:
        raise ValueError("Coefficient integer overflow")
    return symbols.astype(np.int32)
