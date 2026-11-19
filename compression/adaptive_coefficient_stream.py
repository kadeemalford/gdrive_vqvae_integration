"""FRC2: per-band adaptive categorical range coding with fully charged tables.

No probability is learned from untransmitted data: exact symbol counts are in
each frame. Constant and zlib-plane fallbacks compete by complete band bytes.
FRC1 and all historical frames remain unchanged.
"""
import json
import struct
import zlib

import constriction
import numpy as np

from compression.research_coefficient_stream import HEADER, MAX_BYTES

MAGIC = b"FRC2"
BAND = struct.Struct("<B3xIIII")
MAX_ALPHABET = 65536


def inflate(data, expected):
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(data, expected + 1)
    except zlib.error as error:
        raise ValueError("Corrupt coefficient compression") from error
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Invalid coefficient decompression length")
    return raw


def distribution(counts):
    probabilities = counts.astype(np.float64) / counts.sum(dtype=np.float64)
    return constriction.stream.model.Categorical(probabilities.astype(np.float32), perfect=False)


def encode_band(array):
    flat = array.astype("<i4", copy=False).ravel()
    raw = flat.tobytes()
    checksum = zlib.crc32(raw)
    planes = flat.view(np.uint8).reshape(-1, 4).T.copy().tobytes()
    payload = zlib.compress(planes, 6)
    candidates = [(BAND.pack(0, 0, 0, len(payload), checksum) + payload, 0)]
    alphabet, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    if len(alphabet) == 1:
        candidates.append((BAND.pack(1, 1, 0, 4, checksum) + alphabet.astype("<i4").tobytes(), 1))
    elif len(alphabet) <= MAX_ALPHABET:
        table = zlib.compress(alphabet.astype("<i4").tobytes() + counts.astype("<u4").tobytes(), 6)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(inverse.astype(np.int32), distribution(counts))
        payload = encoder.get_compressed().astype("<u4").tobytes()
        candidates.append((BAND.pack(2, len(alphabet), len(table), len(payload), checksum) + table + payload, 2))
    frame, method = min(candidates, key=lambda item: (len(item[0]), item[1]))
    return frame, method


def pack(arrays, descriptor, side, return_details=False):
    if not arrays or len(arrays) > 64 or not isinstance(side, bytes):
        raise ValueError("Expected 1..64 arrays and byte side stream")
    shapes, chunks, methods = [], [], []
    cells = 0
    for array in arrays:
        array = np.asarray(array)
        if array.dtype != np.int32 or not array.size or array.ndim not in range(1, 5):
            raise ValueError("Nonempty 1D..4D int32 coefficients required")
        cells += array.size
        if cells * 4 > MAX_BYTES:
            raise ValueError("Coefficient allocation limit")
        shapes.append(list(array.shape))
        chunk, method = encode_band(array)
        chunks.append(chunk)
        methods.append(method)
    metadata = json.dumps({"shapes": shapes, "codec": descriptor},
                          sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if len(metadata) + len(side) > MAX_BYTES:
        raise ValueError("Metadata allocation limit")
    payload = b"".join(chunks)
    body = metadata + side + payload
    frame = HEADER.pack(MAGIC, len(metadata), len(side), len(payload), zlib.crc32(body)) + body
    if return_details:
        return frame, {"methods": methods, "band_bytes": list(map(len, chunks)),
                       "metadata_bytes": len(metadata), "side_bytes": len(side),
                       "header_bytes": HEADER.size}
    return frame


def unpack(frame):
    if len(frame) < HEADER.size:
        raise ValueError("Truncated adaptive coefficient frame")
    magic, nm, ns, npayload, crc = HEADER.unpack_from(frame)
    body = frame[HEADER.size:]
    if magic != MAGIC or nm + ns + npayload != len(body) or nm + ns > MAX_BYTES:
        raise ValueError("Invalid adaptive coefficient header")
    if zlib.crc32(body) != crc:
        raise ValueError("Adaptive coefficient checksum mismatch")
    try:
        metadata = json.loads(body[:nm])
        shapes = metadata["shapes"]
        descriptor = metadata["codec"]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("Invalid coefficient metadata") from error
    if not isinstance(shapes, list) or not 1 <= len(shapes) <= 64:
        raise ValueError("Invalid coefficient band count")
    sizes = []
    for shape in shapes:
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 4
                or any(type(n) is not int or n < 1 for n in shape)):
            raise ValueError("Invalid adaptive coefficient shape")
        size = 1
        for n in shape:
            size *= n
        sizes.append(size)
    if sum(sizes) * 4 > MAX_BYTES:
        raise ValueError("Coefficient allocation limit")
    offset = nm + ns
    arrays = []
    for shape, size in zip(shapes, sizes):
        if offset + BAND.size > len(body):
            raise ValueError("Truncated coefficient band")
        method, alphabet_size, nt, npayload, checksum = BAND.unpack_from(body, offset)
        offset += BAND.size
        if nt + npayload > len(body) - offset:
            raise ValueError("Invalid coefficient band lengths")
        table = body[offset:offset + nt]
        payload = body[offset + nt:offset + nt + npayload]
        offset += nt + npayload
        if method == 0 and alphabet_size == nt == 0:
            planes = np.frombuffer(inflate(payload, size * 4), np.uint8)
            flat = planes.reshape(4, -1).T.copy().view("<i4").ravel()
        elif method == 1 and alphabet_size == 1 and nt == 0 and npayload == 4:
            flat = np.full(size, np.frombuffer(payload, "<i4")[0], dtype="<i4")
        elif method == 2 and 2 <= alphabet_size <= min(MAX_ALPHABET, size) and npayload >= 4 and npayload % 4 == 0:
            raw_table = inflate(table, alphabet_size * 8)
            alphabet = np.frombuffer(raw_table, "<i4", count=alphabet_size)
            counts = np.frombuffer(raw_table, "<u4", count=alphabet_size, offset=alphabet_size * 4)
            if (np.any(counts == 0) or int(counts.sum(dtype=np.uint64)) != size
                    or np.any(alphabet[1:] <= alphabet[:-1])):
                raise ValueError("Invalid transmitted coefficient table")
            decoder = constriction.stream.queue.RangeDecoder(
                np.frombuffer(payload, "<u4").astype(np.uint32, copy=True))
            try:
                symbols = decoder.decode(distribution(counts), size)
            except Exception as error:
                raise ValueError("Invalid coefficient range stream") from error
            flat = alphabet[symbols].astype("<i4", copy=True)
            if not np.array_equal(np.bincount(symbols, minlength=alphabet_size), counts):
                raise ValueError("Decoded coefficient counts differ")
        else:
            raise ValueError("Invalid adaptive coefficient method")
        if zlib.crc32(flat.tobytes()) != checksum:
            raise ValueError("Decoded coefficient checksum mismatch")
        arrays.append(flat.reshape(shape))
    if offset != len(body):
        raise ValueError("Trailing coefficient payload")
    return arrays, descriptor, body[nm:nm + ns]


def pack_best(arrays, descriptor, side):
    from compression.research_coefficient_stream import pack as pack_legacy
    adaptive = pack(arrays, descriptor, side)
    legacy = pack_legacy(arrays, descriptor, side)
    return min((legacy, adaptive), key=lambda frame: (len(frame), frame[:4]))


def unpack_any(frame):
    from compression.research_coefficient_stream import unpack as unpack_legacy
    if frame[:4] == b"FRC1":
        return unpack_legacy(frame)
    return unpack(frame)
