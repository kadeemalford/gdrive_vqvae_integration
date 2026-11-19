"""FRC3: reversible integer prediction of coefficient bands, no free context."""
import struct
import zlib

import numpy as np

from compression.adaptive_coefficient_stream import encode_band, pack, pack_best, unpack, unpack_any
from compression.research_coefficient_stream import HEADER, MAX_BYTES

MAGIC = b"FRC3"


def difference(values, axis):
    result = values.copy()
    current, previous = [slice(None)] * values.ndim, [slice(None)] * values.ndim
    current[axis], previous[axis] = slice(1, None), slice(None, -1)
    result[tuple(current)] -= values[tuple(previous)]
    return result


def prediction_residual(array, method):
    result = array.astype(np.int64)
    if method in (1, 3):
        result = difference(result, -1)
    if method in (2, 3):
        if result.ndim < 2:
            raise ValueError("Spatial prediction requires at least two dimensions")
        result = difference(result, -2)
    if np.any(result < np.iinfo(np.int32).min) or np.any(result > np.iinfo(np.int32).max):
        raise ValueError("Predictive coefficient overflow")
    return result.astype(np.int32)


def restore_residual(array, method):
    result = array.astype(np.int64)
    if method in (2, 3):
        if result.ndim < 2:
            raise ValueError("Invalid spatial residual shape")
        result = result.cumsum(axis=-2, dtype=np.int64)
    if method in (1, 3):
        result = result.cumsum(axis=-1, dtype=np.int64)
    if np.any(result < np.iinfo(np.int32).min) or np.any(result > np.iinfo(np.int32).max):
        raise ValueError("Restored coefficient overflow")
    return result.astype(np.int32)


def pack_predictive(arrays, descriptor, side):
    transformed, methods = [], []
    for array in arrays:
        candidates = []
        for method in range(4 if array.ndim >= 2 else 2):
            try:
                residual = prediction_residual(array, method)
            except ValueError:
                continue
            band, _ = encode_band(residual)
            candidates.append((len(band), method, residual))
        _, method, residual = min(candidates, key=lambda candidate: candidate[:2])
        methods.append(method)
        transformed.append(residual)
    metadata = {"kind": "integer-coefficient-prediction-v1", "original_codec": descriptor,
                "predictors": methods, "original_crc32": [
                    zlib.crc32(a.astype("<i4").tobytes()) for a in arrays]}
    inner = pack(transformed, metadata, side)
    predicted = HEADER.pack(MAGIC, 0, 0, len(inner), zlib.crc32(inner)) + inner
    fallback = pack_best(arrays, descriptor, side)
    return min((fallback, predicted), key=lambda frame: (len(frame), frame[:4]))


def unpack_predictive(frame):
    if frame[:4] != MAGIC:
        return unpack_any(frame)
    if len(frame) < HEADER.size:
        raise ValueError("Truncated predictive coefficients")
    magic, nm, ns, size, checksum = HEADER.unpack_from(frame)
    inner = frame[HEADER.size:]
    if (magic != MAGIC or nm or ns or size != len(inner) or size > 2 * MAX_BYTES
            or inner[:4] != b"FRC2" or zlib.crc32(inner) != checksum):
        raise ValueError("Invalid predictive coefficient wrapper")
    arrays, metadata, side = unpack(inner)
    methods, checksums = metadata.get("predictors"), metadata.get("original_crc32")
    if (metadata.get("kind") != "integer-coefficient-prediction-v1"
            or not isinstance(methods, list) or not isinstance(checksums, list)
            or len(methods) != len(arrays) or len(checksums) != len(arrays)
            or any(type(method) is not int or method not in range(4) for method in methods)):
        raise ValueError("Invalid predictor metadata")
    restored = [restore_residual(array, method) for array, method in zip(arrays, methods)]
    for array, expected in zip(restored, checksums):
        if zlib.crc32(array.astype("<i4").tobytes()) != expected:
            raise ValueError("Predictive coefficient identity check failed")
    return restored, metadata["original_codec"], side
