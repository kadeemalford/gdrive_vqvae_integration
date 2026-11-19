"""Paid two-core transport and lossless neural magnitude buckets for correction B."""
import hashlib
import io
import json
import struct
import zlib

import numpy as np

from compression.bounded_residual_stream import MAX_CELLS, decode as decode_bound
from compression.predictive_coefficient_stream import pack_predictive, unpack_predictive

PAIR = struct.Struct("<4sIIII")
MAX_BYTES = 128 * 1024 * 1024
MODES = ("none", "aligned", "shifted")
SHAPES = {"w0": (8, 32), "b0": (32,), "w1": (32, 32), "b1": (32,), "w2": (32, 1), "b2": (1,)}


def features(context, group_index, center, scale, mode):
    if (mode not in MODES or group_index not in (0, 1) or context.ndim != 2
            or not np.isfinite(context).all() or not np.isfinite([center, scale]).all() or scale <= 0):
        raise ValueError("Invalid decoded magnitude context")
    h, w = context.shape
    result = np.zeros((h, w, 8), np.float32)
    if mode != "none":
        values = (context.astype(np.float64) - center) / scale
        if mode == "shifted":
            values = np.roll(values, w // 2, axis=1)
        magnitude = np.log1p(np.abs(values))
        signed = np.sign(values) * magnitude
        square = values**2
        time_pad = np.pad(square, ((0, 0), (4, 4)), mode="edge")
        spatial_pad = np.pad(square, ((1, 1), (0, 0)), mode="edge")
        temporal = sum(time_pad[:, offset:offset + w] for offset in range(9)) / 9
        spatial = sum(spatial_pad[offset:offset + h] for offset in range(3)) / 3
        previous = np.pad(signed, ((0, 0), (1, 0)), mode="edge")[:, :w]
        following = np.pad(signed, ((0, 0), (0, 1)), mode="edge")[:, 1:]
        result[..., :6] = np.clip(np.stack(
            (signed, magnitude, np.log1p(temporal), np.log1p(spatial), previous, following), -1), -32, 32)
    result[..., 6 + group_index] = 1
    return result.reshape(-1, 8)


class MagnitudeModel:
    def __init__(self, weights, mode, groups, statistics, thresholds, identifier=None):
        self.weights, self.mode, self.groups = weights, mode, list(groups)
        self.statistics = np.asarray(statistics, np.float64)
        self.thresholds = np.asarray(thresholds, np.float32)
        self.identifier = identifier

    def setup_bytes(self):
        buffer = io.BytesIO()
        np.savez(buffer, **self.weights, mode=np.array(MODES.index(self.mode), np.uint8),
                 groups=np.array(self.groups), statistics=self.statistics, thresholds=self.thresholds)
        return buffer.getvalue()

    @classmethod
    def from_setup(cls, data):
        with np.load(io.BytesIO(data), allow_pickle=False) as saved:
            weights = {key: saved[key].copy() for key in SHAPES}
            mode, groups = saved["mode"], saved["groups"].tolist()
            stats, thresholds = saved["statistics"].copy(), saved["thresholds"].copy()
            if (any(v.dtype != np.float32 or v.shape != SHAPES[k] or not np.isfinite(v).all()
                    for k, v in weights.items()) or mode.shape != () or int(mode) not in range(3)
                    or not isinstance(groups, list) or len(groups) != 2 or len(set(groups)) != 2
                    or any(not isinstance(g, str) for g in groups)
                    or stats.dtype != np.float64 or stats.shape != (2, 3)
                    or not np.isfinite(stats).all() or np.any(stats[:, 1:] <= 0)
                    or thresholds.dtype != np.float32 or thresholds.shape != (2, 7) or not np.isfinite(thresholds).all()
                    or np.any(thresholds[:, 1:] < thresholds[:, :-1])):
                raise ValueError("Invalid magnitude model setup")
            return cls(weights, MODES[int(mode)], groups, stats, thresholds, hashlib.sha256(data).hexdigest())

    def score(self, values):
        weights = self.weights
        values = np.maximum(values @ weights["w0"] + weights["b0"], 0)
        values = np.maximum(values @ weights["w1"] + weights["b1"], 0)
        return (values @ weights["w2"] + weights["b2"])[:, 0]

    def buckets(self, context, group):
        if group not in self.groups:
            raise ValueError("Unknown magnitude group")
        index = self.groups.index(group)
        center, scale, _ = self.statistics[index]
        prediction = self.score(features(context, index, center, scale, self.mode))
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite magnitude score")
        return np.searchsorted(self.thresholds[index], prediction, side="right").reshape(context.shape)


def encode_buckets(original, context, group, model):
    arrays, descriptor, side = unpack_predictive(original)
    if len(arrays) != 1 or descriptor.get("kind") != "absolute-bounded-float32-v1":
        raise ValueError("Magnitude buckets require a bounded-correction frame")
    q, = arrays
    if q.shape != context.shape or not model.identifier:
        raise ValueError("Magnitude context shape or frozen setup missing")
    bucket = model.buckets(context, group)
    present = sorted(np.unique(bucket).tolist())
    values = [q[bucket == index].astype(np.int32) for index in present]
    metadata = {"kind": "neural-magnitude-buckets-v1", "original": descriptor, "group_id": group,
                "model_sha256": model.identifier, "present": present, "shape": list(q.shape),
                "q_sha256": hashlib.sha256(q.astype("<i4").tobytes()).hexdigest()}
    candidate = pack_predictive(values, metadata, side)
    return candidate if len(candidate) < len(original) else original


def restore_buckets(frame, context, group, model):
    arrays, metadata, side = unpack_predictive(frame)
    if metadata.get("kind") != "neural-magnitude-buckets-v1":
        return frame
    if (model is None or metadata.get("model_sha256") != model.identifier
            or metadata.get("group_id") != group or metadata.get("shape") != list(context.shape)):
        raise ValueError("Magnitude model, group or shape mismatch")
    bucket = model.buckets(context, group)
    present = sorted(np.unique(bucket).tolist())
    if metadata.get("present") != present or len(arrays) != len(present):
        raise ValueError("Invalid magnitude bucket map")
    q = np.zeros(context.shape, np.int32)
    for index, values in zip(present, arrays):
        mask = bucket == index
        if values.ndim != 1 or values.size != int(mask.sum()):
            raise ValueError("Magnitude bucket length mismatch")
        q[mask] = values
    if hashlib.sha256(q.astype("<i4").tobytes()).hexdigest() != metadata.get("q_sha256"):
        raise ValueError("Magnitude coefficient identity check failed")
    return pack_predictive([q], metadata["original"], side)


def pack_pair(left_frame, right_frame, metadata, model=None):
    shape = tuple(metadata["shape"])
    context = decode_bound(left_frame, np.zeros(shape, np.float32))
    if model is not None:
        right_frame = encode_buckets(right_frame, context, metadata["group_id"], model)
    descriptor = json.dumps(metadata, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    body = descriptor + left_frame + right_frame
    if len(body) > MAX_BYTES or len(descriptor) > 1024 * 1024:
        raise ValueError("Multicore frame allocation limit")
    return PAIR.pack(b"FMP1", len(descriptor), len(left_frame), len(right_frame), zlib.crc32(body)) + body


def unpack_pair(frame, model=None):
    if len(frame) < PAIR.size:
        raise ValueError("Truncated multicore pair")
    magic, nm, na, nb, checksum = PAIR.unpack_from(frame)
    body = frame[PAIR.size:]
    if (magic != b"FMP1" or nm > 1024 * 1024 or not na or not nb
            or nm + na + nb != len(body) or len(body) > MAX_BYTES or zlib.crc32(body) != checksum):
        raise ValueError("Multicore pair checksum or length mismatch")
    metadata = json.loads(body[:nm])
    shape = metadata.get("shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape) or shape[0] * shape[1] > MAX_CELLS):
        raise ValueError("Invalid multicore shape")
    left = decode_bound(body[nm:nm + na], np.zeros(shape, np.float32))
    right_frame = restore_buckets(body[nm + na:], left, metadata["group_id"], model)
    right = decode_bound(right_frame, np.zeros(shape, np.float32))
    return left, right, metadata, right_frame
