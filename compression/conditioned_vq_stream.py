"""Charge the exact FiLM condition and bind it to the full-tile neural frame."""
import struct
import zlib

import numpy as np

from compression.full_tile_vq_stream import FullTileVQWire
from finetuning.physical_conditioning import condition, set_condition

HEADER = struct.Struct("<4sBBII")
METHODS = ("constant", "hardware", "category")
MAX_BYTES = 128 * 1024 * 1024


def pack_context(frame, method, vector, category=8, missing=False):
    if method not in METHODS or not isinstance(frame, bytes) or not frame or len(frame) > MAX_BYTES:
        raise ValueError("Invalid conditioned inner frame")
    vector = np.asarray(vector, np.float32)
    if vector.shape != (11,) or not np.isfinite(vector).all():
        raise ValueError("Invalid conditioned vector")
    if method == "constant":
        missing = False
        payload = b""
        expected = np.zeros(11, np.float32)
    elif missing:
        payload = b""
        expected = np.zeros(11, np.float32)
        expected[-1] = 1
    elif method == "hardware":
        payload = vector[:10].astype("<f4").tobytes()
        expected = np.r_[vector[:10], np.float32(0)].astype(np.float32)
    else:
        if type(category) is not int or not 0 <= category <= 8:
            raise ValueError("Invalid categorical ID")
        payload = bytes([category])
        expected = np.zeros(11, np.float32)
        if category < 8:
            expected[category] = 1
    if not np.array_equal(vector.view(np.uint32), expected.view(np.uint32)):
        raise ValueError("Condition cannot be derived exactly from charged payload")
    body = payload + frame
    code = METHODS.index(method)
    checksum = zlib.crc32(bytes([code, int(missing)]) + body)
    return HEADER.pack(b"FCM1", code, int(missing), len(body), checksum) + body


def unpack_context(frame, method):
    if len(frame) < HEADER.size or method not in METHODS:
        raise ValueError("Invalid conditioned wrapper")
    magic, code, missing, size, checksum = HEADER.unpack_from(frame)
    body = frame[HEADER.size:]
    if (magic != b"FCM1" or code != METHODS.index(method) or missing not in (0, 1)
            or (method == "constant" and missing) or size != len(body) or size > MAX_BYTES + 40
            or zlib.crc32(bytes([code, missing]) + body) != checksum):
        raise ValueError("Conditioned wrapper checksum or mode mismatch")
    vector = np.zeros(11, np.float32)
    if missing:
        vector[-1] = 1
        count = 0
    elif method == "hardware":
        count = 40
        if len(body) < count:
            raise ValueError("Truncated hardware vector")
        vector[:10] = np.frombuffer(body[:count], "<f4")
    elif method == "category":
        count = 1
        if not body or body[0] > 8:
            raise ValueError("Invalid categorical condition")
        if body[0] < 8:
            vector[body[0]] = 1
    else:
        count = 0
    if not np.isfinite(vector).all() or len(body) <= count:
        raise ValueError("Invalid condition or empty neural frame")
    return vector, body[count:]


class ConditionedVQWire:
    def __init__(self, model, inner, config):
        self.model, self.inner, self.config = model, inner, config

    @classmethod
    def create(cls, model, counts, identifier, config):
        return cls(model, FullTileVQWire(model, counts, identifier), config)

    @classmethod
    def from_setup(cls, model, data, config):
        return cls(model, FullTileVQWire.from_setup(model, data), config)

    def setup_bytes(self):
        return self.inner.setup_bytes()

    def encode(self, raw, group, missing=False):
        method, plan = self.config["method"], self.config["plan"]
        vector = condition(plan, method, group, missing)
        set_condition(self.model, vector)
        inner = self.inner.encode(raw)
        category = plan["seen"].index(group) if group in plan["seen"] else 8
        return pack_context(inner, method, vector, category, missing)

    def decode(self, frame):
        vector, inner = unpack_context(frame, self.config["method"])
        set_condition(self.model, vector)
        return self.inner.decode(inner)
