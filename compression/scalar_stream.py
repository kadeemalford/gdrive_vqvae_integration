"""CRC-framed scalar range stream with explicit int32 tail escapes; no clipping."""
import hashlib
import json
import struct
import zlib

import constriction
import numpy as np
from scipy.special import expit

MAGIC = b'SRD1'
LIMIT = 255
ESCAPE = 511


class ScalarStream:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        if (self.probabilities.ndim != 2 or self.probabilities.shape[1] != 512 or
                not np.isfinite(self.probabilities).all() or np.any(self.probabilities <= 0)):
            raise ValueError('Invalid scalar probability table')
        self.serialized = self.probabilities.astype('<f4').tobytes()
        self.identifier = hashlib.sha256(self.serialized).hexdigest()
        self.models = [constriction.stream.model.Categorical(p, perfect=False) for p in self.probabilities]

    @classmethod
    def from_scales(cls, scales):
        values = np.arange(-LIMIT, LIMIT + 1, dtype=np.float64)
        probabilities = []
        for scale in scales:
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid logistic scale')
            p = expit((values + .5) / scale) - expit((values - .5) / scale)
            tail = 2 * expit(-(LIMIT + .5) / scale)
            table = np.maximum(np.r_[p, tail], 1e-12)
            probabilities.append(table / table.sum())
        return cls(probabilities)

    def encode(self, symbols, shape, side, channel_map_id):
        values = np.asarray(symbols)
        if (values.ndim != 3 or values.shape[0] != len(self.models) or
                not np.isfinite(values).all() or not np.equal(values, np.round(values)).all() or
                values.min() < np.iinfo(np.int32).min or values.max() > np.iinfo(np.int32).max):
            raise ValueError('Expected exact finite CHW int32-range scalar symbols')
        values = values.astype(np.int32)
        payloads, lengths, tails = [], [], []
        for plane, model in zip(values, self.models):
            flat = plane.reshape(-1)
            escaped = (flat < -LIMIT) | (flat > LIMIT)
            codes = np.where(escaped, ESCAPE, flat + LIMIT).astype(np.int32)
            encoder = constriction.stream.queue.RangeEncoder()
            encoder.encode(codes, model)
            tail = flat[escaped].astype('<i4').tobytes()
            payload = encoder.get_compressed().astype('<u4').tobytes()
            payloads.extend([tail, payload]); tails.append(len(tail)); lengths.append(len(payload))
        header = json.dumps({'shape': list(shape), 'latent': list(values.shape), 'side': len(side),
            'tail_bytes': tails, 'range_bytes': lengths, 'prior': self.identifier, 'channel_map_id': channel_map_id,
            'symbols_crc': zlib.crc32(values.astype('<i4').tobytes())}, sort_keys=True, separators=(',', ':')).encode()
        frame = MAGIC + struct.pack('<I', len(header)) + header + side + b''.join(payloads)
        return frame + struct.pack('<I', zlib.crc32(frame))

    def decode(self, frame):
        if len(frame) < 12 or frame[:4] != MAGIC or zlib.crc32(frame[:-4]) != struct.unpack('<I', frame[-4:])[0]:
            raise ValueError('Invalid scalar frame or CRC')
        header_size = struct.unpack_from('<I', frame, 4)[0]
        if header_size > min(65536, len(frame) - 12):
            raise ValueError('Invalid scalar header length')
        header = json.loads(frame[8:8 + header_size])
        channels, height, width = header['latent']
        if (header['prior'] != self.identifier or channels != len(self.models) or
                min(height, width, *header['shape']) < 1 or height * width * channels > 10_000_000 or
                len(header['tail_bytes']) != channels or len(header['range_bytes']) != channels):
            raise ValueError('Incompatible scalar frame')
        offset = 8 + header_size
        side_size = header['side']
        if side_size < 0 or offset + side_size > len(frame) - 4:
            raise ValueError('Invalid scalar side length')
        side = frame[offset:offset + side_size]; offset += side_size
        planes = []
        for model, tail_size, size in zip(self.models, header['tail_bytes'], header['range_bytes']):
            if min(tail_size, size) < 0 or tail_size % 4 or size % 4 or offset + tail_size + size > len(frame) - 4:
                raise ValueError('Invalid scalar payload length')
            tail = np.frombuffer(frame[offset:offset + tail_size], dtype='<i4')
            offset += tail_size
            words = np.frombuffer(frame[offset:offset + size], dtype='<u4').astype(np.uint32, copy=True)
            offset += size
            decoder = constriction.stream.queue.RangeDecoder(words)
            codes = decoder.decode(model, height * width)
            escaped = codes == ESCAPE
            if int(escaped.sum()) != len(tail):
                raise ValueError('Scalar escape count mismatch')
            plane = codes.astype(np.int32) - LIMIT
            plane[escaped] = tail
            planes.append(plane.reshape(height, width))
        values = np.stack(planes)
        if offset != len(frame) - 4 or zlib.crc32(values.astype('<i4').tobytes()) != header['symbols_crc']:
            raise ValueError('Scalar symbol CRC or trailing bytes mismatch')
        return values, tuple(header['shape']), side, header['channel_map_id']
