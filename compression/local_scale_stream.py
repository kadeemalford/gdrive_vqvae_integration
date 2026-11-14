"""Bounded diagnostic scalar stream: common framing, optional transmitted scale IDs."""
import hashlib
import json
import struct
import zlib

import constriction
import numpy as np

from compression.scalar_stream import ScalarStream, LIMIT, ESCAPE

MAGIC = b'LSS1'
TILE = 8
SCALE_BANK = np.geomspace(.05, 256., 64)


def symbol_codes(values):
    values = np.asarray(values)
    if (values.ndim != 3 or not values.size or values.size > 10_000_000
            or not np.isfinite(values).all() or not np.equal(values, np.round(values)).all()
            or values.min() < np.iinfo(np.int32).min or values.max() > np.iinfo(np.int32).max):
        raise ValueError('Finite exact CHW int32-range symbols required')
    values = values.astype(np.int32)
    escaped = (values < -LIMIT) | (values > LIMIT)
    return values, np.where(escaped, ESCAPE, values + LIMIT).astype(np.int32)


class LocalScaleStream:
    def __init__(self, probabilities, channels, adaptive=False):
        probabilities = np.asarray(probabilities, dtype=np.float32)
        required = channels + (64 if adaptive else 0)
        if channels < 1 or probabilities.shape != (required, 512):
            raise ValueError('Expected channel priors plus the fixed 64-scale bank when adaptive')
        prior = ScalarStream(probabilities)
        self.probabilities = prior.probabilities
        self.models = prior.models
        self.serialized = prior.serialized
        self.identifier = hashlib.sha256(self.serialized).hexdigest()
        self.channels, self.adaptive = channels, bool(adaptive)
        self.costs = -np.log2(self.probabilities.astype(np.float64))

    @classmethod
    def from_empirical_counts(cls, counts, adaptive=False):
        counts = np.asarray(counts, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[1] != 512 or not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError('Finite nonnegative channel counts required')
        probabilities = counts + 1.
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if adaptive:
            probabilities = np.concatenate([probabilities, ScalarStream.from_scales(SCALE_BANK).probabilities])
        return cls(probabilities, len(counts), adaptive)

    def encode(self, symbols, shape, side):
        values, codes = symbol_codes(symbols)
        channels, height, width = values.shape
        if channels != self.channels or len(shape) != 2 or any(int(v) != v or v < 1 for v in shape):
            raise ValueError('Invalid original or latent shape')
        encoder = constriction.stream.queue.RangeEncoder()
        choices, tails = [], []
        for channel in range(channels):
            for y in range(0, height, TILE):
                for x in range(0, width, TILE):
                    block = codes[channel, y:y + TILE, x:x + TILE].reshape(-1)
                    table = channel
                    if self.adaptive:
                        candidates = [channel, *range(channels, channels + 64)]
                        costs = self.costs[candidates][:, block].sum(axis=1)
                        choice = int(np.argmin(costs))
                        choices.append(choice)
                        table = candidates[choice]
                    encoder.encode(block, self.models[table])
                    original = values[channel, y:y + TILE, x:x + TILE].reshape(-1)
                    tails.append(original[block == ESCAPE].astype('<i4').tobytes())
        words = encoder.get_compressed().astype('<u4').tobytes()
        tail = b''.join(tails)
        header = json.dumps({'shape': list(map(int, shape)), 'latent': list(values.shape),
            'side': len(side), 'choices': len(choices), 'tail': len(tail), 'range': len(words),
            'adaptive': self.adaptive, 'prior': self.identifier,
            'symbols_crc': zlib.crc32(values.astype('<i4').tobytes())},
            sort_keys=True, separators=(',', ':')).encode()
        payload = MAGIC + struct.pack('<I', len(header)) + header + side + bytes(choices) + tail + words
        return payload + struct.pack('<I', zlib.crc32(payload))

    def decode(self, frame):
        if (len(frame) < 12 or frame[:4] != MAGIC
                or zlib.crc32(frame[:-4]) != struct.unpack('<I', frame[-4:])[0]):
            raise ValueError('Invalid local-scale frame or CRC')
        size = struct.unpack_from('<I', frame, 4)[0]
        if size > min(65536, len(frame) - 12):
            raise ValueError('Invalid local-scale header length')
        header = json.loads(frame[8:8 + size])
        latent, shape = header['latent'], header['shape']
        dimensions = latent + shape
        if (len(latent) != 3 or len(shape) != 2
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in dimensions)
                or latent[0] != self.channels or np.prod(latent, dtype=object) > 10_000_000
                or header['prior'] != self.identifier or header['adaptive'] != self.adaptive):
            raise ValueError('Incompatible local-scale geometry/prior')
        channels, height, width = latent
        groups = channels * ((height + TILE - 1) // TILE) * ((width + TILE - 1) // TILE)
        side_n, choices_n, tail_n, range_n = (header[key] for key in ('side', 'choices', 'tail', 'range'))
        if (any(not isinstance(v, int) or v < 0 for v in (side_n, choices_n, tail_n, range_n))
                or choices_n != (groups if self.adaptive else 0) or tail_n % 4 or range_n % 4
                or 8 + size + side_n + choices_n + tail_n + range_n + 4 != len(frame)):
            raise ValueError('Invalid local-scale payload lengths')
        offset = 8 + size
        side = frame[offset:offset + side_n]; offset += side_n
        choices = frame[offset:offset + choices_n]; offset += choices_n
        if any(choice > 64 for choice in choices):
            raise ValueError('Invalid transmitted scale index')
        tails = np.frombuffer(frame[offset:offset + tail_n], dtype='<i4'); offset += tail_n
        words = np.frombuffer(frame[offset:offset + range_n], dtype='<u4').astype(np.uint32, copy=True)
        decoder = constriction.stream.queue.RangeDecoder(words)
        values = np.empty(latent, dtype=np.int32)
        group, used = 0, 0
        for channel in range(channels):
            for y in range(0, height, TILE):
                for x in range(0, width, TILE):
                    block_shape = (min(TILE, height - y), min(TILE, width - x))
                    choice = choices[group] if self.adaptive else 0
                    table = channel if choice == 0 else channels + choice - 1
                    codes = decoder.decode(self.models[table], block_shape[0] * block_shape[1])
                    escaped = codes == ESCAPE; count = int(escaped.sum())
                    if used + count > len(tails):
                        raise ValueError('Truncated scalar tail escapes')
                    block = codes.astype(np.int32) - LIMIT
                    block[escaped] = tails[used:used + count]; used += count
                    values[channel, y:y + TILE, x:x + TILE] = block.reshape(block_shape)
                    group += 1
        if used != len(tails) or zlib.crc32(values.astype('<i4').tobytes()) != header['symbols_crc']:
            raise ValueError('Local-scale symbol CRC or tail count mismatch')
        return values, tuple(shape), side
