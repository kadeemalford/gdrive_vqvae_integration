"""Lossless canonical Huffman/fixed-width framing for a frozen neural codec.

Probability tables are fitted on training symbols only. This module does not
change quantization, waveform values, or the neural representation.
"""
import hashlib
import heapq
import json
import struct
import zlib

import numpy as np

HEADER = struct.Struct('<4sBBBBIIIII32sI')
MAGIC = b'FSC1'
K = 2048


def code_lengths(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (K,) or np.any(counts < 0):
        raise ValueError('Expected 2048 nonnegative symbol counts')
    heap = [(int(n) + 1, i, i) for i, n in enumerate(counts)]
    heapq.heapify(heap)
    serial = K
    while len(heap) > 1:
        a, _, left = heapq.heappop(heap)
        b, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (a + b, serial, (left, right)))
        serial += 1
    lengths = np.empty(K, dtype=np.uint8)
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, int):
            lengths[node] = depth
        else:
            stack.extend((child, depth + 1) for child in node)
    return lengths


class BitsOut:
    def __init__(self):
        self.data = bytearray()
        self.buffer = self.size = 0

    def write(self, code, size):
        self.buffer = (self.buffer << size) | code
        self.size += size
        while self.size >= 8:
            self.size -= 8
            self.data.append((self.buffer >> self.size) & 255)
        self.buffer &= (1 << self.size) - 1

    def finish(self):
        if self.size:
            self.data.append(self.buffer << (8 - self.size))
        return bytes(self.data)


class BitsIn:
    def __init__(self, data):
        self.data = data
        self.offset = self.buffer = self.size = self.consumed = 0

    def peek(self, size):
        while self.size < size and self.offset < len(self.data):
            self.buffer = (self.buffer << 8) | self.data[self.offset]
            self.offset += 1
            self.size += 8
        if self.size < size:
            return self.buffer << (size - self.size)
        return self.buffer >> (self.size - size)

    def read(self, size):
        value = self.peek(size)
        if self.size < size:
            raise ValueError('Truncated symbol payload')
        self.size -= size
        self.consumed += size
        self.buffer &= (1 << self.size) - 1
        return value

    def validate_end(self):
        remaining = len(self.data) * 8 - self.consumed
        if remaining >= 8 or (remaining and self.read(remaining)):
            raise ValueError('Noncanonical trailing symbol data')


class CanonicalTable:
    PREFIX = 10

    def __init__(self, lengths):
        self.lengths = [int(n) for n in lengths]
        if len(self.lengths) != K or min(self.lengths) < 1 or max(self.lengths) > 32:
            raise ValueError('Invalid code lengths')
        self.max_length = max(self.lengths)
        self.codes = [0] * K
        self.lookup = [None] * (1 << self.PREFIX)
        self.slow = {}
        value = previous = 0
        for length, symbol in sorted((n, i) for i, n in enumerate(self.lengths)):
            value <<= length - previous
            if value >= 1 << length:
                raise ValueError('Oversubscribed Huffman code')
            self.codes[symbol] = value
            if length <= self.PREFIX:
                start = value << (self.PREFIX - length)
                for index in range(start, start + (1 << (self.PREFIX - length))):
                    self.lookup[index] = (symbol, length)
            else:
                self.slow[length, value] = symbol
            value += 1
            previous = length
        if value != 1 << previous:
            raise ValueError('Incomplete Huffman code')

    def decode(self, reader):
        found = self.lookup[reader.peek(self.PREFIX)]
        if found is not None:
            symbol, length = found
            reader.read(length)
            return symbol
        value = reader.read(self.PREFIX)
        for length in range(self.PREFIX + 1, self.max_length + 1):
            value = (value << 1) | reader.read(1)
            found = self.slow.get((length, value))
            if found is not None:
                return found
        raise ValueError('Invalid Huffman prefix')


class SymbolCodec:
    """Select the shortest actual payload among fixed, marginal and context codes."""
    def __init__(self, marginal, conditional, checkpoint_sha256):
        self.marginal = CanonicalTable(marginal)
        if len(conditional) != 17:
            raise ValueError('Expected 17 previous-symbol bucket contexts')
        self.conditional = [CanonicalTable(row) for row in conditional]
        self.spec = {'checkpoint_sha256': checkpoint_sha256, 'K': K,
                     'marginal_lengths': list(map(int, marginal)),
                     'conditional_lengths': [list(map(int, row)) for row in conditional],
                     'context': 'previous_symbol_div_128; start_context_16; reset_each_window'}
        self.serialized = json.dumps(self.spec, separators=(',', ':'), sort_keys=True).encode()
        self.identifier = hashlib.sha256(self.serialized).digest()

    @classmethod
    def from_counts(cls, counts, conditional, checkpoint_sha256):
        return cls(code_lengths(counts), [code_lengths(row) for row in conditional], checkpoint_sha256)

    @classmethod
    def from_bytes(cls, data):
        spec = json.loads(data)
        result = cls(spec['marginal_lengths'], spec['conditional_lengths'], spec['checkpoint_sha256'])
        if result.serialized != data:
            raise ValueError('Noncanonical model table serialization')
        return result

    def encode(self, symbols, shape, latent_shape, side):
        symbols = np.asarray(symbols).reshape(-1)
        if (min(*shape, *latent_shape) < 1 or len(symbols) != np.prod(latent_shape)
                or np.any(symbols < 0) or np.any(symbols >= K)):
            raise ValueError('Invalid indices or shape')
        outputs = [BitsOut() for _ in range(3)]
        previous = 16
        for symbol in symbols:
            symbol = int(symbol)
            outputs[0].write(symbol, 11)
            outputs[1].write(self.marginal.codes[symbol], self.marginal.lengths[symbol])
            table = self.conditional[previous]
            outputs[2].write(table.codes[symbol], table.lengths[symbol])
            previous = symbol // 128
        payloads = [out.finish() for out in outputs]
        choice = min(range(3), key=lambda i: (len(payloads[i]), i))
        payload = payloads[choice]
        crc = zlib.crc32(payload, zlib.crc32(side))
        header = HEADER.pack(MAGIC, 1, choice, 0, 0, *shape, *latent_shape, len(side), self.identifier, crc)
        return header + side + payload, choice

    def decode(self, frame):
        if len(frame) < HEADER.size:
            raise ValueError('Truncated frame')
        magic, version, choice, a, b, h, w, lh, lw, side_size, identifier, crc = HEADER.unpack_from(frame)
        if (magic != MAGIC or version != 1 or choice not in (0, 1, 2) or a or b
                or identifier != self.identifier or min(h, w, lh, lw) < 1
                or lh * lw > 10_000_000 or side_size > len(frame) - HEADER.size):
            raise ValueError('Invalid or incompatible frame header')
        side = frame[HEADER.size:HEADER.size + side_size]
        payload = frame[HEADER.size + side_size:]
        if zlib.crc32(payload, zlib.crc32(side)) != crc:
            raise ValueError('Frame CRC mismatch')
        reader = BitsIn(payload)
        symbols = np.empty(lh * lw, dtype=np.uint16)
        previous = 16
        for index in range(len(symbols)):
            if choice == 0:
                symbol = reader.read(11)
            else:
                table = self.marginal if choice == 1 else self.conditional[previous]
                symbol = table.decode(reader)
            symbols[index] = symbol
            previous = symbol // 128
        reader.validate_end()
        return symbols, (h, w), (lh, lw), side
