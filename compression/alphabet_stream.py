"""Experimental FAK1 framing for power-of-two VQ alphabets, fixed/range fallback."""
import hashlib
import struct
import zlib

import numpy as np

HEADER = struct.Struct('<4sBBHIIIII32sII')
MAGIC = b'FAK1'


class AlphabetStream:
    def __init__(self, counts, checkpoint_hash):
        counts = np.asarray(counts, dtype=np.int64)
        k = len(counts)
        if k < 2 or k > 32768 or k & (k - 1) or np.any(counts < 0):
            raise ValueError('Expected power-of-two nonnegative counts')
        self.k, self.bits = k, k.bit_length() - 1
        self.probabilities = ((counts + 1.) / (counts + 1.).sum()).astype(np.float32)
        self.identifier = hashlib.sha256(bytes.fromhex(checkpoint_hash) + counts.astype('<i8').tobytes()).digest()

    def _frame(self, method, shape, latent, side, payload):
        head = HEADER.pack(MAGIC, 1, method, self.k, *shape, *latent, len(side), self.identifier, len(payload), 0)
        crc = zlib.crc32(head + side + payload)
        return head[:-4] + struct.pack('<I', crc) + side + payload

    def encode(self, indices, shape, latent, side):
        import constriction
        indices = np.asarray(indices, dtype=np.int32).reshape(-1)
        if len(indices) != int(np.prod(latent)) or not len(indices) or indices.min() < 0 or indices.max() >= self.k:
            raise ValueError('Invalid symbols or latent shape')
        shifts = np.arange(self.bits - 1, -1, -1)
        fixed = np.packbits(((indices[:, None] >> shifts) & 1).astype(np.uint8)).tobytes()
        checksum = struct.pack('<I', zlib.crc32(indices.astype('<u2').tobytes()))
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(indices, constriction.stream.model.Categorical(self.probabilities, perfect=False))
        ranged = encoder.get_compressed().astype('<u4').tobytes()
        return min((self._frame(0, shape, latent, side, checksum + fixed),
                    self._frame(1, shape, latent, side, checksum + ranged)), key=len)

    def decode(self, frame):
        import constriction
        if len(frame) < HEADER.size:
            raise ValueError('Truncated alphabet frame')
        values = list(HEADER.unpack_from(frame))
        magic, version, method, k, h, w, lh, lw, side_size, identity, payload_size, crc = values
        if (magic != MAGIC or version != 1 or method not in (0, 1) or k != self.k or identity != self.identifier
                or min(h, w, lh, lw) < 1 or h * w > 16 * 1024 * 1024 or lh * lw > 1024 * 1024
                or HEADER.size + side_size + payload_size != len(frame) or payload_size < 4):
            raise ValueError('Invalid alphabet frame header')
        values[-1] = 0
        if zlib.crc32(HEADER.pack(*values) + frame[HEADER.size:]) != crc:
            raise ValueError('Alphabet frame CRC mismatch')
        side = frame[HEADER.size:HEADER.size + side_size]
        payload = frame[HEADER.size + side_size:]
        expected = struct.unpack_from('<I', payload)[0]
        payload = payload[4:]
        if method == 0:
            nbits = lh * lw * self.bits
            if len(payload) != (nbits + 7) // 8:
                raise ValueError('Invalid fixed payload length')
            bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
            if bits[nbits:].any():
                raise ValueError('Noncanonical bit padding')
            indices = (bits[:nbits].reshape(-1, self.bits) @ (1 << np.arange(self.bits - 1, -1, -1))).astype(np.uint16)
        else:
            if not payload or len(payload) % 4:
                raise ValueError('Invalid range payload')
            decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(payload, dtype='<u4').copy())
            indices = decoder.decode(constriction.stream.model.Categorical(self.probabilities, perfect=False), lh * lw).astype(np.uint16)
        if zlib.crc32(indices.astype('<u2').tobytes()) != expected:
            raise ValueError('Decoded symbols CRC mismatch')
        return indices, (h, w), (lh, lw), side
