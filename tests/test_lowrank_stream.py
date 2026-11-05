import unittest

import numpy as np

from compression.lowrank_stream import factorize, compress_factors, decode_normalized, decompress, with_gain
from compression.numeric_restoration import encode_numeric


class LowRankStreamTests(unittest.TestCase):
    def test_transmitted_rank_one_and_gain(self):
        raw = np.outer(np.linspace(-1, 1, 17), np.sin(np.arange(1000) / 15)).astype(np.float32)
        packet = encode_numeric(raw)
        frame = compress_factors(packet, *factorize(packet), 1)
        decoded = decompress(frame)
        self.assertLess(np.linalg.norm(raw - decoded) / np.linalg.norm(raw), .02)
        values, side, metadata = decode_normalized(frame)
        scaled, _, _ = decode_normalized(with_gain(frame, 1.1))
        np.testing.assert_allclose(scaled, values * 1.1, rtol=2e-7, atol=1e-7)
        self.assertEqual((side, metadata['rank']), (packet.side_stream, 1))

    def test_escapes_and_zero_rank_are_explicit(self):
        raw = np.random.default_rng(9).normal(size=(7, 1000)).astype(np.float32)
        raw.view(np.uint32)[0, 0] = 0x7fffffff
        raw[1, 1] = 1e35
        packet = encode_numeric(raw)
        factors = factorize(packet)
        for rank in (0, 1, 7):
            restored = decompress(compress_factors(packet, *factors, rank))
            np.testing.assert_array_equal(raw.view(np.uint32)[~packet.valid], restored.view(np.uint32)[~packet.valid])

    def test_corrupt_frames_and_invalid_ranks_fail_closed(self):
        packet = encode_numeric(np.ones((3, 1000), dtype=np.float32))
        factors = factorize(packet)
        frame = compress_factors(packet, *factors, 0)
        np.testing.assert_array_equal(decompress(frame), np.ones((3, 1000), dtype=np.float32))
        for bad in (frame[:-1], frame + b'x', b'BAD!' + frame[4:]):
            with self.assertRaises(ValueError):
                decompress(bad)
        with self.assertRaises(ValueError):
            compress_factors(packet, *factors, 4)
        with self.assertRaises(ValueError):
            with_gain(frame, 0)

    def test_bounded_payload_and_affine_side_rejection(self):
        import struct
        import zlib
        from compression.lowrank_stream import HEADER, MAGIC
        from compression.affine_restoration import affine_side
        from finetuning.reversible_normalization import EncodedWindow
        packet = encode_numeric(np.ones((3, 1000), dtype=np.float32))
        side = packet.side_stream
        payload = b'\x00' + zlib.compress(b'\x00' * 100000)
        head = HEADER.pack(MAGIC, 1, 8, 0, 3, 1000, 1., len(side), len(payload), 0)
        crc = zlib.crc32(head + side + payload)
        with self.assertRaises(ValueError):
            decompress(head[:-4] + struct.pack('<I', crc) + side + payload)
        bad = EncodedWindow(packet.values, packet.valid, affine_side(side, 1., 0.))
        with self.assertRaises(ValueError):
            compress_factors(bad, *factorize(packet), 0)
