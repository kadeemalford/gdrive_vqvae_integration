import unittest

import numpy as np

from compression.predictive_stream import compress, compress_packet, decompress, decode_normalized
from compression.numeric_restoration import encode_numeric


class PredictiveStreamTests(unittest.TestCase):
    def test_bound_and_bit_exact_escapes(self):
        raw = np.random.default_rng(12).normal(size=(7, 1000)).astype(np.float32)
        raw[1, 0] = 1e35
        raw.view(np.uint32)[2, 4] = 0x7fffffff
        raw[3, 8] = float('-inf')
        packet = encode_numeric(raw)
        for epsilon in (.1, .01, .001):
            frame, _ = compress_packet(packet, epsilon)
            values, side, metadata = decode_normalized(frame)
            self.assertEqual(side, packet.side_stream)
            self.assertEqual(metadata['normalized_epsilon'], epsilon)
            error = np.max(np.abs(values[packet.valid].astype(np.float64) - packet.values[packet.valid]))
            margin = np.finfo(np.float32).eps * max(1., float(np.abs(packet.values).max()))
            self.assertLessEqual(error, epsilon + margin)
            restored = decompress(frame)
            np.testing.assert_array_equal(raw.view(np.uint32)[~packet.valid], restored.view(np.uint32)[~packet.valid])

    def test_predictors_have_the_same_reconstruction(self):
        import struct
        import zlib
        from compression.predictive_stream import HEADER, MAGIC, _residuals
        packet = encode_numeric(np.random.default_rng(9).normal(size=(9, 1000)).astype(np.float32))
        epsilon = .01
        grid = np.rint(packet.values.astype(np.float64) / (2 * epsilon)).astype(np.int64)
        expected = (grid * (2 * epsilon)).astype(np.float32)
        for method in range(3):
            planes = _residuals(grid, method).astype('<i4').view(np.uint8).reshape(-1, 4).T.copy().tobytes()
            payload = zlib.compress(planes)
            side = packet.side_stream
            head = HEADER.pack(MAGIC, 1, method, *grid.shape, 2 * epsilon, len(side), len(payload), 0)
            crc = zlib.crc32(head + side + payload)
            frame = head[:-4] + struct.pack('<I', crc) + side + payload
            np.testing.assert_array_equal(decode_normalized(frame)[0], expected)

    def test_corruption_and_unsupported_precision_fail_closed(self):
        raw = np.random.default_rng(5).normal(size=(3, 1000)).astype(np.float32)
        frame, _ = compress(raw, .01)
        for bad in (frame[:-1], frame + b'x', b'BAD!' + frame[4:]):
            with self.assertRaises(ValueError):
                decompress(bad)
        for epsilon in (0, -1, float('nan'), 1e-20):
            with self.assertRaises(ValueError):
                compress(raw, epsilon)

    def test_constant_and_single_row_shapes(self):
        for shape in ((1, 1000), (9, 1), (1, 1)):
            raw = np.full(shape, 3.25, dtype=np.float32)
            frame, _ = compress(raw, .01)
            np.testing.assert_array_equal(decompress(frame), raw)

    def test_bounded_decompression_and_explicit_input_side_contract(self):
        import struct
        import zlib
        from compression.predictive_stream import HEADER, MAGIC
        from compression.affine_restoration import affine_side
        from finetuning.reversible_normalization import EncodedWindow
        packet = encode_numeric(np.ones((1, 1), dtype=np.float32))
        payload = zlib.compress(b'\x00' * 100000)
        side = packet.side_stream
        head = HEADER.pack(MAGIC, 1, 0, 1, 1, .02, len(side), len(payload), 0)
        crc = zlib.crc32(head + side + payload)
        frame = head[:-4] + struct.pack('<I', crc) + side + payload
        with self.assertRaises(ValueError):
            decompress(frame)
        affine = EncodedWindow(packet.values, packet.valid, affine_side(side, 1., 0.))
        with self.assertRaises(ValueError):
            compress_packet(affine, .01)
