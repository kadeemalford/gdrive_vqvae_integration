import unittest

import numpy as np

from compression.local_scale_stream import LocalScaleStream, symbol_codes


class LocalScaleStreamTests(unittest.TestCase):
    def test_roundtrip_with_edge_tiles_and_explicit_extreme_escapes(self):
        values = np.random.default_rng(14).integers(-20, 20, size=(4, 9, 17), dtype=np.int32)
        values[0, 0, 0] = np.iinfo(np.int32).min
        values[3, -1, -1] = np.iinfo(np.int32).max
        for adaptive in (False, True):
            stream = LocalScaleStream.from_empirical_counts(np.ones((4, 512)), adaptive)
            frame = stream.encode(values, (25, 263), b'preserved-side')
            receiver = LocalScaleStream(np.frombuffer(stream.serialized, dtype='<f4').reshape(-1, 512), 4, adaptive)
            decoded, shape, side = receiver.decode(frame)
            np.testing.assert_array_equal(values, decoded)
            self.assertEqual(shape, (25, 263))
            self.assertEqual(side, b'preserved-side')
            corrupted = bytearray(frame); corrupted[-5] ^= 1
            with self.assertRaises(ValueError):
                receiver.decode(corrupted)

    def test_mismatched_prior_and_invalid_inputs_fail(self):
        stream = LocalScaleStream.from_empirical_counts(np.ones((1, 512)), True)
        frame = stream.encode(np.zeros((1, 1, 1), dtype=np.int32), (1, 1), b'')
        other = LocalScaleStream.from_empirical_counts(np.ones((1, 512)), False)
        with self.assertRaises(ValueError):
            other.decode(frame)
        for values in (np.array([[[np.nan]]]), np.array([[[.25]]]), np.array([], dtype=np.int32).reshape(1, 0, 1)):
            with self.assertRaises(ValueError):
                symbol_codes(values)

    def test_scale_side_is_actually_present_and_counted(self):
        stream = LocalScaleStream.from_empirical_counts(np.ones((1, 512)), True)
        values = np.zeros((1, 8, 8), dtype=np.int32)
        frame = stream.encode(values, (8, 8), b'')
        np.testing.assert_array_equal(stream.decode(frame)[0], values)
        self.assertEqual(len(stream.serialized), 65 * 512 * 4)
