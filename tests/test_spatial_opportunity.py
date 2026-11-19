import unittest

import numpy as np

from compression.research_coefficient_stream import pack, unpack
from finetuning.spatial_opportunity import decode_frame, encode_frame


class SpatialOpportunityTests(unittest.TestCase):
    def test_coefficients_and_corruption(self):
        x = np.arange(-20, 20, dtype=np.int32).reshape(4, 10)
        frame = pack([x], {"a": 1}, b"side")
        arrays, metadata, side = unpack(frame)
        np.testing.assert_array_equal(x, arrays[0])
        self.assertEqual(side, b"side")
        self.assertEqual(metadata, {"a": 1})
        for broken in (frame[:-1], frame + b"x", frame[:40] + bytes([frame[40] ^ 1]) + frame[41:]):
            with self.assertRaises(ValueError):
                unpack(broken)

    def test_closed_loop_odd_and_single_channels(self):
        rng = np.random.default_rng(1)
        for channels in (1, 2, 3, 17):
            raw = rng.normal(size=(channels, 35)).astype(np.float32)
            for predictive in (False, True):
                frame = encode_frame(raw, .01, .8, predictive)
                restored = decode_frame(frame)
                self.assertEqual(restored.shape, raw.shape)
                self.assertLess(float(np.abs(restored - raw).max()), .03)

    def test_constant_and_escape(self):
        raw = np.zeros((3, 100), np.float32)
        np.testing.assert_array_equal(decode_frame(encode_frame(raw, 2., .8)), raw)
        raw[1, 3] = 1e30
        restored = decode_frame(encode_frame(raw, 2., .8))
        self.assertEqual(restored[1, 3].tobytes(), raw[1, 3].tobytes())
