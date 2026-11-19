import unittest

import numpy as np

from compression.adaptive_coefficient_stream import pack, unpack


class AdaptiveCoefficientStreamTests(unittest.TestCase):
    def test_all_methods_exact_and_deterministic(self):
        rng = np.random.default_rng(95)
        arrays = [np.zeros((1, 1, 32, 1000), np.int32),
                  rng.choice(np.array([-4, 0, 3], np.int32), size=(3, 1000), p=[.02, .96, .02]),
                  rng.integers(-2**30, 2**30, size=(3, 10), dtype=np.int32)]
        frame, details = pack(arrays, {"kind": "test"}, b"side", True)
        recovered, descriptor, side = unpack(frame)
        for a, b in zip(arrays, recovered):
            self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual(a.shape, b.shape)
        self.assertEqual(side, b"side")
        self.assertEqual(descriptor, {"kind": "test"})
        self.assertEqual(details["methods"][0], 1)
        self.assertEqual(details["methods"][1], 2)
        self.assertEqual(frame, pack(arrays, descriptor, side))

    def test_signed_extrema_and_one_cell(self):
        for array in [np.array([np.iinfo(np.int32).min], np.int32),
                      np.array([np.iinfo(np.int32).min, np.iinfo(np.int32).max] * 1000, np.int32)]:
            recovered, _, _ = unpack(pack([array], {}, b""))
            np.testing.assert_array_equal(recovered[0], array)

    def test_corruption_and_inputs(self):
        frame = pack([np.arange(500, dtype=np.int32)], {}, b"")
        for bad in (frame[:-1], frame + b"x", b"XXXX" + frame[4:],
                    frame[:-1] + bytes([frame[-1] ^ 1])):
            with self.assertRaises(ValueError):
                unpack(bad)
        for arrays in ([], [np.ones(3)], [np.ones(0, np.int32)]):
            with self.assertRaises(ValueError):
                pack(arrays, {}, b"")
