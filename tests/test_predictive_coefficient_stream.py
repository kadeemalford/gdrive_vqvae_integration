import unittest

import numpy as np

from compression.adaptive_coefficient_stream import pack_best
from compression.predictive_coefficient_stream import (
    pack_predictive, prediction_residual, restore_residual, unpack_predictive)


class PredictiveCoefficientTests(unittest.TestCase):
    def test_all_reversible_methods(self):
        array = np.random.default_rng(3).integers(-100, 100, size=(1, 1, 17, 31), dtype=np.int32)
        for method in range(4):
            recovered = restore_residual(prediction_residual(array, method), method)
            self.assertEqual(array.tobytes(), recovered.tobytes())

    def test_predictive_stream_and_complete_fallback(self):
        for array in [np.tile(np.arange(1000, dtype=np.int32), (17, 1)),
                      np.random.default_rng(4).integers(-4, 4, size=(17, 1000), dtype=np.int32)]:
            frame = pack_predictive([array], {"kind": "test"}, b"side")
            self.assertLessEqual(len(frame), len(pack_best([array], {"kind": "test"}, b"side")))
            recovered, metadata, side = unpack_predictive(frame)
            self.assertEqual(array.tobytes(), recovered[0].tobytes())
            self.assertEqual(metadata, {"kind": "test"})
            self.assertEqual(side, b"side")
        with self.assertRaises(ValueError):
            unpack_predictive(frame[:-1])

    def test_extreme_fallback_and_overflow(self):
        array = np.array([[np.iinfo(np.int32).min, np.iinfo(np.int32).max]], np.int32)
        self.assertEqual(unpack_predictive(pack_predictive([array], {}, b""))[0][0].tobytes(), array.tobytes())
        with self.assertRaises(ValueError):
            prediction_residual(array, 1)
