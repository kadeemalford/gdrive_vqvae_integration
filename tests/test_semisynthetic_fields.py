import unittest

import numpy as np

from finetuning.semisynthetic_fields import compose


class SemisyntheticFieldTests(unittest.TestCase):
    def test_repeatability_and_preservation_of_original(self):
        raw = np.random.default_rng(1).normal(size=(7, 101)).astype(np.float32)
        before = raw.tobytes()
        a, metadata = compose(raw, 5e-5, .2, .1, np.random.default_rng(104))
        b, other = compose(raw, 5e-5, .2, .1, np.random.default_rng(104))
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(metadata, other)
        self.assertEqual(raw.tobytes(), before)
        self.assertEqual(a.shape, raw.shape)
        self.assertEqual(a.dtype, np.float32)
        self.assertFalse(np.array_equal(a, raw))
        self.assertGreater(metadata["added_rms_float64"], 0)

    def test_small_and_constant_tiles_remain_finite(self):
        for shape in ((1, 1), (1, 17), (3, 1)):
            raw = np.zeros(shape, np.float32)
            result, _ = compose(raw, 5e-5, 4., 1., np.random.default_rng(104))
            self.assertTrue(np.isfinite(result).all())
            self.assertEqual(result.shape, shape)

    def test_invalid_geometry_is_not_silently_repaired(self):
        raw = np.ones((3, 17), np.float32)
        for dt, dx in ((0., 1.), (1., -1.), (np.nan, 1.)):
            with self.assertRaises(ValueError):
                compose(raw, dt, dx, 1., np.random.default_rng(104))
