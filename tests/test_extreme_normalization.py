import unittest

import numpy as np

from finetuning.investigate_corpus_extremes import normalization_impact


class ExtremeNormalizationTests(unittest.TestCase):
    def test_finite_extreme_erases_other_channels(self):
        raw = np.random.default_rng(1).normal(size=(8, 1000)).astype(np.float32)
        raw[-1, 10] = np.float32(1e35)
        result = normalization_impact(raw)
        self.assertEqual(result["extreme_cells"], 1)
        self.assertEqual(result["channels_with_extremes"], 1)
        self.assertEqual(result["varying_channels_erased_by_global_normalization"], 7)

    def test_normal_range_preserves_variation(self):
        raw = np.random.default_rng(2).normal(size=(8, 1000)).astype(np.float32)
        result = normalization_impact(raw)
        self.assertEqual(result["extreme_cells"], 0)
        self.assertEqual(result["varying_channels_erased_by_global_normalization"], 0)


if __name__ == "__main__":
    unittest.main()
