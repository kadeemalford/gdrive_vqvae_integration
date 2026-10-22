import unittest
import numpy as np

from finetuning.evaluate_frozen_range_policies import common_scores
from finetuning.reversible_normalization import encode, decode
from finetuning.robust_escape import encode_robust


class FrozenRangeReferenceTests(unittest.TestCase):
    def test_both_policies_use_identical_reference_not_their_own_denominators(self):
        raw = np.random.default_rng(3).normal(size=(512, 1000)).astype(np.float32) * 1e5
        raw[-2:] = 1e16
        old, reference = encode(raw), encode_robust(raw)
        baseline = common_scores(raw, decode(old.values, old.side_stream), reference)
        robust = common_scores(raw, decode(reference.values, reference.side_stream), reference)
        self.assertEqual(baseline["reference_ordinary_cell_fraction"], robust["reference_ordinary_cell_fraction"])
        self.assertGreater(baseline["relative_rmse"], robust["relative_rmse"] * 1000)
        self.assertEqual(robust["outside_reference_raw_rmse"], 0.)
