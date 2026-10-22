import unittest

import torch

from finetuning.run_frozen_rate_stability import midpoint_state


class FrozenRateStabilityTests(unittest.TestCase):
    def test_fixed_midpoint_includes_embeddings_and_preserves_dtype(self):
        before = {"embedding": torch.tensor([1., 3.]), "count": torch.tensor(2)}
        after = {"embedding": torch.tensor([3., 5.]), "count": torch.tensor(2)}
        result = midpoint_state(before, after)
        self.assertTrue(torch.equal(result["embedding"], torch.tensor([2., 4.])))
        self.assertTrue(torch.equal(before["embedding"], torch.tensor([1., 3.])))
        self.assertEqual(result["count"].dtype, torch.int64)

    def test_incompatible_or_nonfinite_state_refused(self):
        a = {"x": torch.ones(2)}
        for b in ({"y": torch.ones(2)}, {"x": torch.ones(3)},
                  {"x": torch.ones(2, dtype=torch.float64)}, {"x": torch.tensor([1., float("nan")])}):
            with self.assertRaises(ValueError):
                midpoint_state(a, b)
        with self.assertRaises(ValueError):
            midpoint_state({"x": torch.tensor(1)}, {"x": torch.tensor(2)})
