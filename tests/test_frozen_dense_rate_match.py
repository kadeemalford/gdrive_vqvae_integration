import unittest

import torch

from finetuning.frozen_dense_rate_match import choose_indices


class FrozenRateChoiceTests(unittest.TestCase):
    def test_zero_is_nearest_and_rate_can_choose_cheaper_id(self):
        distance = torch.tensor([[0., 1.], [1., 0.]])
        bits = torch.tensor([1., 8.])
        self.assertTrue(torch.equal(choose_indices(distance, bits, 0), distance.argmin(1)))
        self.assertTrue(torch.equal(choose_indices(distance, bits, 1), torch.tensor([0, 0])))

    def test_negative_or_nonfinite_penalty_rejected(self):
        for penalty in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                choose_indices(torch.zeros(1, 2), torch.ones(2), penalty)
