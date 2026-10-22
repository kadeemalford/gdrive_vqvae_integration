from types import SimpleNamespace
import unittest

import numpy as np
import torch

from finetuning.frozen_gain import fitted_gain, fold_gain, window_moments


class FrozenGainTests(unittest.TestCase):
    def test_ordinary_only_fit_and_positive_gain(self):
        x = np.array([[2., 4., 1e20]])
        y = np.array([[1., 2., -1e20]])
        valid = np.array([[True, True, False]])
        self.assertEqual(fitted_gain(window_moments(x, y, valid)), 2.)
        self.assertEqual(fitted_gain({"xy": 1., "yy": 2., "xx": 1.}), .5)
        for totals in ({"xy": 1., "yy": 0., "xx": 1.}, {"xy": -1., "yy": 1., "xx": 1.}):
            with self.assertRaises(ValueError):
                fitted_gain(totals)

    def test_equal_window_not_cell_weighting(self):
        a = window_moments(np.ones(10), np.ones(10), np.ones(10, dtype=bool))
        b = window_moments(np.array([4.]), np.array([2.]), np.array([True]))
        self.assertEqual(fitted_gain({k: a[k] + b[k] for k in a}), float(np.float32(9 / 5)))

    def test_folded_linear_head_matches_output_scaling(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
        model = SimpleNamespace(decoder=SimpleNamespace(net=net))
        x = torch.randn(1, 1, 3, 4)
        before = net(x).detach()
        fold_gain(model, 1.25)
        torch.testing.assert_close(net(x), before * 1.25)
        with self.assertRaises(ValueError):
            fold_gain(model, float("nan"))
        with self.assertRaises(ValueError):
            fold_gain(SimpleNamespace(decoder=SimpleNamespace(net=torch.nn.Sequential(torch.nn.Sigmoid()))), 2.)
