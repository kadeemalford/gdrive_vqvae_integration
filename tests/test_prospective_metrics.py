import unittest

import numpy as np

from finetuning.prospective_metrics import PhysicalAccumulator


class ProspectiveMetricsTests(unittest.TestCase):
    def score(self, x, y, scale):
        accumulator = PhysicalAccumulator()
        accumulator.update(x, y)
        return accumulator.result(scale)

    def test_zero_and_perturbed_constant(self):
        x = np.zeros(100)
        result = self.score(x, x + 5e-9, 1e-3)
        self.assertAlmostEqual(result["nrmse_star"], 5e-6)
        self.assertIsNone(result["correlation"])
        self.assertEqual(self.score(x, x, 1e-3)["rmse"], 0)

    def test_unit_equivariance(self):
        x = np.arange(100) * 1e-5
        y = x + 1e-6
        a = self.score(x, y, 1e-4)
        b = self.score(x * 1e6, y * 1e6, 1e2)
        self.assertAlmostEqual(a["nrmse_star"], b["nrmse_star"])
        self.assertAlmostEqual(a["correlation"], b["correlation"])

    def test_pool_before_ratio(self):
        x = np.arange(100.)
        y = x + 1.
        accumulator = PhysicalAccumulator()
        accumulator.update(x[:10], y[:10])
        accumulator.update(x[10:], y[10:])
        self.assertEqual(accumulator.result(1), self.score(x, y, 1))

    def test_undefined_weak_and_bad_inputs(self):
        self.assertIsNone(self.score(np.array([0, 1e-9]), np.array([0, 1e-9]), 1e-3)["correlation"])
        with self.assertRaises(ValueError):
            self.score([np.nan], [0], 1)
        with self.assertRaises(ValueError):
            self.score([0], [0], 0)

    def test_high_dc_stable_variance(self):
        x = 1e12 + np.arange(100.)
        result = self.score(x, x + 1, 1e-3)
        self.assertAlmostEqual(result["target_std"], float(np.std(x)))
        self.assertAlmostEqual(result["correlation"], 1.)
