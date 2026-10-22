import unittest
import numpy as np
from finetuning.audit_channel_representation import ordinary_fidelity, aggregate


class ChannelRepresentationTests(unittest.TestCase):
    def test_identical_raw_scores_zero_error(self):
        raw = np.random.default_rng(1).normal(size=(4, 1000)).astype(np.float32)
        row = ordinary_fidelity(raw, raw, np.ones_like(raw, dtype=bool))
        self.assertEqual(row["ordinary_global_centered_relative_rmse"], 0)
        self.assertEqual(row["ordinary_channel_macro_relative_rmse"], 0)

    def test_channel_balance_detects_weak_channel_errors(self):
        raw = np.random.default_rng(2).normal(size=(4, 1000)).astype(np.float32)
        raw[0] *= 1000
        predicted = raw.copy()
        predicted[1:] = 0
        row = ordinary_fidelity(raw, predicted, np.ones_like(raw, dtype=bool))
        self.assertLess(row["ordinary_global_centered_relative_rmse"], .01)
        self.assertGreater(row["ordinary_channel_macro_relative_rmse"], .7)

    def test_escape_region_does_not_influence_fidelity(self):
        raw = np.random.default_rng(3).normal(size=(4, 1000)).astype(np.float32)
        raw[0, 0] = 1e35
        valid = np.abs(raw.astype(np.float64)) < 1e20
        predicted = raw.copy()
        predicted[0, 0] = 0
        row = ordinary_fidelity(raw, predicted, valid)
        self.assertEqual(row["ordinary_global_centered_relative_rmse"], 0)

    def test_robust_metrics_do_not_hide_weak_channels_from_main_error(self):
        raw = np.random.default_rng(4).normal(size=(100, 1000)).astype(np.float32)
        raw *= 1000
        raw[-1] *= 1e-7
        predicted = raw.copy()
        predicted[-1] += 10
        row = ordinary_fidelity(raw, predicted, np.ones_like(raw, dtype=bool))
        self.assertEqual(row["diagnostic_weak_variable_channels"], 1)
        self.assertGreater(row["ordinary_global_centered_relative_rmse"], 0)
        self.assertGreater(row["ordinary_channel_macro_relative_rmse"], 100)
        self.assertEqual(row["ordinary_channel_median_relative_rmse"], 0)
        self.assertEqual(row["ordinary_nonweak_channel_macro_relative_rmse"], 0)
