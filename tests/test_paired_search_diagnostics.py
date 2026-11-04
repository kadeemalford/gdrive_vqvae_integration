import unittest

import numpy as np

from finetuning.paired_search_diagnostics import residual_diagnostics, guarded_admit, finite_json
from finetuning.audit_residual_boundaries import boundary_energy


class ResidualDiagnosticsTests(unittest.TestCase):
    def test_boundary_control_removes_target_energy_concentration(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(64, 1000))
        x[:8] *= 10
        x[-8:] *= 10
        x -= x.mean(axis=1, keepdims=True)
        r = boundary_energy(x, x * .8, np.ones_like(x, dtype=bool))
        self.assertAlmostEqual(r['spatial']['edge_interior_relative_error_power_ratio'], 1., places=10)
        self.assertGreater(r['spatial']['edge']['mse'] / r['spatial']['interior']['mse'], 50.)

    def test_undefined_guard_is_rejected_and_not_fabricated(self):
        metric = {'relative_rmse': 0.5, 'correlation': float('nan'), 'ordinary_channel_p90_relative_rmse': 1.}
        self.assertFalse(guarded_admit(metric, metric))
        sanitized = finite_json(metric)
        self.assertIsNone(sanitized['correlation'])
        self.assertFalse(guarded_admit(sanitized, sanitized))

    def test_constant_residual_does_not_fabricate_prediction_gain(self):
        x = np.zeros((3, 1000))
        r = residual_diagnostics(x, x, np.ones_like(x, dtype=bool))
        self.assertIsNone(r['spatial_probe']['relative_rmse_gain'])
        self.assertEqual(r['spatial_probe']['status'], 'constant_target')

    def test_masked_channels_excluded_from_fft(self):
        x = np.ones((3, 1000))
        valid = np.ones_like(x, dtype=bool)
        valid[0, 100] = False
        x[0, 100] = 1e30
        r = residual_diagnostics(x, np.zeros_like(x), valid)
        self.assertEqual(r['fft_channels'], 2)
        self.assertEqual(r['mse'], 1.)

    def test_spatial_dependency_is_measured_on_suffix(self):
        rng = np.random.default_rng(42)
        x = np.tile(rng.normal(size=(1, 1000)), (32, 1))
        r = residual_diagnostics(x, np.zeros_like(x), np.ones_like(x, dtype=bool))
        self.assertGreater(r['spatial_probe']['relative_rmse_gain'], .99)
        self.assertEqual(r['spatial_probe']['fit_cells'], 31 * 500)
        self.assertEqual(r['edge_cells'] + r['interior_cells'], x.size)

    def test_probe_does_not_fit_on_evaluation_suffix(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(2, 1000))
        x[1, :500] = x[0, :500]
        x[1, 500:] = -x[0, 500:]
        r = residual_diagnostics(x, np.zeros_like(x), np.ones_like(x, dtype=bool))
        self.assertGreater(r['spatial_probe']['coefficient'], .99)
        self.assertLess(r['spatial_probe']['relative_rmse_gain'], 0.)

    def test_empty_ordinary_region_is_rejected(self):
        x = np.zeros((2, 1000))
        with self.assertRaises(ValueError):
            residual_diagnostics(x, x, np.zeros_like(x, dtype=bool))

    def test_single_channel_has_no_spatial_probe(self):
        x = np.ones((1, 1000))
        r = residual_diagnostics(x, x, np.ones_like(x, dtype=bool))
        self.assertEqual(r['spatial_probe']['status'], 'insufficient_pairs')

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            residual_diagnostics(np.ones((2, 4)), np.ones((2, 3)), np.ones((2, 4), dtype=bool))
