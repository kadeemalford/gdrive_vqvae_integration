import unittest

import numpy as np

from acquisition.analyze_quantx_pretraining import diagnostics
from acquisition.diagnose_quantx_drift import trend_probe
from acquisition.prepare_quantx_bridge import normalize_phase


class QuantxDiagnosticsTests(unittest.TestCase):
    def test_identical_sinusoidal_channels_have_rank_one(self):
        signal = np.sin(2 * np.pi * np.arange(2048) / 32)
        result = diagnostics(np.tile(signal[:, None], (1, 16)), 1024)
        self.assertAlmostEqual(result['spatial_effective_rank'], 1, places=6)
        self.assertEqual(result['dominant_frequency_hz'], 32)
        self.assertGreater(result['top5_spectral_bin_energy_fraction'], .999)
        self.assertAlmostEqual(result['adjacent_channel_cosine'], 1)

    def test_diagnostics_do_not_modify_original_and_are_scale_invariant(self):
        x = np.random.default_rng(7).normal(size=(2048, 16))
        before = x.copy()
        a, b = diagnostics(x, 10000), diagnostics(5 * x, 10000)
        np.testing.assert_array_equal(x, before)
        for key in ('spatial_effective_rank', 'top5_spectral_bin_energy_fraction', 'top10_channel_energy_fraction'):
            self.assertAlmostEqual(a[key], b[key])
        self.assertGreater(a['spatial_effective_rank'], 15)
        self.assertAlmostEqual(a['above_fs_quarter_energy_fraction'], .5, delta=.03)

    def test_constant_and_nonfinite_data(self):
        self.assertEqual(diagnostics(np.ones((32, 2)), 10), {'constant': True})
        with self.assertRaises(ValueError):
            diagnostics(np.array([[1, np.nan], [2, 3]]), 10)

    def test_drift_probe_separates_linear_ramp_and_tone(self):
        t = np.arange(2048, dtype=np.float64)
        ramp = t[:, None] * np.ones((1, 3))
        result = trend_probe(ramp, 10000, 500)
        self.assertAlmostEqual(result['linear_trend_energy_fraction'], 1)
        signal = np.sin(2 * np.pi * 500 * t / 10000)[:, None]
        result = trend_probe(signal, 10000, 500)
        self.assertGreater(result['detrended_excitation_band']['pooled'], .999)

    def test_phase_normalization_retains_small_changes_at_large_offset(self):
        raw = np.arange(100, dtype=np.int32).reshape(20, 5) + 2_000_000_000
        normalized, mean, scale = normalize_phase(raw)
        self.assertEqual(normalized.shape, (5, 20))
        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized.std()), 1, places=6)
        self.assertGreater(float(np.std(normalized[0])), .9)
        self.assertAlmostEqual(mean, 2_000_000_049.5)
        self.assertGreater(scale, 0)
        constant, _, _ = normalize_phase(np.ones((10, 2), dtype=np.int32))
        self.assertFalse(constant.any())
