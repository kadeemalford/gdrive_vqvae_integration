import unittest

import numpy as np

from finetuning.feasibility_metrics import target_supported_spectra, summarize_target_spectra
from finetuning.prospective_spectral import merge_spectral


class FeasibilityMetricsTests(unittest.TestCase):
    def setUp(self):
        self.x = np.sin(2 * np.pi * 200 * np.arange(1000) / 20000)[None]

    def evaluate(self, x, y, scale=.01):
        return summarize_target_spectra(target_supported_spectra(x, y, 1 / 20000, scale))["0_1khz"]

    def test_erasure_cannot_remove_target_support(self):
        identity = self.evaluate(self.x, self.x)
        erased = self.evaluate(self.x, np.zeros_like(self.x))
        self.assertEqual(identity["supported_bins"], erased["supported_bins"])
        self.assertEqual(identity["supported_target_energy"], erased["supported_target_energy"])
        self.assertEqual(identity["supported_complex_rmse"], 0.)
        self.assertEqual(erased["supported_complex_rmse"], 1.)
        self.assertEqual(erased["missing_phase_energy_fraction"], 1.)
        self.assertIsNone(erased["observed_phase_error_radians"])
        self.assertAlmostEqual(erased["phase_penalty_radians_missing_pi"], np.pi)

    def test_inversion_attenuation_and_unit_equivariance(self):
        inverse = self.evaluate(self.x, -self.x)
        self.assertAlmostEqual(inverse["observed_phase_error_radians"], np.pi)
        self.assertAlmostEqual(inverse["supported_complex_rmse"], 2.)
        attenuated = self.evaluate(self.x, .5 * self.x)
        scaled = self.evaluate(self.x * 1e6, .5 * self.x * 1e6, 1e4)
        self.assertAlmostEqual(attenuated["supported_amplitude_rmse"], .5)
        self.assertAlmostEqual(attenuated["observed_phase_error_radians"], 0.)
        for key in ("supported_complex_rmse", "supported_amplitude_rmse",
                    "missing_phase_energy_fraction", "target_support_energy_fraction"):
            self.assertAlmostEqual(attenuated[key], scaled[key])

    def test_weak_target_stays_visible_without_fabricated_phase(self):
        result = self.evaluate(self.x * 1e-6, self.x)
        self.assertEqual(result["supported_bins"], 0)
        self.assertIsNone(result["observed_phase_error_radians"])
        self.assertIsNone(result["phase_penalty_radians_missing_pi"])
        self.assertGreater(result["relative_complex_rmse"], 1000)

    def test_channel_pooling_matches_sufficient_statistic_merge(self):
        total = {}
        merge_spectral(total, target_supported_spectra(self.x, self.x, 1 / 20000, .01))
        merge_spectral(total, target_supported_spectra(self.x, -self.x, 1 / 20000, .01))
        merged = summarize_target_spectra(total)["0_1khz"]
        pooled = self.evaluate(np.concatenate([self.x, self.x]),
                               np.concatenate([self.x, -self.x]))
        for key in merged:
            self.assertAlmostEqual(merged[key], pooled[key])
        self.assertAlmostEqual(merged["observed_phase_error_radians"], np.pi / 2)

    def test_zero_constant_and_invalid_inputs(self):
        result = self.evaluate(np.zeros_like(self.x), np.zeros_like(self.x))
        self.assertIsNone(result["relative_complex_rmse"])
        self.assertIsNone(result["missing_phase_energy_fraction"])
        constant = self.evaluate(np.ones_like(self.x), np.ones_like(self.x) * 2)
        self.assertEqual(constant["target_energy"], 0.)
        for scale in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.evaluate(self.x, self.x, scale)
        with self.assertRaises(ValueError):
            target_supported_spectra(self.x, self.x, 1 / 10000, .01)

    def test_nyquist_is_charged_once(self):
        x = ((-1.) ** np.arange(1000))[None]
        bands = target_supported_spectra(x, x, 1 / 20000, .01)
        windowed = x[0] * np.hanning(1000)
        # Unnormalized FFT energy equals N times time energy; DC is excluded.
        expected = 1000 * np.square(windowed).sum() - np.square(windowed.sum())
        self.assertAlmostEqual(sum(b["target_energy"] for b in bands.values()), expected, places=7)
