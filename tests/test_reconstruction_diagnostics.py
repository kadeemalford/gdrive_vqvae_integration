import unittest

import numpy as np
import torch

from finetuning.analyze_sampling_results import grouped_metrics, nominal_frequency
from finetuning.diagnose_reconstruction import diagnostic_forward, spectral_metrics


class DiagnosticsTests(unittest.TestCase):
    def test_nominal_frequency_not_sample_rate(self):
        self.assertEqual(nominal_frequency("Copy of 3000Hz.h5"), "3000")
        self.assertEqual(nominal_frequency("unknown.h5"), "unknown")

    def test_identity_spectrum(self):
        values = np.random.default_rng(1).normal(size=(8, 1000)).astype(np.float32)
        result = spectral_metrics(values, values)
        self.assertEqual(result["fft_magnitude_relative_error"], 0)
        self.assertAlmostEqual(result["fft_phase_alignment"], 1)
        self.assertAlmostEqual(result["high_nyquist_power_ratio"], 1)

    def test_attenuation_is_not_good_reconstruction(self):
        values = np.random.default_rng(2).normal(size=(8, 1000)).astype(np.float32)
        result = spectral_metrics(values, values * 0.5)
        self.assertAlmostEqual(result["fft_magnitude_relative_error"], 0.5)
        self.assertAlmostEqual(result["high_nyquist_power_ratio"], 0.25)
        self.assertAlmostEqual(result["fft_phase_alignment"], 1)

    def test_padding_and_crop_matches_forward_geometry(self):
        class IdentityModel:
            downsample_levels = 3
            encoder = staticmethod(lambda x: x)
            decoder = staticmethod(lambda x: x)
            quantizer = staticmethod(lambda x: (x, torch.tensor(0), torch.tensor([0]), torch.tensor(1)))
        tensor = torch.randn(1, 1, 5, 11)
        decoded, continuous, *_ = diagnostic_forward(IdentityModel(), tensor)
        torch.testing.assert_close(decoded, tensor)
        torch.testing.assert_close(continuous, tensor)

    def test_macro_aggregation(self):
        rows = [{"family": "a", "correlation": 0.2, "relative_rmse": 1, "snr_db": 0},
                {"family": "a", "correlation": 0.4, "relative_rmse": 0.8, "snr_db": 1}]
        result = grouped_metrics(rows, ("family",))[0]
        self.assertAlmostEqual(result["correlation"], 0.3)


if __name__ == "__main__":
    unittest.main()
