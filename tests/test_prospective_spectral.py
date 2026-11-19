import unittest

import numpy as np

from finetuning.prospective_spectral import spectral_result, spectral_statistics


class ProspectiveSpectralTests(unittest.TestCase):
    def test_identity_and_undefined_phase(self):
        x = np.sin(2 * np.pi * 200 * np.arange(1000) / 20000)[None]
        result = spectral_result(spectral_statistics(x, x, 1 / 20000, .01))
        self.assertEqual(result["0_1khz"]["relative_rmse"], 0.)
        self.assertAlmostEqual(result["0_1khz"]["phase_error_radians"], 0., places=15)
        zero = spectral_result(spectral_statistics(x * 0, x * 0, 1 / 20000, .01))
        self.assertIsNone(zero["0_1khz"]["phase_error_radians"])
        self.assertIsNone(zero["0_1khz"]["relative_rmse"])

    def test_opposite_phase_and_unit_equivariance(self):
        x = np.sin(2 * np.pi * 200 * np.arange(1000) / 20000)[None]
        a = spectral_result(spectral_statistics(x, -x, 1 / 20000, .01))
        b = spectral_result(spectral_statistics(x * 1e6, -x * 1e6, 1 / 20000, 1e4))
        self.assertAlmostEqual(a["0_1khz"]["phase_error_radians"], np.pi)
        self.assertAlmostEqual(a["0_1khz"]["relative_rmse"], b["0_1khz"]["relative_rmse"])
