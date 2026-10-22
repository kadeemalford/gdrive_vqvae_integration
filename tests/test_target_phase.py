import unittest
import numpy as np
from finetuning.audit_phase_tradeoff import target_weighted_phase


class TargetPhaseTests(unittest.TestCase):
    def signal(self):
        return np.random.default_rng(1).normal(size=(3, 1000))

    def test_amplitude_does_not_change_target_weighted_alignment(self):
        original = self.signal()
        low = target_weighted_phase(original, original * .1)
        high = target_weighted_phase(original, original * .9)
        self.assertAlmostEqual(low["target_phase_all"], 1.)
        self.assertAlmostEqual(low["target_phase_all"], high["target_phase_all"])

    def test_inversion_has_negative_alignment(self):
        original = self.signal()
        self.assertAlmostEqual(target_weighted_phase(original, -original)["target_phase_all"], -1.)

    def test_absent_reconstruction_has_zero_coverage_not_perfect_phase(self):
        result = target_weighted_phase(self.signal(), np.zeros((3, 1000)))
        self.assertEqual(result["target_phase_all"], 0.)
        self.assertEqual(result["target_phase_coverage_all"], 0.)
        self.assertTrue(np.isnan(result["target_phase_conditional_all"]))

    def test_constant_signal_has_no_non_dc_phase(self):
        result = target_weighted_phase(np.ones((3, 1000)), np.ones((3, 1000)))
        self.assertTrue(np.isnan(result["target_phase_all"]))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            target_weighted_phase(np.ones((3, 1000)), np.ones((2, 1000)))
