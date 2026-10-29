"""Tests for raw-only spatial concentration diagnostics."""
import unittest

import numpy as np

from acquisition.audit_spatial_energy_bands import concentration, select_files


class SpatialEnergyBandsTest(unittest.TestCase):
    def test_uniform(self):
        metrics = concentration(np.ones(100), np.ones(100, dtype=bool))
        self.assertAlmostEqual(metrics["top10pct_channel_energy_share"], .1)
        self.assertAlmostEqual(metrics["effective_channel_fraction"], 1.)

    def test_concentrated_tail(self):
        energy = np.zeros(100)
        energy[-10:] = 1.
        metrics = concentration(energy, np.ones(100, dtype=bool))
        self.assertAlmostEqual(metrics["top10pct_channel_energy_share"], 1.)
        self.assertAlmostEqual(metrics["last10pct_position_energy_share"], 1.)
        self.assertAlmostEqual(metrics["effective_channel_fraction"], .1)
        self.assertLessEqual(metrics["span80_fraction_original_channels"], .09)

    def test_nonfinite_channel_positions_not_collapsed(self):
        energy = np.array([1., 0., 1.])
        metrics = concentration(energy, np.array([True, False, True]))
        self.assertEqual(metrics["span80_start_index"], 0)
        self.assertEqual(metrics["span80_end_index"], 2)

    def test_selection_reproducible_and_stratified(self):
        rows = [{"family": family, "configuration": f"config_{config}",
                 "drive_id": f"{family}_{config}_{file}", "title": f"file_{file}"}
                for family in ("doped", "multicore", "multimode", "single_mode")
                for config in range(4) for file in range(4)]
        selected = select_files(rows, 123)
        self.assertEqual(selected, select_files(list(reversed(rows)), 123))
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({r["drive_id"] for r in selected}), 24)
        for family in {r["family"] for r in selected}:
            subset = [r for r in selected if r["family"]==family]
            self.assertEqual(len(subset), 6)
            self.assertEqual(len({r["configuration"] for r in subset}), 3)


if __name__ == "__main__":
    unittest.main()
