import csv
import json
import tempfile
import unittest
from pathlib import Path

from finetuning.benchmark_matched_temporal_rates import completed_windows, nominal_budget


class MatchedTemporalRateTests(unittest.TestCase):
    def test_round_indices_once_and_charge_side(self):
        self.assertEqual(nominal_budget({"fixed_index_bits": "11", "side_bytes": "40"}), 42)

    def test_negative_rate_rejected(self):
        with self.assertRaises(ValueError):
            nominal_budget({"fixed_index_bits": "-1", "side_bytes": "40"})

    def test_partial_attempt_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "COMPLETE").touch()
            (root / "arm.json").write_text(json.dumps({"input_cells": 20}))
            expected = {str(root / "a.npz"), str(root / "b.npz")}
            for number, paths in ((1, sorted(expected)), (2, sorted(expected)[:1])):
                with (root / f"arm.windows.attempt_{number:03d}.csv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["sample_path", "input_cells"])
                    writer.writeheader()
                    writer.writerows({"sample_path": path, "input_cells": 10} for path in paths)
            rows, path = completed_windows(root, "arm", expected)
            self.assertEqual(len(rows), 2)
            self.assertEqual(path.name, "arm.windows.attempt_001.csv")
            with self.assertRaisesRegex(ValueError, "No complete"):
                completed_windows(root, "arm", expected | {str(root / "c.npz")})
