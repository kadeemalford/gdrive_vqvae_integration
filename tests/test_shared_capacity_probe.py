import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from finetuning.run_capacity_probe import select_cases
from finetuning.run_shared_capacity_probe import check_partitions, evaluate_split, select_pool_and_controls, update_schedule


class SharedCapacityTests(unittest.TestCase):
    def test_schedule_balanced_and_resume_prefix_stable(self):
        schedule = update_schedule(8, 4000, 2026)
        self.assertEqual(Counter(schedule), {i: 500 for i in range(8)})
        self.assertEqual(schedule[:500], update_schedule(8, 500, 2026))

    def test_pool_contains_original_anchors_and_distinct_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            rows = [{"family": family, "configuration": config, "drive_id": f"{family}-{config}-{file}",
                     "sample_path": f"{family}-{config}-{file}-{block}", "block_index": str(block), "split": "train"}
                    for family in ("doped", "multicore", "multimode", "single_mode")
                    for config in ("one", "two") for file in range(2) for block in range(3)]
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            pool, controls = select_pool_and_controls(path, 2026)
            self.assertEqual(len(pool), 8)
            self.assertTrue({r["sample_path"] for r in select_cases(path, 2026)} <= {r["sample_path"] for r in pool})
            self.assertEqual(len({r["drive_id"] for r in pool}), 8)
            self.assertEqual(Counter(r["family"] for r in pool), {f: 2 for f in ("doped", "multicore", "multimode", "single_mode")})
            self.assertFalse({r["sample_path"] for r in pool} & {r["sample_path"] for r in controls})
            self.assertEqual((pool, controls), select_pool_and_controls(path, 2026))

    def test_configuration_leakage_rejected(self):
        train = {"sample_path": "a", "drive_id": "1", "family": "doped", "configuration": "one", "split": "train"}
        val = {**train, "sample_path": "b", "drive_id": "2", "split": "val"}
        with self.assertRaisesRegex(ValueError, "Configuration"):
            check_partitions([train], [], [val])

    def test_control_in_pool_rejected(self):
        train = {"sample_path": "a", "drive_id": "1", "family": "doped", "configuration": "one", "split": "train"}
        with self.assertRaises(ValueError):
            check_partitions([train], [train], [])

    def test_evaluation_preserves_per_sample_and_macro_metrics(self):
        class Identity(torch.nn.Module):
            downsample_levels = 2
            encoder = staticmethod(lambda x: x)
            decoder = staticmethod(lambda x: x)
        cases = [{"sample_path": f"{family}.npz", "drive_id": family, "family": family, "configuration": "one"}
                 for family in ("doped", "multicore")]
        arrays = {r["sample_path"]: np.random.default_rng(1).normal(size=(2, 512)).astype(np.float32) for r in cases}
        result = evaluate_split(Identity(), cases, arrays, "continuous", torch.device("cpu"))
        self.assertEqual(result["macro"]["macro_relative_rmse"], 0)
        self.assertAlmostEqual(result["macro"]["macro_correlation"], 1)
        self.assertEqual(len(result["samples"]), 2)
        self.assertEqual(set(result["families"]), {"doped", "multicore"})


if __name__ == "__main__":
    unittest.main()
