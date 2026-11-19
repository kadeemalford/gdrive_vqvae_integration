import copy
import random
import unittest
from unittest.mock import patch

import torch

from finetuning.run_feasibility_context_alignment import (
    make_schedule, microbatches, parts, normalized_loss, select_training_rows,
)


def rows():
    return [{"drive_id": f"{family}{file}", "group_id": f"{family}/g{group}",
             "split": "train", "sample_path": f"data/train/{family}{group}{file}{row}.npz",
             "channels": "18"}
            for family in ("a", "b") for group in (0, 1)
            for file in (0, 1) for row in (0, 1)]


class AlignmentTests(unittest.TestCase):
    def test_schedule_order_seed_round_robin_and_full_rows(self):
        selected = rows()
        first = make_schedule(selected, 128)
        self.assertEqual(first, make_schedule(list(reversed(selected)), 128))
        self.assertEqual([r["group_id"][0] for r in first], ["a", "b"] * 64)
        self.assertNotEqual(first, make_schedule(selected, 128, seed=2118))
        rng = random.Random(2117)
        for index, result in enumerate(first):
            family = ("a", "b")[index % 2]
            group = rng.choice([f"{family}/g0", f"{family}/g1"])
            file_id = rng.choice([f"{family}0", f"{family}1"])
            expected = rng.choice(sorted([r for r in selected if r["group_id"] == group
                                          and r["drive_id"] == file_id],
                                         key=lambda r: r["sample_path"]))
            self.assertEqual(result, {"update": index + 1, **expected})

    def test_paired_cell_exposure_and_tail_isolation(self):
        for channels in (1, 2, 8, 10, 32, 512, 514):
            for arm in ("A", "B"):
                plan = parts(channels, arm)
                self.assertEqual([i for a, b in plan for i in range(a, b)], list(range(channels)))
                self.assertEqual(sum((b-a)*1000 for a, b in plan), channels*1000)
                batches = microbatches(channels, arm)
                self.assertEqual([p for batch in batches for p in batch], plan)
                if arm == "A":
                    self.assertTrue(all(len(batch) <= 4 for batch in batches))
                    if channels % 8:
                        self.assertEqual(len(batches[-1]), 1)
        self.assertEqual(len(microbatches(512, "A")), 16)
        self.assertEqual(microbatches(2, "A"), [[(0, 2)]])
        for bad in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                parts(bad, "A")

    def test_physical_weighted_loss_and_gradient(self):
        target = torch.zeros(2, 1, 8, 10)
        prediction = torch.ones_like(target, requires_grad=True)
        mask = torch.ones_like(target, dtype=torch.bool)
        qloss = torch.tensor(3., requires_grad=True)
        result = normalized_loss(prediction, target, mask, qloss, [80, 80], 180)
        expected = (1. + 3.) * 160 / 180
        self.assertAlmostEqual(float(result.detach()), expected, places=5)
        result.backward()
        self.assertAlmostEqual(float(qloss.grad), 160/180, places=6)
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_accumulation_matches_partition_weighted_objective(self):
        total = 18000
        accumulated = 0.
        for batch in microbatches(18, "A"):
            count = batch[0][1] - batch[0][0]
            x = torch.zeros(len(batch), 1, count, 1000)
            accumulated += normalized_loss(x+2, x, torch.ones_like(x, dtype=torch.bool),
                                         torch.tensor(1.),
                                         [count*1000]*len(batch), total)
        self.assertAlmostEqual(float(accumulated), 4.)

    def test_pool_uses_only_two_conservative_exclusions_not_roles(self):
        files = [{"drive_id": str(i), "group_id": "a/g"} for i in range(32)]
        freeze = {"f2_candidate_pool": {"files": files,
                  "historical_check_ids": [f"check{i}" for i in range(43)]}}
        exclusions = {"kind": "feasibility-clock-envelope-sensitivity-v1",
                      "removed_file_ids": ["0", "1"], "remaining_file_count": 30,
                      "candidate_roles": {str(i): "check" for i in range(32)}}
        prepared = [{"drive_id": str(i), "group_id": "a/g", "channels": "2",
                     "sample_path": f"data/train/{i}.npz", "split": "train"} for i in range(32)]
        with patch("numpy.load", side_effect=AssertionError("No waveform reads")):
            selected = select_training_rows(prepared, freeze, exclusions)
        self.assertEqual({r["drive_id"] for r in selected}, {str(i) for i in range(2, 32)})
        bad = copy.deepcopy(prepared)
        bad[2]["split"] = "test"
        with self.assertRaises(ValueError):
            select_training_rows(bad, freeze, exclusions)
        with self.assertRaises(ValueError):
            select_training_rows(prepared[3:], freeze, exclusions)

    def test_reserved_schedule_rejected_before_waveform_access(self):
        for split in ("test", "val"):
            bad = rows()
            bad[0]["split"] = split
            with self.assertRaises(ValueError):
                make_schedule(bad, 2)

    def test_empty_valid_mask_preserves_qloss_gradient(self):
        prediction = torch.ones(1, 1, 2, 10, requires_grad=True)
        qloss = torch.tensor(2., requires_grad=True)
        loss = normalized_loss(prediction, prediction.detach(),
                             torch.zeros_like(prediction, dtype=torch.bool),
                             qloss, [20], 20)
        self.assertEqual(float(loss.detach()), 2.)
        loss.backward()
        self.assertEqual(float(qloss.grad), 1.)
        self.assertEqual(float(prediction.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
