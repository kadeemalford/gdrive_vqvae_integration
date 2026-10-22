import json
import tempfile
import unittest
from pathlib import Path

import torch

from finetuning.run_sampling_experiment import ROOT, training_plan


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / "window"
        self.samples = Path("data/finetune/fold_0")
        self.protocol = {"epochs": 10}

    def save(self, epoch=6, mode="window", patience=0, best_epoch=None):
        self.target.mkdir(exist_ok=True)
        checkpoint = {"epoch": epoch, "finetune": {
            "sampling": mode, "samples_dir": str(ROOT / self.samples),
            "epochs_without_improvement": patience, "best_epoch": best_epoch or epoch}}
        torch.save(checkpoint, self.target / "last.pt")
        torch.save(checkpoint, self.target / "best.pt")

    def test_new_arm_gets_full_budget(self):
        self.assertEqual(training_plan(self.target, "window", self.protocol, self.samples), (10, None))

    def test_interrupted_arm_only_gets_remaining_epochs(self):
        self.save()
        self.assertEqual(training_plan(self.target, "window", self.protocol, self.samples), (4, self.target / "last.pt"))

    def test_complete_budget_skipped(self):
        self.save(epoch=10)
        self.assertEqual(training_plan(self.target, "window", self.protocol, self.samples)[0], 0)

    def test_initial_checkpoint_gets_full_remaining_budget(self):
        self.save(epoch=0)
        self.assertEqual(training_plan(self.target, "window", self.protocol, self.samples), (10, self.target / "last.pt"))

    def test_early_stopping_without_summary_skipped(self):
        self.save(epoch=7, patience=3, best_epoch=4)
        self.assertEqual(training_plan(self.target, "window", self.protocol, self.samples)[0], 0)

    def test_success_summary_skipped(self):
        self.save()
        (self.target / "summary.json").write_text(json.dumps({"epochs_completed": 6}))
        self.assertEqual(training_plan(self.target, "window", self.protocol, self.samples)[0], 0)

    def test_mismatched_sampling_rejected(self):
        self.save(mode="family-group")
        with self.assertRaises(ValueError):
            training_plan(self.target, "window", self.protocol, self.samples)


if __name__ == "__main__":
    unittest.main()
