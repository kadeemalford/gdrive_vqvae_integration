import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from finetuning.run_temporal_density_long import remaining_epochs, require_cuda


class DensityLongResumeTests(unittest.TestCase):
    def test_robust_policy_requires_matching_continuation_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "temporal-4x"
            self.make_checkpoint(target, 5, 0)
            checkpoint = torch.load(target / "last.pt", weights_only=False)
            checkpoint["finetune"]["normalization"] = "robust-escape-v1"
            torch.save(checkpoint, target / "last.pt")
            with self.assertRaises(ValueError):
                remaining_epochs(target, {"epochs": 24, "patience": 25}, "temporal-4x")
            self.assertEqual(remaining_epochs(target, {"epochs": 24, "patience": 25, "normalization": "robust-escape-v1"}, "temporal-4x")[0], 19)
    def test_unavailable_cuda_fails_without_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is not automatic"):
                require_cuda()

    def test_available_cuda_preflight_passes(self):
        with patch("torch.cuda.is_available", return_value=True):
            require_cuda()

    def make_checkpoint(self, target, epoch, stale):
        target.mkdir()
        torch.save({"epoch": epoch, "config": {"latent_geometry": "temporal-4x"},
                    "finetune": {"latent_geometry": "temporal-4x", "physical_validation": True,
                                 "normalization": "escape-v1", "early_stopping_metric": "macro-rmse",
                                 "epochs_without_improvement": stale}}, target / "last.pt")
        (target / "best.pt").touch()
        (target / "best_macro.pt").touch()

    def test_remaining_budget_not_extended(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "temporal-4x"
            self.make_checkpoint(target, 5, 0)
            count, last = remaining_epochs(target, {"epochs": 8, "patience": 3}, "temporal-4x")
            self.assertEqual(count, 3)
            self.assertEqual(last, target / "last.pt")

    def test_interruption_after_early_stop_checkpoint_does_not_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "temporal-4x"
            self.make_checkpoint(target, 4, 3)
            self.assertEqual(remaining_epochs(target, {"epochs": 8, "patience": 3}, "temporal-4x")[0], 0)

    def test_geometry_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "temporal-4x"
            self.make_checkpoint(target, 2, 0)
            with self.assertRaisesRegex(ValueError, "geometry"):
                remaining_epochs(target, {"epochs": 8, "patience": 3}, "temporal-2x")
