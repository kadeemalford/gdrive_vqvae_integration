import tempfile
import unittest
from pathlib import Path
import torch

from finetuning.run_temporal_density_long import remaining_epochs


class HighRateBudgetTests(unittest.TestCase):
    def test_continuation_respects_remaining_40_epoch_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in ("best.pt", "best_macro.pt"):
                (target / name).touch()
            state = {"config": {"latent_geometry": "spatial4-temporal4"}, "epoch": 24,
                     "finetune": {"latent_geometry": "spatial4-temporal4", "physical_validation": True,
                                  "normalization": "escape-v1", "early_stopping_metric": "macro-rmse",
                                  "epochs_without_improvement": 0}}
            torch.save(state, target / "last.pt")
            protocol = {"epochs": 40, "patience": 41}
            self.assertEqual(remaining_epochs(target, protocol, "spatial4-temporal4")[0], 16)
            state["epoch"] = 40
            torch.save(state, target / "last.pt")
            self.assertEqual(remaining_epochs(target, protocol, "spatial4-temporal4")[0], 0)
