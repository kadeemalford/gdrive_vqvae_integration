from copy import deepcopy
from pathlib import Path
import unittest

from finetuning.run_robust_rate_continuation import validate_source


class RobustRateContinuationTests(unittest.TestCase):
    def source(self):
        return {"epoch": 24, "optimizer_state_dict": {}, "finetune": {
            "latent_geometry": "spatial4-temporal4", "normalization": "robust-escape-v1",
            "physical_validation": True, "learning_rate": 3e-5, "weight_decay": 1e-5,
            "steps_per_epoch": 4096, "sampling": "window", "early_stopping_metric": "macro-rmse",
            "trainable": "full", "samples_dir": "/data/fold_0", "rng_state": {},
            "spectral_weight": 0.0, "codebook_refresh": "none", "domain_family": None}}

    def test_source_policy_and_fold(self):
        saved = self.source()
        validate_source(saved, "spatial4-temporal4", Path("/data/fold_0"))
        for key, value in (("normalization", "escape-v1"), ("latent_geometry", "spatial-4x"),
                           ("samples_dir", "/data/fold_1"), ("spectral_weight", 0.1),
                           ("codebook_refresh", "enabled"), ("steps_per_epoch", 100)):
            changed = deepcopy(saved)
            changed["finetune"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_source(changed, "spatial4-temporal4", Path("/data/fold_0"))

    def test_epoch_and_resume_state_required(self):
        for key in ("epoch", "optimizer_state_dict", "rng_state"):
            saved = self.source()
            if key == "epoch":
                saved[key] = 23
            elif key == "rng_state":
                del saved["finetune"][key]
            else:
                del saved[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_source(saved, "spatial4-temporal4", Path("/data/fold_0"))
