import csv
from pathlib import Path
import tempfile
import unittest

from finetuning.run_guarded_decoder_pilot import trim_uncommitted_history, verify_phase0, weights_only, validate_training_source


class GuardedDecoderRunnerTests(unittest.TestCase):
    def test_confirmation_refuses_cross_fold_weights_or_changed_representation(self):
        saved = {"epoch": 24, "finetune": {"samples_dir": "/data/fold_1", "normalization": "escape-v1",
                                          "latent_geometry": "spatial4-temporal4"}}
        validate_training_source(saved, Path("/data/fold_1"), "escape-v1")
        with self.assertRaises(ValueError):
            validate_training_source(saved, Path("/data/fold_0"), "escape-v1")
        with self.assertRaises(ValueError):
            validate_training_source(saved, Path("/data/fold_1"), "robust-escape-v1")

    def test_weights_snapshot_cannot_resume_optimizer(self):
        saved = {"config": {}, "model_state_dict": {}, "epoch": 2, "optimizer_state_dict": {},
                 "finetune": {"rng_state": {}, "normalization": "robust-escape-v1"}}
        snapshot = weights_only(saved)
        self.assertNotIn("optimizer_state_dict", snapshot)
        self.assertNotIn("rng_state", snapshot["finetune"])
        self.assertIn("rng_state", saved["finetune"])

    def test_trim_preserves_interrupted_records_and_committed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            path = target / "metrics.csv"
            path.write_text("epoch,value\n1,a\n2,b\n3,c\n")
            trim_uncommitted_history(target, 2)
            with path.open() as stream:
                self.assertEqual([r["epoch"] for r in csv.DictReader(stream)], ["1", "2"])
            self.assertIn("3,c", (target / "metrics.csv.uncommitted_001").read_text())

    def test_phase0_matches_reference_without_using_admission_epsilon(self):
        base = {"family": "a", "configuration": "c", "windows": "10", "relative_rmse": ".5",
                "correlation": ".8", "ordinary_channel_p90_relative_rmse": ".9"}
        history = [{**base, "epoch": "0", "relative_rmse": ".50000001"}]
        self.assertEqual(len(verify_phase0([base], history)), 1)
        history[0]["relative_rmse"] = ".6"
        with self.assertRaises(ValueError):
            verify_phase0([base], history)
