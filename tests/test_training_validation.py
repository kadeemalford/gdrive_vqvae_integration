import contextlib
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from finetuning.finetune_vqvae import main, save_checkpoint, configure_trainable


class IdentityTrainable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, values):
        return values * self.weight, self.weight * 0, torch.zeros(1, dtype=torch.long), self.weight * 0 + 1


class TrainingValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.samples = root / "samples"
        for split in ("train", "val"):
            (self.samples / split).mkdir(parents=True)
            np.savez(self.samples / split / "sample.npz", data=np.random.default_rng(1).normal(size=(4, 512)).astype(np.float32))
        with (self.samples / "samples_manifest.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_path", "split", "family", "configuration", "drive_id"])
            writer.writeheader()
            writer.writerow(dict(sample_path=str(self.samples / "val/sample.npz"), split="val", family="a", configuration="one", drive_id="1"))
        self.output = root / "run"
        package = types.ModuleType("vqvae")
        package.__path__ = []
        module = types.ModuleType("vqvae.evaluate_vqvae")
        module.load_checkpoint_config = lambda *args: ({"model_state_dict": IdentityTrainable().state_dict(), "epoch": 30}, {})
        module.build_model = lambda *args: IdentityTrainable()
        self.modules = {"vqvae": package, "vqvae.evaluate_vqvae": module}
        self.argv = ["finetune", "--repo-root", str(root), "--checkpoint", str(root / "base.pt"),
                     "--samples-dir", str(self.samples), "--output-dir", str(self.output),
                     "--epochs", "1", "--steps-per-epoch", "1", "--device", "cpu",
                     "--validation-details", "--evaluate-initial"]

    def invoke(self, extra=()):
        # Restore only our stubs, not modules lazily imported by PyTorch's optimizer.
        # Reimporting removed torch.library modules would register operators twice.
        previous = {key: sys.modules.get(key) for key in self.modules}
        sys.modules.update(self.modules)
        try:
            with patch.object(sys, "argv", self.argv + list(extra)), patch.object(sys, "path", list(sys.path)):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return main()
        finally:
            for key, value in previous.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_initial_checkpoint_retained_and_resume_appends_metrics(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(torch.load(self.output / "best_macro.pt", weights_only=False)["epoch"], 0)
        self.assertEqual(torch.load(self.output / "last.pt", weights_only=False)["epoch"], 1)
        self.assertEqual(self.invoke(["--resume", str(self.output / "last.pt")]), 0)
        with (self.output / "validation_metrics.csv").open() as stream:
            self.assertEqual([r["epoch"] for r in csv.DictReader(stream)], ["0", "1", "2"])
        with (self.output / "metrics.csv").open() as stream:
            self.assertEqual([r["epoch"] for r in csv.DictReader(stream)], ["1", "2"])
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["best_macro_epoch"], 0)
        self.assertEqual(summary["epochs_completed"], 2)

    def test_resume_rejects_changed_objective(self):
        self.invoke()
        with self.assertRaisesRegex(ValueError, "spectral weight"):
            self.invoke(["--resume", str(self.output / "last.pt"), "--spectral-weight", "0.1"])

    def test_robust_training_policy_roundtrip_and_resume(self):
        def physical_model(*args):
            model = IdentityTrainable()
            model.downsample_levels = 4
            model.num_embeddings = 16
            return model
        self.modules["vqvae.evaluate_vqvae"].build_model = physical_model
        flags = ["--normalization", "robust-escape-v1", "--physical-validation", "--channel-diagnostics"]
        self.assertEqual(self.invoke(flags), 0)
        checkpoint = torch.load(self.output / "last.pt", weights_only=False)
        self.assertEqual(checkpoint["finetune"]["normalization"], "robust-escape-v1")
        self.assertEqual(self.invoke(flags + ["--resume", str(self.output / "last.pt")]), 0)

    def test_channel_objective_roundtrip_and_resume_guard(self):
        def physical_model(*args):
            model = IdentityTrainable()
            model.downsample_levels = 4
            model.num_embeddings = 16
            return model
        self.modules["vqvae.evaluate_vqvae"].build_model = physical_model
        flags = ["--normalization", "robust-escape-v1", "--physical-validation", "--channel-mse-weight", "0.05"]
        self.assertEqual(self.invoke(flags), 0)
        state = torch.load(self.output / "last.pt", weights_only=False)
        self.assertEqual(state["finetune"]["channel_mse_weight"], .05)
        self.assertEqual(self.invoke(flags + ["--resume", str(self.output / "last.pt")]), 0)
        with self.assertRaisesRegex(ValueError, "channel MSE objective"):
            self.invoke(["--normalization", "robust-escape-v1", "--physical-validation", "--resume", str(self.output / "last.pt")])

    def test_zero_channel_weight_preserves_historical_update_and_skips_auxiliary(self):
        with patch("finetuning.finetune_vqvae.bounded_channel_mse", side_effect=AssertionError("must not execute")):
            self.assertEqual(self.invoke(), 0)
            baseline = torch.load(self.output / "last.pt", weights_only=False)
            other = self.output.parent / "explicit_zero"
            self.assertEqual(self.invoke(["--output-dir", str(other), "--channel-mse-weight", "0"]), 0)
            zero = torch.load(other / "last.pt", weights_only=False)
        for key in baseline["model_state_dict"]:
            self.assertTrue(torch.equal(baseline["model_state_dict"][key], zero["model_state_dict"][key]))

    def test_family_branch_filters_and_resume_cannot_change_domain(self):
        with (self.samples / "samples_manifest.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_path", "split", "family", "configuration", "drive_id"])
            writer.writeheader()
            for split in ("train", "val"):
                writer.writerow(dict(sample_path=str(self.samples / split / "sample.npz"), split=split,
                                     family="doped", configuration="one", drive_id=split))
        self.assertEqual(self.invoke(["--domain-family", "doped"]), 0)
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["domain_family"], "doped")
        with self.assertRaisesRegex(ValueError, "domain family"):
            self.invoke(["--resume", str(self.output / "last.pt")])

    def test_resume_rejects_refresh_policy_change(self):
        self.invoke()
        path = self.output / "last.pt"
        state = torch.load(path, weights_only=False)
        state["finetune"]["codebook_refresh"] = "dead-v1"
        save_checkpoint(state, path)
        with self.assertRaisesRegex(ValueError, "refresh policy"):
            self.invoke(["--resume", str(path)])

    def test_refresh_requires_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "requires codebook-diagnostics"):
            self.invoke(["--codebook-refresh", "dead-v1"])

    def test_resume_rejects_changed_diagnostic_policy(self):
        self.invoke()
        with self.assertRaisesRegex(ValueError, "codebook_diagnostics"):
            self.invoke(["--resume", str(self.output / "last.pt"), "--codebook-diagnostics"])

    def test_codebook_diagnostics_are_recorded_and_resume_preserved(self):
        def diagnostic_model(*args):
            model = IdentityTrainable()
            model.num_embeddings = 16
            return model
        self.modules["vqvae.evaluate_vqvae"].build_model = diagnostic_model
        self.assertEqual(self.invoke(["--codebook-diagnostics"]), 0)
        self.assertEqual(self.invoke(["--codebook-diagnostics", "--resume", str(self.output / "last.pt")]), 0)
        with (self.output / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(int(rows[0]["train_unique_codes"]), 1)
        self.assertEqual(float(rows[0]["train_code_usage"]), 1 / 16)
        self.assertEqual(float(rows[0]["train_mean_perplexity"]), 1.)

    def test_resume_rejects_changed_architecture(self):
        self.invoke()
        path = self.output / "last.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["config"]["architecture"] = "temporal-dilated-v1"
        save_checkpoint(state, path)
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.invoke(["--resume", str(path)])

    def test_escape_training_and_resume_policy(self):
        for split in ("train", "val"):
            raw = np.random.default_rng(3).normal(size=(4, 512)).astype(np.float32)
            raw[-1, 10] = 1e30
            np.savez(self.samples / split / "sample.npz", data=raw)
        self.assertEqual(self.invoke(["--normalization", "escape-v1"]), 0)
        state = torch.load(self.output / "last.pt", weights_only=False)
        self.assertEqual(state["finetune"]["normalization"], "escape-v1")
        with (self.output / "validation_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(float(rows[0]["macro_relative_rmse"]), 0)
        self.assertGreater(float(rows[0]["macro_escaped_cell_fraction"]), 0)
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.invoke(["--resume", str(self.output / "last.pt")])
        self.assertEqual(self.invoke(["--normalization", "escape-v1", "--resume", str(self.output / "last.pt")]), 0)

    def test_escape_rejects_spectral_objective(self):
        with self.assertRaisesRegex(ValueError, "spectral-weight"):
            self.invoke(["--normalization", "escape-v1", "--spectral-weight", "0.1"])

    def test_atomic_save_failure_preserves_existing_checkpoint(self):
        path = Path(self.directory.name) / "checkpoint.pt"
        save_checkpoint({"epoch": 1}, path)
        original = path.read_bytes()
        with patch("finetuning.finetune_vqvae.torch.save", side_effect=RuntimeError("disk error")):
            with self.assertRaisesRegex(RuntimeError, "disk error"):
                save_checkpoint({"epoch": 2}, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(path.parent.glob("checkpoint.pt.*.tmp")))

    def test_detailed_metrics_ignore_escape_predictions(self):
        from finetuning.reconstruction_objective import evaluate_detailed
        from finetuning.finetune_vqvae import load_sample, load_valid_mask
        from functools import partial
        path = self.samples / "val/sample.npz"
        raw = np.random.default_rng(4).normal(size=(4, 512)).astype(np.float32)
        raw[-1, 10] = 1e30
        np.savez(path, data=raw)

        class WrongEscapeOnly(IdentityTrainable):
            def forward(self, values):
                recon = values.clone()
                recon[0, 0, -1, 10] = 100
                return recon, self.weight * 0, None, None

        macro, _ = evaluate_detailed(WrongEscapeOnly(), [path], {path: {"family": "a", "configuration": "b"}},
                                    torch.device("cpu"), partial(load_sample, normalization="escape-v1"),
                                    load_valid_mask=load_valid_mask)
        self.assertEqual(macro["macro_relative_rmse"], 0)
        self.assertEqual(macro["temporal_loss"], 0)

    def test_decoder_only_keeps_encoder_and_quantizer_bit_exact(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(4, 4)
        model.quantizer = torch.nn.Linear(4, 4)
        model.decoder = torch.nn.Linear(4, 4)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        parameters = configure_trainable(model, "decoder")
        optimizer = torch.optim.AdamW(parameters, lr=.01)
        values = torch.ones(1, 4)
        model.decoder(model.quantizer(model.encoder(values))).square().mean().backward()
        optimizer.step()
        for key, value in model.state_dict().items():
            if not key.startswith("decoder."):
                self.assertTrue(torch.equal(value, initial[key]))
        self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in model.state_dict().items() if key.startswith("decoder.")))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_macro_stop_and_resume_guards(self):
        self.assertEqual(self.invoke(["--early-stopping-metric", "macro-rmse"]), 0)
        with self.assertRaisesRegex(ValueError, "early-stopping metric"):
            self.invoke(["--resume", str(self.output / "last.pt")])
        self.assertEqual(self.invoke(["--early-stopping-metric", "macro-rmse", "--resume", str(self.output / "last.pt")]), 0)

    def test_resume_rejects_changed_trainable_policy(self):
        self.invoke()
        saved = torch.load(self.output / "last.pt", weights_only=False)
        saved["finetune"]["trainable"] = "decoder"
        torch.save(saved, self.output / "last.pt")
        with self.assertRaisesRegex(ValueError, "trainable policy"):
            self.invoke(["--resume", str(self.output / "last.pt")])

    def test_channel_training_uses_physical_metrics_and_resume_guard(self):
        def build(*args):
            model = IdentityTrainable()
            model.downsample_levels = 4
            model.num_embeddings = 2048
            return model
        self.modules["vqvae.evaluate_vqvae"].build_model = build
        self.assertEqual(self.invoke(["--normalization", "channel-escape-v1", "--physical-validation"]), 0)
        with (self.output / "validation_metrics.csv").open() as stream:
            row = next(csv.DictReader(stream))
        self.assertLess(float(row["macro_relative_rmse"]), 1e-6)
        self.assertGreater(float(row["macro_side_bits_per_cell"]), 0)
        saved = torch.load(self.output / "last.pt", weights_only=False)
        self.assertTrue(saved["finetune"]["physical_validation"])

    def test_channel_policy_rejects_incomparable_validation(self):
        with self.assertRaisesRegex(ValueError, "physical-validation"):
            self.invoke(["--normalization", "channel-escape-v1"])

    def test_validation_callback_receives_each_window(self):
        from finetuning.reconstruction_objective import evaluate_detailed
        from finetuning.finetune_vqvae import load_sample
        path = self.samples / "val/sample.npz"
        captured = []
        summary, _ = evaluate_detailed(IdentityTrainable(), [path], {path: {"family": "a", "configuration": "b"}},
                                      torch.device("cpu"), load_sample,
                                      on_sample=lambda p, row: captured.append((p, dict(row))))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], path)
        self.assertEqual(captured[0][1]["relative_rmse"], 0)
        self.assertEqual(captured[0][1]["val_loss"], summary["val_loss"])

    def test_resume_rejects_explicit_geometry_change(self):
        self.invoke()
        saved = torch.load(self.output / "last.pt", weights_only=False)
        saved["config"]["latent_geometry"] = "temporal-2x"
        saved["finetune"]["latent_geometry"] = "temporal-2x"
        torch.save(saved, self.output / "last.pt")
        with self.assertRaisesRegex(ValueError, "latent geometry"):
            self.invoke(["--latent-geometry", "standard", "--resume", str(self.output / "last.pt")])

    def test_native_dense_training_resume_inherits_geometry(self):
        with patch.object(sys, "path", ["/home/murilo-barroso/source/smartgrids", *sys.path]):
            from vqvae.train_vqvae import VQVAE
        config = {"base_channels": 8, "embedding_dim": 8, "num_embeddings": 16, "downsample_levels": 4}
        initial = VQVAE(1, 8, 8, 16, 4, .25).state_dict()
        module = self.modules["vqvae.evaluate_vqvae"]
        module.load_checkpoint_config = lambda *args: ({"model_state_dict": initial, "config": config}, config)
        def build(*args):
            model = VQVAE(1, 8, 8, 16, 4, .25)
            model.load_state_dict(initial)
            return model
        module.build_model = build
        common = ["--normalization", "escape-v1", "--physical-validation"]
        self.assertEqual(self.invoke([*common, "--latent-geometry", "temporal-2x"]), 0)
        self.assertEqual(self.invoke([*common, "--resume", str(self.output / "last.pt")]), 0)
        saved = torch.load(self.output / "last.pt", weights_only=False)
        self.assertEqual(saved["epoch"], 2)
        self.assertEqual(saved["config"]["latent_geometry"], "temporal-2x")
        self.assertEqual(saved["finetune"]["latent_geometry"], "temporal-2x")


if __name__ == "__main__":
    unittest.main()
