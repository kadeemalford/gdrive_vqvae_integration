import csv
import tempfile
import unittest
from pathlib import Path

import torch

from finetuning.reconstruction_objective import (aggregate_validation, multiresolution_spectral_loss,
                                                path_fingerprint, stratified_validation, validation_metadata)


class ObjectiveTests(unittest.TestCase):
    def test_spectral_identity_zero_with_finite_gradient(self):
        target = torch.randn(1, 1, 3, 1000)
        recon = target.clone().requires_grad_()
        loss = multiresolution_spectral_loss(recon, target)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(recon.grad).all())

    def test_attenuation_positive_and_differentiable(self):
        target = torch.randn(1, 1, 3, 1000)
        recon = (target * 0.5).requires_grad_()
        loss = multiresolution_spectral_loss(recon, target)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(recon.grad).all())

    def test_silence_is_finite(self):
        recon = torch.zeros(1, 1, 2, 512, requires_grad=True)
        loss = multiresolution_spectral_loss(recon, torch.zeros_like(recon))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(recon.grad).all())

    def test_short_input_rejected(self):
        with self.assertRaises(ValueError):
            multiresolution_spectral_loss(torch.zeros(1, 1, 2, 64), torch.zeros(1, 1, 2, 64))

    def test_macro_weights_families_not_window_counts(self):
        rows = [{"family": "a", "configuration": "a1", "relative_rmse": 1., "correlation": 0.} for _ in range(100)]
        rows += [{"family": "a", "configuration": "a2", "relative_rmse": 0., "correlation": 1.},
                 {"family": "b", "configuration": "b1", "relative_rmse": 0., "correlation": 1.}]
        macro, groups = aggregate_validation(rows)
        self.assertEqual(macro["macro_relative_rmse"], 0.25)
        self.assertEqual(macro["macro_correlation"], 0.75)
        self.assertEqual(macro["worst_family_relative_rmse"], 0.5)
        self.assertEqual(len(groups), 3)

    def test_stratified_selection_deterministic_per_file(self):
        paths = [Path(f"a{i}") for i in range(20)] + [Path("b")]
        metadata = {p: {"drive_id": "b" if str(p) == "b" else "a"} for p in paths}
        a = stratified_validation(paths, metadata, 2, 2026)
        b = stratified_validation(paths, metadata, 2, 2026)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        self.assertIn(Path("b"), a)
        self.assertEqual(path_fingerprint(a), path_fingerprint(list(reversed(a))))

    def test_test_window_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "samples_manifest.csv"
            with manifest.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_path", "split"])
                writer.writeheader()
                writer.writerow({"sample_path": "bad", "split": "test"})
            with self.assertRaises(ValueError):
                validation_metadata([Path("bad")], manifest)


if __name__ == "__main__":
    unittest.main()
