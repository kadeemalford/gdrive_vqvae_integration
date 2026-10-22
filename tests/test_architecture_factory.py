import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from finetuning.latent_geometry import latent_positions
from finetuning.model_factory import architecture, new_model, build_model
from finetuning.run_architecture_initialization_experiment import initialize_codebook


class ArchitectureFactoryTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(base_channels=8, embedding_dim=8, num_embeddings=16, downsample_levels=4,
                           commitment_cost=.25, latent_geometry="temporal-4x")
        self.path_patch = patch.object(sys, "path", ["/home/murilo-barroso/source/smartgrids", *sys.path])
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def test_parameters_grid_and_gradient_are_matched(self):
        tensor = torch.randn(1, 1, 17, 65)
        counts = []
        for name in ("native-v1", "temporal-dilated-v1"):
            config = {**self.config, "architecture": name}
            model = new_model(config, "cpu")
            counts.append(sum(p.numel() for p in model.parameters()))
            recon, vq, indices, _ = model(tensor)
            self.assertEqual(recon.shape, tensor.shape)
            self.assertEqual(indices.numel(), latent_positions(model, 17, 65))
            (torch.square(recon - tensor).mean() + vq).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(counts[0], counts[1])

    def test_alternative_checkpoint_roundtrip_restores_architecture_and_stride(self):
        config = {**self.config, "architecture": "temporal-dilated-v1"}
        model = new_model(config, "cpu").eval()
        tensor = torch.randn(1, 1, 17, 65)
        saved = {"config": config, "model_state_dict": model.state_dict()}
        restored = build_model(config, saved, "cpu")
        with torch.no_grad():
            torch.testing.assert_close(model(tensor)[0], restored(tensor)[0], rtol=0, atol=0)
        self.assertEqual(restored.latent_geometry, "temporal-4x")

    def test_unknown_architecture_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            architecture({"architecture": "unspecified"})

    def test_checkpoint_restores_spatial_padding_without_manual_hook(self):
        tensor = torch.randn(1, 1, 17, 65)
        for geometry in ("balanced-2x", "spatial-4x", "spatial4-temporal2", "spatial4-temporal4"):
            config = {**self.config, "latent_geometry": geometry}
            model = new_model(config, "cpu").eval()
            restored = build_model(config, {"config": config, "model_state_dict": model.state_dict()}, "cpu")
            with torch.no_grad():
                torch.testing.assert_close(model(tensor)[0], restored(tensor)[0], rtol=0, atol=0)
                self.assertEqual(restored(tensor)[2].numel(), latent_positions(restored, 17, 65))

    def test_wrong_factory_cannot_silently_load_alternative(self):
        config = {**self.config, "architecture": "temporal-dilated-v1"}
        checkpoint = {"config": config}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            build_model(self.config, checkpoint, "cpu")

    def test_training_reservoir_is_deterministic_and_does_not_update_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            np.savez(path, data=np.random.default_rng(7).normal(size=(17, 65)).astype(np.float32))
            embeddings = []
            for _ in range(2):
                torch.manual_seed(2026)
                model = new_model(self.config, "cpu")
                before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
                count = initialize_codebook(model, [path], 2026)
                self.assertGreaterEqual(count, model.num_embeddings)
                self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.encoder.state_dict().items()))
                embeddings.append(model.quantizer.embedding.weight.detach().clone())
            torch.testing.assert_close(embeddings[0], embeddings[1], rtol=0, atol=0)
