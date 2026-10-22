import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from finetuning.latent_geometry import apply_geometry, checkpoint_geometry, latent_positions, require_standard_geometry


class LatentGeometryTests(unittest.TestCase):
    def test_balanced_grid_matches_temporal_rate_and_spatial_hooks_do_not_stack(self):
        with patch.object(sys, "path", ["/home/murilo-barroso/source/smartgrids", *sys.path]):
            from vqvae.train_vqvae import VQVAE
        model = VQVAE(1, 8, 8, 16, 4, .25).eval()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        for channels in (1, 17, 45):
            tensor = torch.randn(1, 1, channels, 65)
            counts = {}
            with torch.no_grad():
                for policy in ("temporal-4x", "balanced-2x", "balanced-2x", "spatial-4x", "spatial4-temporal2", "spatial4-temporal4", "standard", "temporal-4x"):
                    apply_geometry(model, policy)
                    decoded, _, indices, _ = model(tensor)
                    self.assertEqual(decoded.shape, tensor.shape)
                    self.assertEqual(indices.numel(), latent_positions(model, channels, 65))
                    counts[policy] = indices.numel()
            self.assertEqual(counts["temporal-4x"], counts["balanced-2x"])
        self.assertTrue(all(torch.equal(value, original[key]) for key, value in model.state_dict().items()))

    def test_rate_allocation_real_window_counts(self):
        model = SimpleNamespace(downsample_levels=4)
        for channels in (23, 45, 70, 512):
            model.latent_geometry = "temporal-4x"
            reference = latent_positions(model, channels, 1000)
            model.latent_geometry = "balanced-2x"
            self.assertEqual(latent_positions(model, channels, 1000), reference)
            model.latent_geometry = "spatial-4x"
            self.assertAlmostEqual(latent_positions(model, channels, 1000) / reference, 1.008)

    def test_higher_rates_preserve_spatial_grid_and_count_extra_indices(self):
        model = SimpleNamespace(downsample_levels=4)
        for channels in (23, 45, 70, 512):
            model.latent_geometry = "temporal-4x"
            base = latent_positions(model, channels, 1000)
            for policy, multiplier in (("spatial4-temporal2", 2), ("spatial4-temporal4", 4)):
                model.latent_geometry = policy
                self.assertEqual(latent_positions(model, channels, 1000), base * multiplier)

    def test_shape_and_actual_index_counts_using_native_small_model(self):
        with patch.object(sys, "path", ["/home/murilo-barroso/source/smartgrids", *sys.path]):
            from vqvae.train_vqvae import VQVAE
        model = VQVAE(1, 8, 8, 16, 4, .25).eval()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        tensor = torch.randn(1, 1, 17, 65)
        observed = []
        with torch.no_grad():
            for policy in ("standard", "temporal-2x", "temporal-4x", "standard"):
                apply_geometry(model, policy)
                restored, _, indices, _ = model(tensor)
                self.assertEqual(restored.shape, tensor.shape)
                self.assertEqual(indices.numel(), latent_positions(model, 17, 65))
                observed.append(indices.numel())
                self.assertTrue(all(torch.equal(value, original[key]) for key, value in model.state_dict().items()))
        self.assertEqual(observed, [10, 18, 36, 10])

    def test_real_window_rates_include_padding_and_kernel_offsets(self):
        model = SimpleNamespace(downsample_levels=4)
        self.assertEqual(latent_positions(model, 512, 1000), 2016)
        model.latent_geometry = "temporal-2x"
        self.assertEqual(latent_positions(model, 512, 1000), 4000)
        model.latent_geometry = "temporal-4x"
        self.assertEqual(latent_positions(model, 512, 1000), 8000)

    def test_conflicting_metadata_and_old_tools_rejected(self):
        checkpoint = {"config": {"latent_geometry": "temporal-4x"}, "finetune": {"latent_geometry": "standard"}}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            checkpoint_geometry(checkpoint)
        with self.assertRaisesRegex(ValueError, "historical tool"):
            require_standard_geometry({"config": {"latent_geometry": "temporal-2x"}})
        self.assertEqual(checkpoint_geometry({}), "standard")
