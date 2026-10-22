import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from finetuning.physical_validation import restore_for_validation
from finetuning.channel_normalization import encode_channel
from finetuning.reversible_normalization import encode
from finetuning.robust_escape import encode_robust
from finetuning.finetune_vqvae import load_sample, load_valid_mask


class PhysicalValidationTests(unittest.TestCase):
    def test_robust_loader_mask_and_reference_agree(self):
        raw = np.random.default_rng(4).normal(size=(512, 1000)).astype(np.float32) * 1e5
        raw[-2:] = 1e16
        packet = encode_robust(raw)
        model = SimpleNamespace(downsample_levels=4, num_embeddings=2048, latent_geometry="spatial4-temporal4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, data=raw)
            values = load_sample(path, normalization="robust-escape-v1")
            valid = load_valid_mask(path, normalization="robust-escape-v1")
            target, prediction, extras = restore_for_validation(path, values, "robust-escape-v1", model, detailed_channels=True)
        np.testing.assert_array_equal(valid, packet.valid)
        np.testing.assert_array_equal(target, values)
        np.testing.assert_allclose(prediction, values, atol=1e-6)
        self.assertLess(extras["ordinary_channel_p90_relative_rmse"], 1e-6)
        self.assertEqual(extras["side_bits_per_cell"], 8 * len(packet.side_stream) / raw.size)
    def test_common_target_and_charged_side_information(self):
        raw = np.random.default_rng(1).normal(size=(4, 1000)).astype(np.float32)
        raw *= np.array([1, 10, 100, 1000], np.float32)[:, None]
        raw[3, 0] = 1e30
        model = SimpleNamespace(downsample_levels=4, num_embeddings=2048)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, data=raw)
            a = restore_for_validation(path, encode(raw).values, "escape-v1", model)
            b = restore_for_validation(path, encode_channel(raw).values, "channel-escape-v1", model)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], atol=2e-7)
        self.assertGreater(b[2]["side_bits_per_cell"], a[2]["side_bits_per_cell"])
        self.assertLess(b[2]["index_plus_side_nominal_ratio"], a[2]["index_plus_side_nominal_ratio"])

    def test_detailed_channels_opt_in_preserves_training_metric_schema(self):
        raw = np.random.default_rng(2).normal(size=(4, 1000)).astype(np.float32)
        model = SimpleNamespace(downsample_levels=4, num_embeddings=2048)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(path, data=raw)
            default = restore_for_validation(path, encode(raw).values, "escape-v1", model)
            detailed = restore_for_validation(path, encode(raw).values, "escape-v1", model, detailed_channels=True)
        self.assertNotIn("ordinary_channel_median_relative_rmse", default[2])
        self.assertIn("ordinary_channel_median_relative_rmse", detailed[2])
        np.testing.assert_array_equal(default[1], detailed[1])
