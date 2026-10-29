"""Regression tests for checkpoint-aware reconstruction and side streams."""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

from compression.encode_decode_samples import (
    encode_and_decode, normalize, prepare_chunk, restore_chunk,
)
from finetuning.latent_geometry import apply_geometry, latent_positions


class InferencePoliciesTest(unittest.TestCase):
    def test_legacy_is_unchanged(self):
        raw = np.arange(72, dtype=np.float32).reshape(6, 12)
        expected, mean, std = normalize(raw)
        values, side = prepare_chunk(raw, "per-time-chunk")
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(restore_chunk(values, side, "per-time-chunk"),
                                      expected * (std + 1e-6) + mean)
        self.assertEqual(len(side), 16)

    def test_escape_roundtrips(self):
        raw = np.arange(72, dtype=np.float32).reshape(6, 12)
        raw[0, 0] = 1e25
        for policy in ("escape-v1", "robust-escape-v1"):
            with self.subTest(policy=policy):
                values, side = prepare_chunk(raw, policy)
                restored = restore_chunk(values, side, policy)
                self.assertEqual(raw[0, 0].tobytes(), restored[0, 0].tobytes())
                np.testing.assert_allclose(restored, raw, atol=1e-5)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            prepare_chunk(np.ones((4, 4), dtype=np.float32), "unknown")

    def test_dense_index_count_and_output_shape(self):
        repo = Path("/home/murilo-barroso/source/smartgrids")
        if not repo.is_dir():
            self.skipTest("External VQVAE repository unavailable")
        sys.path.insert(0, str(repo))
        from vqvae.train_vqvae import VQVAE
        model = VQVAE(in_channels=1, base_channels=8, embedding_dim=8,
                      num_embeddings=8, downsample_levels=4, commitment_cost=.25).eval()
        for policy in ("standard", "spatial4-temporal4"):
            apply_geometry(model, policy)
            with torch.no_grad():
                reconstruction, indices = encode_and_decode(model, torch.zeros(1, 1, 31, 40))
            self.assertEqual(tuple(reconstruction.shape), (1, 1, 31, 40))
            self.assertEqual(indices.numel(), latent_positions(model, 31, 40))

    def test_extraction_removes_only_invalid_tail_and_preserves_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with h5py.File(root / "raw.h5", "w") as handle:
                group = handle.create_group("zone")
                raw = np.ones((1, 16, 4), dtype=np.float32)
                raw[:, :, -1] = np.nan
                group.create_dataset("Strain Rate", data=raw)
                group.attrs["Spacing"] = [.2, .05]
                group.attrs["Origin"] = [0., 0.]
                group.attrs["Extent"] = [100, 103, 0, 15]
            config = {"files": [{"id": "fixture", "local_filename": "raw.h5"}], "time_samples": 16,
                      "source_block_samples": 16, "source_distance_start_m": 20.,
                      "source_distance_end_m": 20.6}
            (root / "config.json").write_text(json.dumps(config))
            script = Path(__file__).resolve().parents[1] / "acquisition/extract_febus_windows.py"
            subprocess.run([sys.executable, str(script), "--config", str(root / "config.json"),
                            "--raw-dir", str(root), "--output-dir", str(root / "samples"),
                            "--manifest-output", str(root / "manifest.json"),
                            "--drop-nonfinite-channels", "--spatial-band-policy", "none"], check=True, capture_output=True)
            with np.load(root / "samples/raw.npz", allow_pickle=False) as packet:
                self.assertEqual(packet["data"].shape, (1, 3, 16))
                metadata = json.loads(str(packet["metadata_json"].item()))
                self.assertEqual(metadata["retained_source_channel_indices"], [0, 1, 2])
                self.assertEqual(metadata["removed_nonfinite_source_channel_indices"], [3])
                self.assertAlmostEqual(metadata["distance_start_m"], 20.)
                self.assertAlmostEqual(metadata["distance_end_m"], 20.4)


if __name__ == "__main__":
    unittest.main()
