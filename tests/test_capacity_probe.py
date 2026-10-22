import csv
import tempfile
import unittest
from pathlib import Path

import torch

from finetuning.run_capacity_probe import atomic_save, nominal_rates, probe_forward, select_cases


class CapacityTests(unittest.TestCase):
    def test_atomic_checkpoint_replaces_file_without_leaving_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe_last.pt"
            atomic_save({"step": 0}, path)
            atomic_save({"step": 100}, path)
            self.assertEqual(torch.load(path, weights_only=False)["step"], 100)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_continuous_forward_has_encoder_decoder_gradients_not_quantizer(self):
        class Model(torch.nn.Module):
            downsample_levels = 2
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Conv2d(1, 2, 1)
                self.decoder = torch.nn.Conv2d(2, 1, 1)
                self.quantizer = torch.nn.Linear(2, 2)
        model = Model()
        target = torch.randn(1, 1, 3, 11)
        recon, vq, indices, perplexity = probe_forward(model, target, "continuous")
        self.assertEqual(recon.shape, target.shape)
        recon.square().mean().backward()
        self.assertIsNotNone(model.encoder.weight.grad)
        self.assertIsNotNone(model.decoder.weight.grad)
        self.assertIsNone(model.quantizer.weight.grad)
        self.assertEqual(vq.item(), 0)
        self.assertIsNone(indices)
        self.assertIsNone(perplexity)

    def test_nominal_rate_includes_padding_and_does_not_equate_continuous_codec(self):
        rates = nominal_rates(23, 1000, 4, 2048, 128)
        self.assertEqual(rates["latent_positions"], 2 * 63)
        self.assertEqual(rates["index_bits"], 11)
        self.assertGreater(rates["vq_index_only_ratio"], rates["continuous_float32_latent_ratio"])

    def test_only_train_metadata_used_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            rows = [{"family": f, "configuration": "one", "drive_id": f, "sample_path": f + str(i), "split": "train"}
                    for f in ("doped", "multicore", "multimode", "single_mode") for i in range(4)]
            rows.append({"family": "doped", "configuration": "held", "drive_id": "held", "sample_path": "held", "split": "test"})
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            first = select_cases(path, 2026)
            self.assertEqual(first, select_cases(path, 2026))
            self.assertEqual(len(first), 4)
            self.assertTrue(all(row["split"] == "train" for row in first))
            self.assertTrue(all(row["sample_path"] != "held" for row in first))


if __name__ == "__main__":
    unittest.main()
