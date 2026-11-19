import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from finetuning.development_validation import load_validation, scales_for_rows


class DevelopmentValidationTests(unittest.TestCase):
    def test_nonvalidation_waveforms_rejected_before_file_access(self):
        for split in ("train", "test"):
            with patch.object(Path, "read_bytes") as reader:
                with self.assertRaises(ValueError):
                    load_validation(Path("unused"), {"split": split})
                reader.assert_not_called()

    def test_native_shape_provenance_and_frozen_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "val").mkdir()
            raw = np.ones((3, 17), np.float32)
            np.savez(root / "val/sample.npz", data=raw, metadata_json=json.dumps({"split": "val"}))
            row = {"split": "val", "sample_path": "irrelevant/val/sample.npz", "channels": "3", "time_samples": "17"}
            np.testing.assert_array_equal(load_validation(root, row), raw)
            with self.assertRaises(ValueError):
                load_validation(root, row, "0" * 64)
            with self.assertRaises(ValueError):
                load_validation(root, {**row, "channels": "2"})

    def test_unseen_configuration_uses_training_family_not_its_waveform(self):
        calibration = {"scales": {"known/a": 3.}, "family_scales_for_unseen_configurations": {"known": 2.}}
        scales, provenance = scales_for_rows([{"group_id": "known/new", "family": "known"}], calibration)
        self.assertEqual(scales, {"known/new": 2.})
        self.assertEqual(provenance["known/new"], "frozen training family median")
