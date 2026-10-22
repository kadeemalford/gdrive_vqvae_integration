import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from finetuning.prepare_finetune_samples import main as prepare
from finetuning.prune_obsolete_artifacts import prune, targets
from finetuning.restore_pruned_samples import restore
from finetuning.run_sampling_experiment import check_splits


class ArtifactPruningTests(unittest.TestCase):
    def test_prune_and_recover_npz_preserves_raw_manifest_and_protected_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "data/finetune/raw"
            raw_dir.mkdir(parents=True)
            with h5py.File(raw_dir / "id.h5", "w") as handle:
                dataset = handle.create_dataset("Source/Zone/Strain Rate", data=np.arange(32, dtype=np.float32).reshape(1, 8, 4))
                dataset.parent.attrs["Spacing"] = [.2, .05]
            manifest = root / "source.csv"
            manifest.write_text("drive_id,family,configuration,fold,title\nid,doped,one,2,example.h5\n")
            folder = root / "data/finetune/fold_0"
            args = ["prepare", "--manifest", str(manifest), "--raw-dir", str(raw_dir), "--output-dir", str(folder),
                    "--time-samples", "4", "--windows-per-block", "2"]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                prepare()
            samples = sorted((folder / "train").glob("*.npz"))
            contents = {}
            for sample in samples:
                with np.load(sample, allow_pickle=False) as payload:
                    contents[sample] = (payload["data"].copy(), str(payload["metadata_json"]))
            protected = [root / "data/finetune/fold_1/train/keep.npz", root / "models/base/best.pt",
                         root / "results/experiments/temporal_density_pilot_20261008/temporal-4x/last.pt"]
            for path in protected:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"keep")
            audit = root / "results/maintenance/cleanup"
            summary = prune(root, audit)
            self.assertEqual(summary["removed_files"]["prepared_npz"], 2)
            self.assertTrue(all(not path.exists() for path in samples))
            self.assertTrue((raw_dir / "id.h5").exists())
            self.assertTrue((folder / "samples_manifest.csv").exists())
            self.assertTrue(all(path.read_bytes() == b"keep" for path in protected))
            with self.assertRaisesRegex(ValueError, "pruned/incomplete"):
                check_splits(folder, 0)
            result = restore(root, folder)
            self.assertEqual(result["restored"], 2)
            self.assertEqual(result["missing"], 0)
            for path, (values, metadata) in contents.items():
                with np.load(path, allow_pickle=False) as payload:
                    np.testing.assert_array_equal(payload["data"], values)
                    self.assertEqual(str(payload["metadata_json"]), metadata)
            check_splits(folder, 0)

    def test_missing_raw_prevents_deleting_prepared_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "data/finetune/fold_0"
            folder.mkdir(parents=True)
            (folder / "samples_manifest.csv").write_text("sample_path,drive_id\ndata/finetune/fold_0/train/a.npz,missing\n")
            with self.assertRaisesRegex(ValueError, "without raw source"):
                targets(root)

    def test_incomplete_old_experiment_never_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "results/experiments/spectral_pilot_20261007"
            folder.mkdir(parents=True)
            (folder / "last.pt").write_bytes(b"not-complete")
            with self.assertRaisesRegex(ValueError, "not complete"):
                targets(root)
            self.assertEqual((folder / "last.pt").read_bytes(), b"not-complete")
