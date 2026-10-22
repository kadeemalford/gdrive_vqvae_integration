import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from finetuning.prepare_finetune_samples import main
from finetuning.audit_channel_maps import scan_finite, read_window


class PreparationGeometryTests(unittest.TestCase):
    def test_new_preparation_has_map_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            raw = np.arange(48, dtype=np.float32).reshape(1, 8, 6)
            raw[0, 0, 1] = np.nan
            raw[0, 2, 4] = np.inf
            with h5py.File(raw_dir / "id.h5", "w") as handle:
                ds = handle.create_dataset("Source/Zone/Strain Rate", data=raw)
                ds.parent.attrs["Spacing"] = [.2, .05]
                ds.parent.attrs["Extent"] = [100, 105, 0, 7, 0, 0]
                ds.parent.attrs["Origin"] = [0, 0, 0]
                np.testing.assert_array_equal(scan_finite(ds, 3), np.array([[True, False, True, True, False, True]]))
                np.testing.assert_array_equal(read_window(ds, 0, 0, 4), raw[0, :4].T)
            manifest = root / "manifest.csv"
            manifest.write_text("drive_id,family,configuration,fold,title\nid,doped,one,2,example.h5\n")
            output = root / "prepared"
            argv = ["prepare", "--manifest", str(manifest), "--raw-dir", str(raw_dir), "--output-dir", str(output),
                    "--time-samples", "4", "--windows-per-block", "2", "--max-channels", "3"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            maps = [json.loads(line) for line in (output / "channel_maps.jsonl").read_text().splitlines()]
            self.assertEqual(len(maps), 1)
            self.assertEqual(maps[0]["source_channel_indices"], [0, 2, 3])
            samples = sorted((output / "train").glob("*.npz"))
            self.assertEqual(len(samples), 2)
            before = samples[0].read_bytes()
            with np.load(samples[0], allow_pickle=False) as payload:
                meta = json.loads(str(payload["metadata_json"]))
                self.assertEqual(meta["channel_map_id"], maps[0]["channel_map_id"])
                np.testing.assert_array_equal(payload["data"], raw[0, :4, :].T[[0, 2, 3]])
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                main()
            self.assertEqual(samples[0].read_bytes(), before)
