"""Persistent spatial filtering must preserve events, geometry and retained bytes."""
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

from finetuning.spatial_band_filter import detect_spatial_bands, retained_regions, POLICY
from finetuning.prepare_finetune_samples import main as prepare
from acquisition.extract_febus_windows import main as extract


class SpatialBandFilterTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.random.default_rng(123).normal(size=(1, 40, 8000)).astype(np.float32)

    def test_persistent_band_removed_and_retained_samples_exact(self):
        self.raw[:, -4:] *= 50
        before = self.raw.copy()
        result = detect_spatial_bands(self.raw)
        self.assertEqual(result["removed_source_channel_indices"], list(range(36, 40)))
        np.testing.assert_array_equal(self.raw, before)
        regions = retained_regions(self.raw[0], result, 16)
        np.testing.assert_array_equal(np.concatenate(regions), np.arange(36))
        self.assertTrue(all(len(r)<=16 and np.all(np.diff(r)==1) for r in regions))

    def test_point_spikes_and_short_events_preserved(self):
        self.raw[:, 10:14, 42] = 1e30
        self.raw[:, 20:24, :1000] *= 100
        self.assertEqual(detect_spatial_bands(self.raw)["removed_source_channel_indices"], [])

    def test_interior_band_not_joined(self):
        self.raw[:, 18:21] *= 50
        result = detect_spatial_bands(self.raw)
        self.assertEqual(result["removed_source_channel_indices"], [18, 19, 20])
        regions = retained_regions(self.raw[0], result)
        self.assertEqual(len(regions), 2)
        np.testing.assert_array_equal(regions[0], np.arange(18))
        np.testing.assert_array_equal(regions[1], np.arange(21, 40))

    def test_global_gain_and_coherent_event_not_removed(self):
        self.raw *= 1e15
        self.assertEqual(detect_spatial_bands(self.raw)["removed_source_channel_indices"], [])

    def test_persistent_large_offset_detected_without_clipping(self):
        self.raw[:, -4:] += 100
        result = detect_spatial_bands(self.raw)
        self.assertEqual(result["removed_source_channel_indices"], list(range(36,40)))

    def test_coverage_guard_abstains(self):
        self.raw[:, :12] *= 100
        result = detect_spatial_bands(self.raw)
        self.assertEqual(result["status"], "abstained_excessive_coverage_loss")
        self.assertEqual(result["removed_source_channel_indices"], [])

    def test_too_short_or_constant_abstains(self):
        for values in (self.raw[..., :1000], np.zeros_like(self.raw)):
            self.assertEqual(detect_spatial_bands(values)["status"], "abstained_insufficient_windows")

    def test_single_channel_excess_not_a_band(self):
        self.raw[:, 10] *= 100
        result = detect_spatial_bands(self.raw)
        self.assertEqual(result["candidate_source_channel_indices"], [10])
        self.assertEqual(result["removed_source_channel_indices"], [])

    def test_file_decision_cannot_follow_one_bad_block(self):
        raw = np.repeat(self.raw, 5, axis=0)
        raw[0, -4:] *= 100
        self.assertEqual(detect_spatial_bands(raw)["removed_source_channel_indices"], [])

    def test_training_and_inference_share_policy_and_source_coordinates(self):
        self.raw[:, 18:21] *= 50
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with h5py.File(root/"id.h5", "w") as handle:
                dataset = handle.create_dataset("Source/Zone/Strain Rate", data=self.raw.transpose(0,2,1))
                dataset.parent.attrs["Spacing"] = [.2, .05]
                dataset.parent.attrs["Origin"] = [0., 0.]
                dataset.parent.attrs["Extent"] = [100, 139, 0, 7999]
            (root/"manifest.csv").write_text("drive_id,family,configuration,fold,title\nid,single_mode,one,2,test.h5\n")
            args = ["prepare", "--manifest", str(root/"manifest.csv"), "--raw-dir", str(root),
                    "--output-dir", str(root/"prepared"), "--windows-per-block", "2", "--max-channels", "16"]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                prepare()
            decisions = json.loads((root/"prepared/preprocessing.json").read_text())["file_decisions"]
            self.assertEqual(decisions["id"]["removed_source_channel_indices"], [18,19,20])
            maps = [json.loads(line) for line in (root/"prepared/channel_maps.jsonl").read_text().splitlines()]
            for item in maps:
                indices = item["source_channel_indices"]
                self.assertTrue(item["selected_rows_contiguous"])
                self.assertFalse(set(indices)&{18,19,20})
            for path in (root/"prepared/train").glob("*.npz"):
                with np.load(path) as packet:
                    meta = json.loads(str(packet["metadata_json"].item()))
                    mapping = next(m for m in maps if m["channel_map_id"]==meta["channel_map_id"])
                    start = meta["time_start"]
                    np.testing.assert_array_equal(packet["data"], self.raw[0, mapping["source_channel_indices"], start:start+1000])
            config = {"files": [{"id": "id", "local_filename": "id.h5"}], "time_samples": 8000,
                      "source_block_samples": 8000, "source_distance_start_m": 20., "source_distance_end_m": 27.8}
            (root/"config.json").write_text(json.dumps(config))
            args = ["extract", "--config", str(root/"config.json"), "--raw-dir", str(root),
                    "--output-dir", str(root/"inference"), "--manifest-output", str(root/"extraction.json")]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                extract()
            paths = sorted((root/"inference").glob("*.npz"))
            self.assertEqual(len(paths), 2)
            selected = []
            for path in paths:
                with np.load(path) as packet:
                    meta = json.loads(str(packet["metadata_json"].item()))
                    self.assertEqual(meta["spatial_band_policy"], POLICY)
                    self.assertEqual(meta["spatial_filter_id"], decisions["id"]["filter_id"])
                    indices = meta["retained_source_channel_indices"]
                    selected.extend(indices)
                    np.testing.assert_array_equal(packet["data"], self.raw[:, indices])
                    self.assertAlmostEqual(meta["distance_start_m"], 20+indices[0]*.2)
            self.assertEqual(selected, list(range(18))+list(range(21,40)))


if __name__ == "__main__":
    unittest.main()
