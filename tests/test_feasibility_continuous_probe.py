"""Scoped synthetic HDF5 tests; never opens real FEBUS waveforms."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from finetuning.run_feasibility_continuous_probe import (
    ROOT, build_plan, checked_dataset, inspect_sources, load_region, run,
)
from finetuning.run_feasibility_npz_probe import (
    evaluate_window, fit_windows, quantized_window,
)
from compression.feasibility_tables import contexts


class ContinuousProbeTests(unittest.TestCase):
    def fixture(self, root):
        fid = "synthetic"
        row = {
            "drive_id": fid, "group_id": "synthetic/test", "role": "fit",
            "raw_path": f"data/finetune/raw/{fid}.h5",
            "sample_path": "synthetic/r0/t00000",
            "dataset": "instrument/Source1/Zone1/Strain Rate",
            "source_shape": [2, 21000, 4], "block_index": 1,
            "source_time_start": 0, "time_samples": 5000,
            "raw_block_conditional_clock_interval_s": [1000 - .525, 1000 + .525],
            "regions": [{
                "source_channel_indices": [1, 2], "source_extent_channel_indices": [101, 102],
                "metadata_positions_m": [20.2, 20.4],
                "dataset_relative_positions_m": [.2, .4],
            }],
        }
        path = root / row["raw_path"]
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as handle:
            source = handle.create_group("instrument/Source1")
            source.create_dataset("time", data=[999., 1000.])
            zone = source.create_group("Zone1")
            zone.attrs.update({"Spacing": [.2, .05], "Origin": [0., -525.],
                               "Extent": [100, 103, 0, 20999],
                               "BlockOverlap": 5, "BlockRate": 1000})
            ds = zone.create_dataset("Strain Rate", shape=(2, 21000, 4), dtype="f4")
            ds[0] = -100.
            ds[1, :5000] = np.arange(20000, dtype="f4").reshape(5000, 4)
            ds[1, 5000:] = -200.
        return row

    def test_exact_block_and_region_no_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            window = load_region(root, row, [1, 2])
            expected = np.arange(20000, dtype="f4").reshape(5000, 4)[:, 1:3]
            np.testing.assert_array_equal(window["raw"], expected)
            self.assertEqual(inspect_sources(root, {"rows": [row]})["blockers"], [])

    def test_split_slice_and_reset_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            full = load_region(root, row, [1, 2])["raw"]
            q = full.astype("i4")
            whole = contexts(q, np.array([], dtype="u4"), [1, 2], "spatiotemporal")
            for start in range(0, 5000, 1000):
                part = {**row, "source_time_start": start, "time_samples": 1000}
                np.testing.assert_array_equal(load_region(root, part, [1, 2])["raw"],
                                              full[start:start + 1000])
                np.testing.assert_array_equal(
                    contexts(q[start:start + 1000], np.array([], dtype="u4"),
                             [1, 2], "spatiotemporal"), whole[start:start + 1000])

    def test_headers_bounds_metadata_and_clock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            mutations = (
                {"source_shape": [2, 21000, 5]}, {"block_index": 2},
                {"source_time_start": 1000, "time_samples": 5000},
                {"raw_block_conditional_clock_interval_s": [0, 1]},
            )
            for mutation in mutations:
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_region(root, {**row, **mutation}, [1, 2])
            with h5py.File(root / row["raw_path"], "r+") as handle:
                handle[row["dataset"]].parent.attrs["Spacing"] = [.2, .1]
            self.assertTrue(inspect_sources(root, {"rows": [row]})["blockers"])

    def test_missing_and_nonhdf5_block_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            path = root / row["raw_path"]
            path.unlink()
            audit = inspect_sources(root, {"rows": [row]})
            self.assertEqual(audit["blockers"][0]["error_type"], "FileNotFoundError")
            path.write_text("not HDF5")
            self.assertEqual(inspect_sources(root, {"rows": [row]})["waveform_cells_read"], 0)

    def test_native_three_axis_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.fixture(root)
            with h5py.File(root / row["raw_path"], "r+") as handle:
                attrs = handle[row["dataset"]].parent.attrs
                attrs["Spacing"] = [.2, .05, 1.]
                attrs["Origin"] = [0., -525., 0.]
                attrs["Extent"] = [100, 103, 0, 20999, 0, 0]
                attrs["BlockOverlap"] = [5]
                attrs["BlockRate"] = [1000]
            self.assertEqual(load_region(root, row, [1, 2])["raw"].shape, (5000, 2))

    def test_authorization_precedes_source_access(self):
        with self.assertRaisesRegex(ValueError, "authorization"):
            run({}, {}, Path("/never-created"))

    def test_real_metadata_plan_only(self):
        revalidated = json.loads((ROOT / "results/experiments/feasibility_centered_clock_revalidation_20261118/results.json").read_text())
        scales = json.loads((ROOT / "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json").read_text())
        with (ROOT / "data/finetune/spatial_band_v1_fold0_20261024/samples_manifest.csv").open() as stream:
            manifest = list(csv.DictReader(stream))
        plan = build_plan(revalidated, scales, manifest)
        self.assertEqual(plan["planned_original_cells"], 17_575_000)
        self.assertEqual(len(plan["sigma_p0"]), 9)
        self.assertEqual(len(plan["ratios"]), 9)
        self.assertTrue(all(r["channels"] * r["time_samples"] <= 2_000_000
                            for r in plan["rows"]))
        bad = copy.deepcopy(revalidated)
        bad["files"][0]["regions"][0]["time_stop_source_index_exclusive"] = 5001
        with self.assertRaises(ValueError):
            build_plan(bad, scales, manifest)

    def test_multigeometry_roundtrip_specials_all_arms(self):
        rng = np.random.default_rng(117)
        windows = []
        for ids in ([2, 3], [10, 11, 12]):
            raw = rng.normal(size=(1000, len(ids))).astype("f4")
            raw.view("u4").flat[:4] = [0x80000000, 0x7fc00123, 0x7f800000, 0xff800000]
            windows.append(quantized_window({
                "raw": raw, "channel_ids": ids, "group_id": "synthetic/test",
                "role": "fit"}, .03))
        hashes = {}
        for mode in ("marginal", "temporal", "spatiotemporal"):
            model = fit_windows(windows, .03, 15, mode)
            for index, window in enumerate(windows):
                record, _, _, _ = evaluate_window(window, model, .03)
                pair = (record["q_sha256"], record["waveform_sha256"])
                if index in hashes:
                    self.assertEqual(hashes[index], pair)
                hashes[index] = pair
                self.assertEqual(record["bound_violations"], 0)
                self.assertTrue(record["portable_roundtrip"])


if __name__ == "__main__":
    unittest.main()
