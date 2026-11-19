"""Synthetic and metadata-only tests for exact frozen-table amortization."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from compression.feasibility_tables import fit_tables, quantize
from finetuning import run_feasibility_npz_amortization as amort


GROUP = "single_mode/Configuration 8"
PREFIX = "data/finetune/spatial_band_v1_fold0_20261024/train/"


def metadata():
    base = {"drive_id": "check", "split": "train", "group_id": GROUP,
            "family": "single_mode", "channels": "2", "time_samples": "1000",
            "dt_s": "0.00005", "channel_map_id": "a", "dataset": "dataset",
            "block_index": "0", "time_start": "0"}
    original = {**base, "sample_path": PREFIX + "original.npz"}
    rows = [original] + [
        {**base, "sample_path": PREFIX + f"new{i}.npz", "time_start": str(i * 1000)}
        for i in range(1, 26)]
    maps = {"a": {"drive_id": "check", "source_channel_indices": [2, 3]},
            "b": {"drive_id": "check", "source_channel_indices": [2, 3]},
            "c": {"drive_id": "check", "source_channel_indices": [3, 4]}}
    return rows, [original], maps


def installation(mode="marginal", raw=None):
    if raw is None:
        raw = np.ones((1000, 2), dtype=np.float32)
    q, positions, _ = quantize(raw, .1)
    model = fit_tables(q, positions, .1, [2, 3], 15, mode)
    blob = model.dump()
    decoder = (amort.ROOT / "compression/feasibility_portable.py").read_bytes()
    archive = amort.zip_bytes({amort.digest(blob) + ".ftb": blob, "decoder.py": decoder,
                               "platform.json": amort.canonical({"platform": "test"})})
    return archive, blob, decoder


class SelectionTests(unittest.TestCase):
    def test_hash_order_cap_and_no_values(self):
        rows, original, maps = metadata()
        with patch.object(np, "load", side_effect=AssertionError("No values")):
            selected, coverage = amort.select_windows(rows, original, maps)
        expected = sorted(rows[1:], key=lambda r: (
            amort.selection_key(r["sample_path"]), r["sample_path"]))[:20]
        self.assertEqual([r["sample_path"] for r in selected],
                         [r["sample_path"] for r in expected])
        self.assertEqual(coverage[GROUP]["selected"], 20)
        self.assertEqual(amort.selection_key("x"),
                         amort.digest(b"feasibility-amortization-v1|x"))

    def test_physical_indices_not_map_ids(self):
        rows, original, maps = metadata()
        rows = rows[:3]
        rows[1]["channel_map_id"] = "b"
        rows[2]["channel_map_id"] = "c"
        selected, _ = amort.select_windows(rows, original, maps)
        self.assertEqual([r["sample_path"] for r in selected], [rows[1]["sample_path"]])

    def test_original_interval_excluded_and_touching_allowed(self):
        rows, original, maps = metadata()
        rows[1]["time_start"] = "999"
        selected, _ = amort.select_windows(rows[:3], original, maps)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["time_start"], "2000")
        self.assertFalse(amort.overlaps(original[0], {**original[0], "time_start": "1000"}))

    def test_greedy_overlap_and_cross_block_not_claimed_unique(self):
        rows, original, maps = metadata()
        rows = rows[:4]
        rows[1]["time_start"] = "1100"
        rows[2]["time_start"] = "1200"
        rows[3].update({"block_index": "1", "time_start": "0"})
        selected, _ = amort.select_windows(rows, original, maps)
        self.assertEqual(len(selected), 2)
        self.assertEqual({r["block_index"] for r in selected}, {"0", "1"})
        self.assertFalse(amort.overlaps(rows[3], original[0]))

    def test_geometry_dataset_dt_and_length_filter_without_replacement(self):
        for field, value in (("dataset", "other"), ("dt_s", ".1"),
                             ("channels", "1"), ("time_samples", "999")):
            rows, original, maps = metadata()
            rows[1][field] = value
            with self.subTest(field=field):
                selected, _ = amort.select_windows(rows[:2], original, maps)
                self.assertEqual(selected, [])

    def test_unsafe_or_validation_candidates_fail_closed(self):
        for field, value in (("split", "val"), ("sample_path", "../bad.npz")):
            rows, original, maps = metadata()
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                amort.select_windows(rows[:2], original, maps)

    def test_missing_map_and_duplicate_paths_fail_closed(self):
        rows, original, maps = metadata()
        with self.assertRaises(ValueError):
            amort.select_windows(rows + [rows[1]], original, maps)
        rows[1]["channel_map_id"] = "missing"
        with self.assertRaises(KeyError):
            amort.select_windows(rows[:2], original, maps)


class InstallationTests(unittest.TestCase):
    def test_exact_serialized_cdf_and_geometry_all_modes(self):
        for mode in amort.MODES:
            archive, blob, decoder = installation(mode)
            model, restored = amort.read_installation(
                archive, amort.digest(archive), mode, .1, [2, 3], decoder)
            self.assertEqual(restored, blob)
            self.assertEqual(model.dump(), blob)
            self.assertIsNone(model.probabilities)

    def test_sha_geometry_mode_epsilon_decoder_fail_closed(self):
        archive, _, decoder = installation()
        good = [archive, amort.digest(archive), "marginal", .1, [2, 3], decoder]
        for index, value in ((1, "bad"), (2, "temporal"), (3, .03),
                             (4, [3, 4]), (5, b"other")):
            args = copy.copy(good)
            args[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                amort.read_installation(*args)

    def test_unexpected_archive_member_rejected(self):
        archive, _, decoder = installation()
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            entries = {n: z.read(n) for n in z.namelist()}
        entries["extra"] = b"unpaid"
        changed = amort.zip_bytes(entries)
        with self.assertRaises(ValueError):
            amort.read_installation(changed, amort.digest(changed), "marginal", .1, [2, 3], decoder)

    def test_roundtrip_special_values_no_fit_and_no_float_nll(self):
        raw = np.random.default_rng(115).normal(size=(1000, 2)).astype(np.float32)
        raw.view("u4").reshape(-1)[:4] = [0x80000000, 0x7fc12345, 0x7f800000, 0xff800000]
        for mode in amort.MODES:
            archive, _, decoder = installation(mode)
            model, blob = amort.read_installation(
                archive, amort.digest(archive), mode, .1, [2, 3], decoder)
            qw = amort.quantized_window({"raw": raw, "channel_ids": [2, 3]}, .1)
            with patch("compression.feasibility_tables.fit_tables", side_effect=AssertionError("No fit")):
                record, transport = amort.evaluate(qw, model, blob)
            self.assertTrue(record["portable_roundtrip"])
            self.assertEqual(record["bound_violations"], 0)
            self.assertIsNone(record["float_nll_bits"])
            self.assertEqual(record["reset_first_sample_cells"], 2)
            self.assertEqual(sum(record["components"].values()), len(transport))


class ExecutionTests(unittest.TestCase):
    def test_changed_frozen_input_blocks_before_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input").write_bytes(b"changed")
            plan = {"input_sha256": {"input": amort.digest(b"original")}}
            with patch.object(np, "load", side_effect=AssertionError("No values")):
                with self.assertRaises(ValueError):
                    amort.run(plan, root / "output", root)
            self.assertFalse((root / "output").exists())

    def test_synthetic_end_to_end_freeze_cumulative_bytes_and_no_fit(self):
        rows, original, maps = metadata()
        selected, coverage = amort.select_windows(rows[:3], original, maps)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = {}
            for mode in amort.MODES:
                archive, table_blob, _ = installation(mode)
                path = mode + ".zip"
                (root / path).write_bytes(archive)
                specs[f"{GROUP}|0.1|{mode}"] = {
                    "path": path, "sha256": amort.digest(archive), "bytes": len(archive),
                    "table_sha256": amort.digest(table_blob), "epsilon": .1,
                    "k": 15, "channel_ids": [2, 3]}
            decoder_path = root / "compression/feasibility_portable.py"
            decoder_path.parent.mkdir()
            decoder_path.write_bytes((amort.ROOT / "compression/feasibility_portable.py").read_bytes())
            plan = {"input_sha256": {}, "rows": selected, "coverage": coverage,
                    "geometry": {r["sample_path"]: [2, 3] for r in selected},
                    "installations": specs, "ratios": [.1], "checkpoints": [1, 5, 10, 20],
                    "temporal_scope": "Not unique physical cells", "float_probability_policy": "unavailable"}
            reads = []
            output = root / "output"
            def loader(root, row, ids):
                self.assertTrue((output / "freeze.json").exists())
                self.assertEqual(json.loads((output / "freeze.json").read_text())["waveform_cells_read"], 0)
                reads.append(row["sample_path"])
                return {**row, "raw": np.ones((1000, 2), dtype=np.float32), "channel_ids": ids}
            with patch("compression.feasibility_tables.fit_tables", side_effect=AssertionError("No fit")), \
                 patch("finetuning.run_feasibility_npz_probe.fit_windows", side_effect=AssertionError("No fit")):
                result = amort.run(plan, output, root, loader)
            self.assertEqual(len(reads), 2)
            self.assertEqual(result["npz_cell_occurrences_read"], 4000)
            self.assertEqual(len(result["rows"]), 6)
            for curve in result["curves"]:
                payload = output / f"group00_0.1_{curve['mode']}.n{curve['windows']:02d}.transport.zip"
                self.assertEqual(payload.stat().st_size, curve["transport_bytes"])
                self.assertEqual(curve["total_bytes"],
                                 curve["transport_bytes"] + curve["installation_bytes"])
            for aggregate in result["aggregates"]:
                self.assertEqual(aggregate["windows"], min(aggregate["horizon"], 2))
            with self.assertRaises(FileExistsError):
                amort.run(plan, output, root, loader)

    def test_failure_preserves_freeze_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            with patch.object(amort, "execute_plan", side_effect=ValueError("blocked")):
                with self.assertRaises(ValueError):
                    amort.run({"input_sha256": {}}, output)
            self.assertTrue((output / "freeze.json").exists())
            self.assertFalse(json.loads((output / "blocker.json").read_text())["selection_changed"])


if __name__ == "__main__":
    unittest.main()
