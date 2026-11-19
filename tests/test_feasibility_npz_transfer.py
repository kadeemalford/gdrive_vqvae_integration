"""Metadata, gate and synthetic byte-ledger tests; no real waveform reads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from finetuning import run_feasibility_npz_transfer as transfer
from tests.test_feasibility_npz_amortization import GROUP, installation, metadata

amort = transfer.amort


def gate_result():
    return {"kind": "feasibility-npz-amortization-results-v1", "aggregates": [
        {"ratio": ratio, "horizon": 20, "mode": mode, "groups": 8,
         "windows": 160, "cells": 5420000,
         "total_bytes": 100000 if mode == "marginal" else 90000}
        for ratio in (.03, .1) for mode in amort.MODES]}


def transfer_metadata():
    rows, originals, maps = metadata()
    refs = {GROUP: originals[0]}
    candidates = []
    for fid in ("new-a", "new-b"):
        maps[fid] = {"drive_id": fid, "source_channel_indices": [2, 3]}
        candidates.extend({**r, "drive_id": fid, "channel_map_id": fid,
                           "sample_path": r["sample_path"].replace("new", fid)}
                          for r in rows[1:])
    envelope = {"kind": "feasibility-clock-envelope-sensitivity-v1",
                "candidate_roles": {"fit": "fit"},
                "components": [{"members": [fid], "check_ids": [fid]}
                               for fid in ("new-a", "new-b")]}
    return candidates, refs, maps, envelope


class GateTests(unittest.TestCase):
    def test_both_ratios_strictly_positive_required(self):
        self.assertEqual(len(transfer.gate115(gate_result())), 2)
        for value in (100000, 100001):
            r = gate_result()
            next(a for a in r["aggregates"] if a["ratio"] == .03
                 and a["mode"] == "spatiotemporal")["total_bytes"] = value
            with self.assertRaises(ValueError):
                transfer.gate115(r)

    def test_missing_arm_and_population_mismatch_block(self):
        r = gate_result()
        r["aggregates"].pop()
        with self.assertRaises(ValueError):
            transfer.gate115(r)
        r = gate_result()
        r["aggregates"][0]["cells"] = 1
        with self.assertRaises(ValueError):
            transfer.gate115(r)

    def test_gate_checked_before_manifest_or_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / amort.OUTPUT
            directory.mkdir(parents=True)
            p = {"input_sha256": {}}
            (directory / "plan.json").write_text(json.dumps(p))
            (directory / "freeze.json").write_text(json.dumps(
                {"plan_sha256": amort.digest(amort.canonical(p))}))
            r = gate_result()
            next(a for a in r["aggregates"] if a["mode"] == "spatiotemporal")["total_bytes"] = 100000
            (directory / "results.json").write_text(json.dumps(r))
            for path in (transfer.PROTOCOL, *transfer.SOURCES):
                f = root / path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"protocol/source")
            with patch.object(np, "load", side_effect=AssertionError("No values")):
                with self.assertRaisesRegex(ValueError, "gate failed"):
                    transfer.build_plan(root)


class SelectionTests(unittest.TestCase):
    def test_independent_hash_no_original_exclusion_for_new_file(self):
        rows, refs, maps, envelope = transfer_metadata()
        with patch.object(np, "load", side_effect=AssertionError("No values")):
            selected, files, groups = transfer.select_windows(
                rows, refs, maps, envelope, {"fit"}, strict=False)
        self.assertEqual(len(selected), 40)
        self.assertEqual(groups[GROUP]["compatible_files"], 2)
        for fid in files:
            expected = sorted([r for r in rows if r["drive_id"] == fid],
                              key=lambda r: (transfer.selection_key(r["sample_path"]), r["sample_path"]))[:20]
            self.assertEqual([r["sample_path"] for r in selected if r["drive_id"] == fid],
                             [r["sample_path"] for r in expected])
        self.assertEqual(transfer.selection_key("x"), amort.digest(b"feasibility-transfer-v1|x"))

    def test_fit_touching_component_excluded(self):
        rows, refs, maps, envelope = transfer_metadata()
        envelope["components"][0]["members"].append("fit")
        selected, files, _ = transfer.select_windows(rows, refs, maps, envelope, set(), strict=False)
        self.assertEqual(set(files), {"new-b"})
        self.assertEqual(len(selected), 20)

    def test_strict_indices_not_map_ids_dataset_dt_and_missing_coverage(self):
        rows, refs, maps, envelope = transfer_metadata()
        maps["new-a"]["source_channel_indices"] = [3, 4]
        for r in rows:
            if r["drive_id"] == "new-b":
                r["dataset"] = "not-equivalent"
        selected, files, groups = transfer.select_windows(rows, refs, maps, envelope, set(), strict=False)
        self.assertEqual(selected, [])
        self.assertEqual(groups[GROUP]["compatible_files"], 0)
        self.assertEqual(len(files), 2)
        with self.assertRaises(ValueError):
            transfer.select_windows(rows, refs, maps, envelope, set())

    def test_no_overlap_per_file_block_and_fewer_than_twenty(self):
        rows, refs, maps, envelope = transfer_metadata()
        rows = [r for r in rows if r["drive_id"] == "new-a"][:3]
        rows[0]["time_start"] = "1000"
        rows[1]["time_start"] = "1001"
        rows[2].update({"time_start": "1000", "block_index": "1"})
        selected, files, _ = transfer.select_windows(rows, refs, maps, envelope, set(), strict=False)
        self.assertEqual(len(selected), 2)
        self.assertEqual(files["new-a"]["selected"], 2)
        self.assertFalse(amort.overlaps(*selected))

    def test_nontrain_candidates_fail_closed(self):
        args = transfer_metadata()
        args[0][0]["split"] = "val"
        with self.assertRaises(ValueError):
            transfer.select_windows(*args, set(), strict=False)


class ExecutionTests(unittest.TestCase):
    def test_two_files_pay_one_group_installation_exact_artifacts(self):
        rows, refs, maps, envelope = transfer_metadata()
        rows = [r for r in rows if int(r["time_start"]) <= 2000]
        selected, files, coverage = transfer.select_windows(rows, refs, maps, envelope, set(), strict=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = root / amort.OUTPUT
            prior.mkdir(parents=True)
            (prior / "results.json").write_text(json.dumps(gate_result()))
            decoder = root / "compression/feasibility_portable.py"
            decoder.parent.mkdir()
            decoder.write_bytes((amort.ROOT / "compression/feasibility_portable.py").read_bytes())
            specs = {}
            for ratio in (.03, .1):
                for mode in amort.MODES:
                    archive, blob, _ = installation(mode)
                    path = f"{ratio}_{mode}.zip"
                    (root / path).write_bytes(archive)
                    specs[f"{GROUP}|{ratio:g}|{mode}"] = {
                        "path": path, "sha256": amort.digest(archive), "table_sha256": amort.digest(blob),
                        "bytes": len(archive), "epsilon": .1, "k": 15, "channel_ids": [2, 3]}
            plan = {"kind": "feasibility-npz-transfer-plan-v1", "input_sha256": {},
                    "f4_authorized": False, "gate115": transfer.gate115(gate_result()),
                    "rows": selected, "files": files, "coverage": coverage, "installations": specs,
                    "geometry": {r["sample_path"]: [2, 3] for r in selected}, "ratios": [.03, .1],
                    "checkpoints": [1, 5, 10, 20], "temporal_scope": "Not independent",
                    "deployment": "One installation per group", "float_probability_policy": "Unavailable"}
            output = root / "output"
            reads = []
            def loader(root, row, ids):
                self.assertTrue((output / "freeze.json").exists())
                reads.append(row["sample_path"])
                return {**row, "raw": np.ones((1000, 2), dtype=np.float32), "channel_ids": ids}
            with patch("compression.feasibility_tables.fit_tables", side_effect=AssertionError("No fit")), \
                 patch("finetuning.run_feasibility_npz_probe.fit_windows", side_effect=AssertionError("No fit")):
                result = transfer.run(plan, output, root, loader)
            self.assertEqual(len(reads), 4)
            self.assertEqual(len(result["rows"]), 24)
            for c in result["curves"]:
                spec = specs[f"{GROUP}|{c['ratio']:g}|{c['mode']}"]
                self.assertEqual(c["installation_bytes"], spec["bytes"])
                self.assertEqual(c["files"], 2)
                self.assertEqual(c["windows"], 2 * min(c["horizon"], 2))
                self.assertEqual(c["total_bytes"], c["transport_bytes"] + spec["bytes"])
                if c["horizon"] in plan["checkpoints"]:
                    path = output / f"group00_{c['ratio']:g}_{c['mode']}.n{c['horizon']:02d}.transport.zip"
                    self.assertEqual(path.stat().st_size, c["transport_bytes"])
                    with zipfile.ZipFile(path) as z:
                        self.assertEqual(len(z.namelist()), 2)
            self.assertFalse(result["f4_authorized"])
            with self.assertRaises(FileExistsError):
                transfer.run(plan, output, root, loader)

    def test_success_requires_both_aggregate_thresholds_and_no_group_loss_above_one_percent(self):
        aggregates, curves = [], []
        for ratio in (.03, .1):
            for mode, total in (("marginal", 1000), ("spatiotemporal", 800)):
                aggregates.append({"ratio": ratio, "horizon": 20, "mode": mode,
                                   "total_bytes": total, "cells": 10000})
            curves.append({"group_id": GROUP, "ratio": ratio, "mode": "spatiotemporal",
                           "horizon": 20, "total_bytes": 800, "total_bytes_saving_vs_marginal": 200})
        self.assertTrue(transfer.transfer_decision(aggregates, curves)["passes_both"])
        curves[0].update({"total_bytes": 1020, "total_bytes_saving_vs_marginal": -20})
        self.assertFalse(transfer.transfer_decision(aggregates, curves)["passes_both"])
        curves[0].update({"total_bytes": 800, "total_bytes_saving_vs_marginal": 200})
        aggregates[0]["cells"] = 10000000
        self.assertFalse(transfer.transfer_decision(aggregates, curves)["passes_both"])


if __name__ == "__main__":
    unittest.main()
