"""Synthetic/metadata-only coverage of the isolated-NPZ F2 runner."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from compression.feasibility_stream import decode, encode
from compression.feasibility_tables import contexts, fit_tables, quantize, reconstruct
from finetuning import run_feasibility_npz_probe as probe


GROUP = "single_mode/Configuration 8"
PREFIX = "data/finetune/spatial_band_v1_fold0_20261024/train/"


def metadata():
    rows = []
    for fid, channels in (("fit-a", 2), ("fit-b", 3), ("check", 4)):
        for index in range(3):
            rows.append({"drive_id": fid, "sample_path": f"{PREFIX}{fid}_{index}.npz",
                         "split": "train", "group_id": GROUP, "family": "single_mode",
                         "channels": str(channels), "time_samples": "1003",
                         "dt_s": "0.00005", "channel_map_id": fid})
    envelope = {
        "kind": "feasibility-clock-envelope-sensitivity-v1",
        "candidate_roles": {"fit-a": "fit", "fit-b": "fit", "check": "check"},
        "contrasts": {GROUP: {"has_candidate_contrast": True}},
        "components": [{"members": [fid]} for fid in ("fit-a", "fit-b", "check")],
    }
    scales = {"version": "physical-pooled-train-scale-v1", "scales": {GROUP: .01}}
    maps = {fid: {"drive_id": fid, "source_channel_indices": list(range(n))}
            for fid, n in (("fit-a", 2), ("fit-b", 3), ("check", 4))}
    return rows, envelope, scales, maps


def window(raw, role="fit", group=GROUP, ids=None):
    raw = np.asarray(raw, dtype=np.float32)
    return {"raw": raw, "role": role, "group_id": group,
            "channel_ids": list(range(raw.shape[1])) if ids is None else ids}


class MetadataTests(unittest.TestCase):
    def test_selection_hash_path_only_and_no_signal_reads(self):
        rows, envelope, _, _ = metadata()
        with patch.object(np, "load", side_effect=AssertionError("No signals")):
            selected = probe.select_windows(rows, envelope, strict=False)
        self.assertEqual(len(selected), 3)
        for row in selected:
            candidates = [r for r in rows if r["drive_id"] == row["drive_id"]]
            expected = min(candidates, key=lambda r: (
                hashlib.sha256(("feasibility-npz-v1|" + r["sample_path"]).encode()).hexdigest(),
                r["sample_path"]))
            self.assertEqual(row["sample_path"], expected["sample_path"])

    def test_excluded_roles_multicore_and_no_contrast_never_replaced(self):
        rows, envelope, _, _ = metadata()
        envelope["candidate_roles"]["fit-b"] = "excluded"
        rows.append({**rows[0], "drive_id": "mc", "group_id": "multicore/C1",
                     "family": "multicore", "sample_path": PREFIX + "mc.npz"})
        envelope["candidate_roles"]["mc"] = "fit"
        envelope["contrasts"]["multicore/C1"] = {"has_candidate_contrast": True}
        selected = probe.select_windows(rows, envelope, strict=False)
        self.assertEqual({r["drive_id"] for r in selected}, {"fit-a", "check"})
        envelope["contrasts"][GROUP]["has_candidate_contrast"] = False
        self.assertEqual(probe.select_windows(rows, envelope, strict=False), [])

    def test_cross_split_component_and_unsafe_paths_rejected(self):
        for mutation in ("split", "component", "path"):
            rows, envelope, _, _ = metadata()
            if mutation == "split":
                for r in rows:
                    if r["drive_id"] == "check":
                        r["split"] = "val"
            elif mutation == "component":
                envelope["components"] = [{"members": ["fit-a", "check"]}]
            else:
                for r in rows:
                    r["sample_path"] = "../" + r["sample_path"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.select_windows(rows, envelope, strict=False)

    def test_plan_uses_p0_without_recalibration_and_charges_real_geometry(self):
        with patch.object(np, "load", side_effect=AssertionError("No signals")):
            plan = probe.build_plan(*metadata(), strict=False)
        self.assertEqual(plan["sigma_p0"], {GROUP: 1.})
        self.assertEqual(plan["ratios"], [.03, .1])
        self.assertEqual(plan["absent_families"], ["multicore"])
        self.assertEqual(plan["census"]["fit"]["cells"], 500 * 10 + 15)
        self.assertEqual(plan["widths"][GROUP], {"fit": [2, 3], "check": [4]})
        self.assertIsNone(plan["generalist_gate"])
        with self.assertRaises(ValueError):
            probe.build_plan(*metadata())  # Production census must be exact.

    def test_invalid_map_and_zero_scale_are_blockers(self):
        for mutation in ("map", "scale"):
            rows, envelope, scales, maps = metadata()
            if mutation == "map":
                maps["check"]["source_channel_indices"] = [0, 1, 1, 3]
            else:
                scales["scales"][GROUP] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.build_plan(rows, envelope, scales, maps, strict=False)

    def test_protocol_required_before_any_signal_reads(self):
        plan = probe.build_plan(*metadata(), strict=False)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new"
            with patch.object(np, "load", side_effect=AssertionError("No signals")):
                with self.assertRaises(ValueError):
                    probe.run(plan, {}, output)
            self.assertFalse(output.exists())
        protocol = {"kind": "feasibility-npz-execution-v1",
                    "execute_npz_train_probe": True, "plan_sha256": probe.digest(probe.canonical(plan))}
        probe.validate_authorization(plan, protocol)
        changed = copy.deepcopy(plan)
        changed["ratios"] = [.01]
        with self.assertRaises(ValueError):
            probe.validate_authorization(changed, protocol)


class CoderTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(113)

    def test_single_window_fit_matches_existing_coder(self):
        w = probe.quantized_window(window(self.rng.normal(size=(1003, 3))), .03)
        for mode in probe.MODES:
            a = probe.fit_windows([w], .03, 15, mode)
            b = fit_tables(w["q"], w["positions"], .03, w["channel_ids"], 15, mode)
            self.assertEqual(a.dump(), b.dump())

    def test_accumulation_resets_each_window_and_handles_different_widths(self):
        a = probe.quantized_window(window(np.ones((1003, 2))), .1)
        b = probe.quantized_window(window(np.ones((1003, 3)), ids=[3, 4, 9]), .1)
        tables = probe.fit_windows([a, b], .1, 15, "spatiotemporal")
        # Each first time sample has unavailable temporal context, including tails.
        for w in (a, b):
            c = contexts(w["q"], w["positions"], w["channel_ids"], "spatiotemporal")
            self.assertTrue(np.all(c[[0, 1000]] // 18 == 17))
        bound = probe.bind_geometry(tables, [20, 21, 40, 41])
        self.assertTrue(np.array_equal(bound.frequencies, tables.frequencies))
        self.assertTrue(np.array_equal(bound.mapping, tables.mapping))
        self.assertNotEqual(bound.dump(), tables.dump())
        self.assertEqual(tables.channel_ids, [0, 1])

    def test_checks_and_other_configurations_cannot_fit_or_select_support(self):
        a = probe.quantized_window(window(np.ones((10, 2))), .1)
        b = probe.quantized_window(window(np.ones((10, 2)), role="check"), .1)
        for windows in ([b], [a, b], [a, {**a, "group_id": "doped/5m SMF"}]):
            with self.assertRaises(ValueError):
                probe.fit_windows(windows, .1, 15, "temporal")
            with self.assertRaises(ValueError):
                probe.choose_support(windows, .1, supports=(15,))

    def test_fit_support_costs_include_every_window_and_unique_installations(self):
        windows = [probe.quantized_window(window(self.rng.normal(size=(12, n))), .1)
                   for n in (2, 3, 2)]
        k, candidates = probe.choose_support(windows, .1, supports=(15, 63))
        self.assertIn(k, (15, 63))
        model = probe.fit_windows(windows, .1, 15, "marginal")
        blobs = {probe.bind_geometry(model, w["channel_ids"]).dump() for w in windows}
        self.assertEqual(candidates[0]["fit_installation_bytes"], sum(map(len, blobs)))
        self.assertEqual(candidates[0]["fit_total_bytes"],
                         candidates[0]["fit_transport_bytes"] + sum(map(len, blobs)))

    def test_all_arms_identical_q_waveform_special_bits_tails_and_ledger(self):
        fit = probe.quantized_window(window(self.rng.normal(size=(1003, 3))), .03)
        raw = self.rng.normal(size=(1003, 4)).astype(np.float32)
        raw.view("u4").reshape(-1)[:6] = [
            0x80000000, 0x7fc12345, 0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff]
        check = probe.quantized_window(window(raw, role="check", ids=[1, 2, 7, 8]), .03)
        expected = reconstruct(check["q"], check["positions"], check["bits"], .03)
        for mode in probe.MODES:
            model = probe.fit_windows([fit], .03, 15, mode)
            record, transport, installation, obj = probe.evaluate_window(check, model, .03)
            q, decoded = decode(transport, installation)
            self.assertTrue(np.array_equal(q, check["q"]))
            self.assertEqual(decoded.tobytes(), expected.tobytes())
            self.assertEqual(sum(record["components"].values()), len(transport))
            self.assertEqual(record["bound_violations"], 0)
            self.assertEqual(record["nonfinite_cells"], 3)
            self.assertEqual(record["negative_zero_cells"], 1)
            self.assertEqual(record["python_stdlib_object_bytes"], len(obj))
            self.assertEqual(record["reset_first_sample_cells"], 8)

    def test_forced_contextual_decoder_trace_is_causal_and_resets(self):
        w = probe.quantized_window(window(self.rng.normal(size=(1003, 3)),
                                          ids=[2, 3, 9]), .03)
        for mode in ("temporal", "spatiotemporal"):
            model = probe.fit_windows([w], .03, 15, mode)
            transport, _ = encode(w["q"], w["positions"], w["bits"], model, forced=2)
            trace = []
            decode(transport, model.dump(), trace=trace)
            expected = contexts(w["q"], w["positions"], w["channel_ids"], mode).reshape(-1)
            self.assertEqual([c for _, c, _ in trace], expected.tolist())

    def test_end_to_end_synthetic_counts_objects_and_no_check_fit(self):
        plan = probe.build_plan(*metadata(), strict=False)
        protocol = {"kind": "feasibility-npz-execution-v1",
                    "execute_npz_train_probe": True, "plan_sha256": probe.digest(probe.canonical(plan))}
        reads = []

        def loader(root, row, ids):
            reads.append(row["drive_id"])
            return {**row, **window(self.rng.normal(size=(1003, len(ids))),
                                   role=row["role"], ids=ids)}

        choose_support = probe.choose_support
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with patch.object(probe, "choose_support",
                              side_effect=lambda w, e: choose_support(w, e, supports=(15,))):
                result = probe.run(plan, protocol, output, loader=loader)
            self.assertEqual(sorted(reads), ["check", "fit-a", "fit-b"])
            self.assertEqual(result["original_cells_read"], plan["planned_original_cells"])
            self.assertEqual(len(result["rows"]), 6)
            self.assertTrue((output / "freeze.json").exists())
            for summary in result["configurations"]:
                arms = summary["arms"]
                self.assertEqual(len({a["rmse"] for a in arms.values()}), 1)
                for arm in arms.values():
                    self.assertEqual(arm["fit_file_ids"], ["fit-a", "fit-b"])
            for path in output.glob("*.zip"):
                with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
                    self.assertIsNone(archive.testzip())
            with self.assertRaises(FileExistsError):
                probe.run(plan, protocol, output, loader=loader)

    def test_failure_preserved_without_reselection(self):
        plan = probe.build_plan(*metadata(), strict=False)
        protocol = {"kind": "feasibility-npz-execution-v1",
                    "execute_npz_train_probe": True, "plan_sha256": probe.digest(probe.canonical(plan))}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            def loader(*args):
                raise ValueError("Synthetic missing NPZ")
            with self.assertRaisesRegex(ValueError, "missing NPZ"):
                probe.run(plan, protocol, output, loader=loader)
            self.assertFalse(json.loads((output / "blocker.json").read_text())["selection_changed"])
            self.assertTrue((output / "freeze.json").exists())


if __name__ == "__main__":
    unittest.main()
