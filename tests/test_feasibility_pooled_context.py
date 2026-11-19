"""Synthetic-only checks for the single report120 candidate."""
import tempfile
import json
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from compression.feasibility_tables import Tables, integer_frequencies
from finetuning import run_feasibility_pooled_context as p


def window():
    q = np.tile(np.asarray([[1, -1], [-1, 1]], dtype=np.int32), (500, 1))
    return dict(q=q, positions=np.asarray([], dtype=np.uint32),
                bits=np.asarray([], dtype=np.uint32), role="fit",
                group_id="synthetic", channel_ids=[2, 3],
                raw=(q * .2).astype(np.float32))


class PoolTests(unittest.TestCase):
    def test_exact_mapping_and_distinct_temporal_states(self):
        self.assertEqual(len(p.CONTEXT_GROUPS), 324)
        self.assertEqual(len(set(p.CONTEXT_GROUPS)), 180)
        for t in range(18):
            row = p.CONTEXT_GROUPS[t * 18:(t + 1) * 18]
            self.assertEqual(row[0], t * 10)
            self.assertEqual(row[17], t * 10 + 9)
            np.testing.assert_array_equal(row[1:9], row[9:17])
        self.assertEqual(p.CONTEXT_GROUPS[18], 10)

    def test_pool_counts_before_smoothing_threshold_and_unseen_row_zero(self):
        w = window()
        # Two sign contexts individually below 256; pooled count is 300.
        w["q"] = np.ones((300, 1), dtype=np.int32)
        contexts = np.asarray([1] * 150 + [9] * 150).reshape(300, 1)
        with patch.object(p, "contexts", return_value=contexts):
            model = p.fit_pooled([w], .1, 15, [2, 3])
        self.assertEqual(len(model.frequencies), 2)
        self.assertEqual(model.mapping[1], 1)
        self.assertEqual(model.mapping[9], 1)
        self.assertEqual(model.mapping[0], 0)
        self.assertEqual(model.mapping[17], 0)
        self.assertEqual(model.mapping[19], 0)
        counts = np.zeros(32)
        counts[16] = 300
        marginal = (counts + .5) / (300 + 16)
        expected = 300 / 556 * marginal + 256 / 556 * marginal
        np.testing.assert_array_equal(model.frequencies[1], integer_frequencies(expected))

    def test_below_threshold_backs_off(self):
        w = window()
        w["q"] = np.ones((255, 1), dtype=np.int32)
        with patch.object(p, "contexts", return_value=np.ones((255, 1), dtype=np.int32)):
            model = p.fit_pooled([w], .1, 15, [2, 3])
        self.assertEqual(len(model.frequencies), 1)
        self.assertFalse(np.any(model.mapping))

    def test_original_and_portable_decoders_unchanged_roundtrip(self):
        w = window()
        model = p.fit_pooled([w], .1, 15, [2, 3])
        blob = model.dump()
        self.assertEqual(blob[:4], b"FTB1")
        loaded = Tables.load(blob)
        self.assertEqual(len(loaded.mapping), 324)
        self.assertLessEqual(len(loaded.frequencies), 181)
        record, _ = p.a.evaluate(w, loaded, blob)
        self.assertTrue(record["portable_roundtrip"])
        self.assertTrue(record["waveform_bit_exact"])
        self.assertTrue(record["q_roundtrip"])

    def test_fit_rejects_check_role(self):
        w = window()
        w["role"] = "check"
        with self.assertRaises(ValueError):
            p.fit_pooled([w], .1, 15, [2, 3])

    def test_fit_retains_each_original_window_geometry(self):
        contiguous, gapped = window(), window()
        gapped["channel_ids"] = [2, 5]
        model = p.fit_pooled([contiguous, gapped], .1, 15, [10, 11])
        self.assertEqual(model.channel_ids, [10, 11])
        self.assertEqual(len(model.mapping), 324)
        # A physical gap must contribute to unavailable, not adjacent-neighbor groups.
        c = p.contexts(gapped["q"], gapped["positions"], [2, 5], "spatiotemporal")
        self.assertTrue(np.all(c % 18 == 17))

    def test_hash_mismatch_is_blocker(self):
        r = dict(q_sha256="a", waveform_sha256="b", original_float32_sha256="c")
        p.check_hashes(r, dict(r))
        for key in r:
            with self.assertRaises(RuntimeError):
                p.check_hashes(r, {**r, key: "changed"})

    def test_final_protocol_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "progress_reports").mkdir()
            with self.assertRaises(ValueError):
                p.authorization(root)
            path = root / "progress_reports/120-SYNTHETIC.md"
            path.write_text("# Metas provisórias\n")
            with self.assertRaises(ValueError):
                p.authorization(root)
            path.write_text("# Compatibilidade de tabela contextual\n")
            self.assertEqual(p.authorization(root), "progress_reports/120-SYNTHETIC.md")

    def test_primary_three_percent_and_configuration_guardrail(self):
        c = dict(ratio=.03, horizon=20, total_bytes=970,
                 vs_spatiotemporal={"total_bytes_saving": 30})
        aggregate = [c, {**c, "ratio": .1, "total_bytes": 10000}]
        groups = [{**c, "group_id": "one"}]
        self.assertTrue(p.primary_decision(aggregate, groups)["passes"])
        groups[0].update(total_bytes=1011,
                         vs_spatiotemporal={"total_bytes_saving": -11})
        self.assertFalse(p.primary_decision(aggregate, groups)["passes"])
        aggregate[0] = {**c, "total_bytes": 971,
                        "vs_spatiotemporal": {"total_bytes_saving": 29}}
        self.assertFalse(p.primary_decision(aggregate, [{**c, "group_id": "one"}])["passes"])

    def test_plan_metadata_only_and_exact_retry_population(self):
        with patch.object(np, "load", side_effect=AssertionError("No waveform in planning")):
            plan = p.build_plan()
        self.assertEqual(len(plan["rows"]), 380)
        self.assertEqual(len(plan["fit_rows"]), 18)
        self.assertEqual(plan["candidate"], p.CANDIDATE)
        self.assertTrue(plan["source_sha256"])
        self.assertEqual(plan["waveform_cells_read"], 0)

    def test_synthetic_freeze_hashes_transport_and_one_installation_per_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "progress_reports/120-SYNTHETIC.md"
            protocol.parent.mkdir()
            protocol.write_text("# Compatibilidade de tabela contextual\n")
            decoder = (p.ROOT / "compression/feasibility_portable.py").read_bytes()
            (root / "compression").mkdir()
            (root / "compression/feasibility_portable.py").write_bytes(decoder)
            model = p.fit_pooled([window()], .1, 15, [2, 3])
            table = model.dump()
            archive = p.a.zip_bytes({p.a.digest(table) + ".ftb": table,
                                      "decoder.py": decoder,
                                      "platform.json": b'{"platform":"Python 3 standard library"}'})
            (root / "original.zip").write_bytes(archive)
            fit = dict(group_id="synthetic", sample_path="fit.npz", role="fit", drive_id="fit")
            rows = [dict(group_id="synthetic", sample_path=f"{fid}.npz",
                         role="check", drive_id=fid) for fid in ("file-a", "file-b")]
            baseline_rows = []
            reference_curves, reference_files, reference_aggregates = [], [], []
            record, _ = p.a.evaluate(window(), Tables.load(table), table)
            for mode in ("marginal", "spatiotemporal"):
                for row in rows:
                    baseline_rows.append({**record, **row, "ratio": .03, "mode": mode})
                    reference_files.append({**row, "ratio": .03, "mode": mode,
                                            "windows": 1, "cells": 2000,
                                            "transport_bytes": 2000, "integer_cdf_nll_bits": 8000})
                for horizon in range(1, 21):
                    c = dict(ratio=.03, horizon=horizon, mode=mode, windows=2,
                             cells=4000, transport_bytes=4000, total_bytes=8000,
                             installation_bytes=4000, integer_cdf_nll_bits=16000)
                    reference_curves.append({**c, "group_id": "synthetic"})
                    reference_aggregates.append(c)
            ref = root / p.REFERENCE
            ref.mkdir(parents=True)
            p.a.write_json(ref / "results.json", dict(
                rows=baseline_rows, curves=reference_curves,
                aggregates=reference_aggregates, file_curves=reference_files))
            plan = dict(candidate=p.CANDIDATE, authorization=str(protocol.relative_to(root)),
                        f4_authorized=False, input_sha256={},
                        runtime={"executable_sha256": p.a.digest(Path(sys.executable).read_bytes())},
                        fit_rows=[fit], rows=rows, fit_geometry={"fit.npz": [2, 3]},
                        geometry={r["sample_path"]: [2, 3] for r in rows},
                        coverage={"synthetic": {}}, files={r["drive_id"]: {"selected": 1} for r in rows},
                        ratios=[.03], checkpoints=[1, 5, 10, 20],
                        installations={"synthetic|0.03|spatiotemporal": dict(
                            path="original.zip", epsilon=.1, k=15, channel_ids=[2, 3])})
            output = root / "output"
            reads = []
            def loader(root, row, ids):
                self.assertTrue((output / "freeze.json").exists())
                if row["role"] == "check":
                    self.assertTrue((output / "group00.fit_freeze.json").exists())
                reads.append(row["sample_path"])
                return {**window(), **row}
            result = p.run(plan, output, root, loader)
            self.assertEqual(reads, ["fit.npz", "file-a.npz", "file-b.npz"])
            self.assertEqual(len(result["installations"]), 1)
            self.assertEqual(result["npz_cell_occurrences_read"], 6000)
            self.assertTrue(all(r["matches_report116_hashes"] for r in result["rows"]))
            for c in result["curves"]:
                self.assertEqual(c["total_bytes"], c["transport_bytes"] + c["installation_bytes"])
                self.assertEqual(c["installation_bytes"], result["installations"][0]["bytes"])
                if c["horizon"] in plan["checkpoints"]:
                    payload = (output / f"group00_0.03_pooled.n{c['horizon']:02d}.transport.zip").read_bytes()
                    self.assertEqual(len(payload), c["transport_bytes"])
                    self.assertEqual(p.a.digest(payload), c["transport_sha256"])
                    with zipfile.ZipFile(io.BytesIO(payload)) as z:
                        self.assertEqual(len(z.namelist()), 2)
            freeze = json.loads((output / "freeze.json").read_text())
            self.assertEqual(freeze["plan_sha256"], p.a.digest(p.a.canonical(plan)))
            with self.assertRaises(FileExistsError):
                p.run(plan, output, root, loader)


if __name__ == "__main__":
    unittest.main()
