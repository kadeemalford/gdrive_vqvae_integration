"""Synthetic-only routing tests: no FEBUS waveform access."""
import inspect
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from finetuning import run_feasibility_fit_routing as routing


class FitRoutingTests(unittest.TestCase):
    def fold(self, fid, candidate, baseline=10000):
        return {"held_out_drive_id": fid, "marginal_total_bytes": baseline,
                "spatiotemporal_total_bytes": candidate}

    def test_threshold_exact_equality_and_every_file(self):
        folds = [self.fold("a", 9700), self.fold("b", 9600)]
        self.assertEqual(routing.select_mode(folds, 2), "spatiotemporal")
        folds[1] = self.fold("b", 9701)
        self.assertEqual(routing.select_mode(folds, 2), "marginal")

    def test_less_than_two_fit_files_is_marginal(self):
        self.assertEqual(routing.select_mode([], 0), "marginal")
        self.assertEqual(routing.select_mode([self.fold("a", 1)], 1), "marginal")

    def test_incomplete_duplicate_and_zero_denominator(self):
        with self.assertRaises(ValueError):
            routing.select_mode([self.fold("a", 1)], 2)
        with self.assertRaises(ValueError):
            routing.select_mode([self.fold("a", 1), self.fold("a", 1)], 2)
        self.assertEqual(routing.select_mode([self.fold("a", 0, 0),
                                            self.fold("b", 1)], 2), "marginal")

    def test_holdout_entire_file_and_fit_only(self):
        windows = [{"drive_id": f, "role": "fit"} for f in ("a", "a", "b", "b")]
        train, held = routing.split_fit(windows, "a")
        self.assertEqual([w["drive_id"] for w in train], ["b", "b"])
        self.assertEqual([w["drive_id"] for w in held], ["a", "a"])
        with self.assertRaises(ValueError):
            routing.split_fit(windows + [{"drive_id": "c", "role": "check"}], "a")

    def test_shared_source_once_both_arms_route_paid(self):
        rows = [{"ratio": ratio, "group_id": str(i), "family": str(i % 4), "cells": 100,
                 "marginal_payload_bytes": 1000, "selected_payload_bytes": 800,
                 "saving_bytes": 200} for ratio in routing.RATIOS for i in range(9)]
        summaries = routing.aggregate(rows, 123)
        shared = summaries[0]
        self.assertEqual(shared["marginal_total_bytes"], 9 * 1001 + 123)
        self.assertEqual(shared["selected_total_bytes"], 9 * 801 + 123)
        self.assertEqual(shared["route_bytes_per_arm"], 9)
        self.assertTrue(shared["measured_coverage_thresholds_pass"])
        self.assertEqual(summaries[1]["marginal_total_bytes"], 9 * (1001 + 123))

    def test_equal_family_is_not_success_even_if_rate_passes(self):
        rows = [{"ratio": ratio, "group_id": str(i), "family": str(i % 4), "cells": 100,
                 "marginal_payload_bytes": 1000,
                 "selected_payload_bytes": 1000 if i % 4 == 0 else 500,
                 "saving_bytes": 0 if i % 4 == 0 else 500}
                for ratio in routing.RATIOS for i in range(9)]
        for summary in routing.aggregate(rows, 1):
            self.assertTrue(summary["aggregate_rate_thresholds_pass"])
            self.assertTrue(summary["no_configuration_regression"])
            self.assertFalse(summary["measured_coverage_thresholds_pass"])
            self.assertFalse(summary["strictly_positive_all_families"])

    def test_check_stage_requires_decisions_before_ledgers(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(routing, "fit_windows", side_effect=AssertionError("fit forbidden")):
                with self.assertRaises(FileNotFoundError):
                    routing.evaluate_frozen_check({}, Path(tmp), time.monotonic())

    def test_check_stage_has_no_encoder_or_waveform_calls(self):
        source = inspect.getsource(routing.evaluate_frozen_check)
        for forbidden in ("evaluate_window(", "load_region(", "quantized_window(",
                          "choose_support(", "fit_windows(", "pack("):
            # unpack is the permitted exact-byte installation decoder.
            self.assertNotIn(forbidden, source.replace("unpack(", "restore("))

    def test_source_integrity_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.py"
            path.write_text("original")
            expected = routing.identity(path)
            routing.verify(path, expected)
            path.write_text("modified")
            with self.assertRaises(ValueError):
                routing.verify(path, expected)

    def test_budget(self):
        with self.assertRaises(TimeoutError):
            routing.check_budget(time.monotonic() - 3601)

    def test_active_report_required_before_freeze(self):
        with patch.object(routing, "REPORT", Path("/nonexistent/report120.md")):
            with self.assertRaisesRegex(ValueError, "report120"):
                routing.freeze()

    def test_synthetic_real_helper_bundle_roundtrip_and_accounting(self):
        rng = np.random.default_rng(120)
        windows = []
        for fid in ("a", "b"):
            raw = rng.normal(size=(30, 3)).astype("f4")
            raw.view("u4").flat[:4] = [0x80000000, 0x7fc00123, 0x7f800000, 0xff800000]
            windows.append(routing.quantized_window({
                "raw": raw, "channel_ids": [2, 3, 4], "group_id": "synthetic/test",
                "role": "fit", "drive_id": fid, "dataset": "synthetic",
                "block_index": 0, "region_index": 0, "source_time_start": 0,
                "time_samples": 30, "sample_path": fid + "/r0/t00000"}, .03))
        with tempfile.TemporaryDirectory() as tmp:
            arms = []
            for mode in routing.MODES:
                train, held = routing.split_fit(windows, "b")
                model = routing.fit_windows(train, .03, 15, mode)
                result = routing.fit_arm(held, model, .03, "synthetic stdlib", 123,
                                         Path(tmp) / mode)
                self.assertEqual(result["total_bytes"],
                                 result["transport_bytes"] + result["envelope_bytes"] + 124)
                self.assertTrue(result["records"][0]["portable_roundtrip"])
                self.assertEqual(result["records"][0]["bound_violations"], 0)
                self.assertEqual(result["records"][0]["original_float32_sha256"],
                                 routing.digest(held[0]["raw"].astype("<f4").tobytes()))
                arms.append(result)
            self.assertEqual(arms[0]["records"][0]["q_sha256"], arms[1]["records"][0]["q_sha256"])
            self.assertEqual(arms[0]["records"][0]["waveform_sha256"],
                             arms[1]["records"][0]["waveform_sha256"])


if __name__ == "__main__":
    unittest.main()
