import copy
import unittest
from unittest.mock import patch

import numpy as np

from finetuning.feasibility_context import plan_context_windows
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust
from finetuning.run_feasibility_context import (
    MATURE_SHA, MODELS, NORMS, context_pairs, normalize, physical_inverse,
    score_core, slice_packet, validate_panel, validate_weights_manifest,
)


def manifest():
    base = {"latent_geometry": "spatial4-temporal4", "num_embeddings": 2048,
            "embedding_dim": 128}
    return {
        MODELS[0]: {"path": "mature.pt", "sha256": MATURE_SHA, "update": None,
                    "config": {**base, "architecture": "native-v1"}},
        MODELS[1]: {"path": "diversity.pt", "sha256": "1" * 64, "update": 8192,
                    "config": {"architecture": "dense-full-tile-product-v1",
                               "variant": "flat", "base_config": base}},
        MODELS[2]: {"path": "film.pt", "sha256": "2" * 64, "update": 16384,
                    "config": {"architecture": "physical-film-v1", "method": "constant",
                               "base_config": base}},
    }


class ContextNormalizationTests(unittest.TestCase):
    def test_fixed_slice_never_refits_quantiles_or_affine(self):
        raw = np.arange(64 * 1000, dtype=np.float32).reshape(64, 1000) + 1
        raw[20, 17] = 1e30
        raw[50, 31] = -1e30
        full = normalize(raw)
        with patch("finetuning.run_feasibility_context.encode_robust",
                   side_effect=AssertionError("must not refit")):
            cropped = slice_packet(full, 16, 32)
        np.testing.assert_array_equal(cropped.values, full.values[16:32])
        np.testing.assert_array_equal(cropped.valid, full.valid[16:32])
        self.assertEqual((cropped.mean, cropped.scale), (full.mean, full.scale))
        self.assertEqual(cropped.indices.tolist(), [4 * 1000 + 17])
        restored = physical_inverse(cropped.values, cropped)
        self.assertEqual(restored[4, 17].tobytes(), raw[20, 17].tobytes())
        full_restored = physical_inverse(full.values, full)
        self.assertEqual(restored.tobytes(), full_restored[16:32].tobytes())
        self.assertNotEqual(normalize(raw[16:32]).mean, cropped.mean)

    def test_full_native_fixed_bit_identity(self):
        raw = np.random.default_rng(107).normal(size=(32, 1000)).astype(np.float32)
        full, native = normalize(raw), normalize(raw)
        fixed = slice_packet(full, 0, 32)
        self.assertEqual(native.side_stream(), fixed.side_stream())
        self.assertEqual(native.values.tobytes(), fixed.values.tobytes())
        self.assertEqual(physical_inverse(native.values, native).tobytes(),
                         physical_inverse(fixed.values, fixed).tobytes())

    def test_historical_inverse_and_normalized_values_unchanged(self):
        raw = np.random.default_rng(3).normal(size=(8, 1000)).astype(np.float32)
        raw[0, 1] = 1e30
        old, new = encode_robust(raw), normalize(raw)
        np.testing.assert_array_equal(old.values, new.values)
        prediction = new.values * np.float32(.77)
        self.assertEqual(decode(prediction, old.side_stream).tobytes(),
                         physical_inverse(prediction, new).tobytes())

    def test_exact_zero_escapes_do_not_change_the_historical_model_input(self):
        raw = np.arange(8000, dtype=np.float32).reshape(8, 1000)
        raw[2, 8] = -0.
        old, new = encode_robust(raw), normalize(raw)
        self.assertEqual(old.values.tobytes(), new.values.tobytes())
        self.assertNotEqual(float(new.values[0, 0]), 0.)
        restored = physical_inverse(np.ones_like(new.values), new)
        self.assertEqual(restored[raw == 0].tobytes(), raw[raw == 0].tobytes())

    def test_exact_special_payloads_and_signed_zeros(self):
        raw = np.ones((8, 1000), np.float32)
        raw[3, :6] = np.array([0x7fc01234, 0xffc04321, 0x7f800000,
                              0xff800000, 0, 0x80000000], np.uint32).view(np.float32)
        full = normalize(raw)
        packet = slice_packet(full, 2, 5)
        result = physical_inverse(np.zeros_like(packet.values), packet)
        self.assertEqual(result[1, :6].tobytes(), raw[3, :6].tobytes())
        self.assertFalse(packet.valid[1, :6].any())

    def test_constants_and_all_specials(self):
        for value in (0., -0., 123.5, -3., np.inf, np.nan):
            raw = np.full((2, 1000), value, np.float32)
            packet = normalize(raw)
            self.assertEqual(physical_inverse(packet.values, packet).tobytes(), raw.tobytes())

    def test_no_manufactured_channels(self):
        raw = np.ones((2, 1000), np.float32)
        packet = normalize(raw)
        self.assertEqual(len(plan_context_windows(2)["windows"]), 1)
        for start, stop in ((0, 8), (-1, 1), (1, 1)):
            with self.assertRaises(ValueError):
                slice_packet(packet, start, stop)


class ContextScoringTests(unittest.TestCase):
    def test_scores_physical_core_and_time_axis_only(self):
        raw = np.repeat(np.arange(64, dtype=np.float32)[:, None], 1000, axis=1) + 1
        plan = plan_context_windows(64)
        for window in plan["windows"]:
            prediction = raw[window["start"]:window["stop"]].copy()
            prediction += 100
            a, b = window["score_start_in_window"], window["score_stop_in_window"]
            prediction[a:b] = raw[plan["core_start"]:plan["core_stop"]] + 2
            score = score_core(raw, prediction, plan, window, 1., [107, 108])
            self.assertEqual(score["rmse"], 2.)
            self.assertEqual(score["finite_cells"], 2000)
            self.assertEqual(len(score["channels"]), 2)
            for record in score["channels"]:
                self.assertEqual(record["physical"]["bias"], 2.)
                self.assertTrue(record["constant_target"])
                self.assertIsNone(record["physical"]["correlation"])
                self.assertEqual(record["spectral_statistics"]["0_1khz"]["total_bins"], 49)

    def test_nonfinite_spectrum_unavailable_not_zero_filled(self):
        raw = np.ones((2, 1000), np.float32)
        raw[0, 4] = np.nan
        plan = plan_context_windows(2)
        score = score_core(raw, raw.copy(), plan, plan["windows"][0], 1., [1, 2])
        self.assertEqual(score["finite_cells"], 1999)
        self.assertFalse(score["channels"][0]["spectral_available"])
        self.assertIsNone(score["channels"][0]["spectral"])
        self.assertTrue(score["channels"][1]["spectral_available"])

    def records(self):
        records = []
        for group in ("a", "b"):
            for file in ("1", "2"):
                for width, rmse in ((8, 1.), (16, 1.1), (32, .8)):
                    records.append({"model": MODELS[0], "normalization": NORMS[0],
                                    "sample_path": group + file, "drive_id": group + file,
                                    "group_id": group, "context_channels": width,
                                    "metrics": {"rmse": rmse}})
        return records

    def test_gate_requires_same_model_norm_pair_two_files_two_groups(self):
        records = self.records()
        pairs, gates = context_pairs(records)
        self.assertEqual(len(pairs), 12)
        self.assertTrue(all(g["descriptive_gate_met"] for g in gates))
        self.assertTrue(any(p["relative_difference"] < 0 for p in pairs))
        records[3]["model"] = MODELS[1]
        records[9]["normalization"] = NORMS[1]
        _, gates = context_pairs(records)
        self.assertFalse(any(g["descriptive_gate_met"] for g in gates if 8 in g["context_pair"]))

    def test_zero_rmse_ratio_undefined_with_absolute_difference(self):
        records = self.records()[:2]
        records[0]["metrics"]["rmse"] = 0.
        pairs, _ = context_pairs(records)
        self.assertIsNone(pairs[0]["relative_difference"])
        self.assertEqual(pairs[0]["absolute_difference"], 1.1)
        self.assertFalse(pairs[0]["threshold_met"])


class ContextManifestTests(unittest.TestCase):
    def test_accepts_cpu_inspection_schema(self):
        value = manifest()
        value[MODELS[0]]["keys"] = ["config", "model_state_dict"]
        value[MODELS[0]]["device_inspection"] = "cpu"
        self.assertIs(validate_weights_manifest(value), value)

    def test_rejects_wrong_weights_config_updates_and_missing_models(self):
        changes = [(MODELS[0], "sha256", "0" * 64),
                   (MODELS[1], "update", 4096), (MODELS[2], "update", 8192),
                   (MODELS[0], "path", ""), (MODELS[1], "sha256", "no")]
        for name, key, bad in changes:
            value = manifest()
            value[name][key] = bad
            with self.assertRaises(ValueError):
                validate_weights_manifest(value)
        for name, key, bad in ((MODELS[2], "method", "hardware"),
                               (MODELS[1], "variant", "product")):
            value = copy.deepcopy(manifest())
            value[name]["config"][key] = bad
            with self.assertRaises(ValueError):
                validate_weights_manifest(value)
        value = manifest()
        del value[MODELS[1]]
        with self.assertRaises(ValueError):
            validate_weights_manifest(value)

    def panel(self):
        # All full-width rows have 80 windows total; that budget must be rejected.
        rows = [{"drive_id": str(i), "sample_path": f"data/train/{i}.npz",
                 "split": "train", "group_id": f"doped/{i // 2}", "channels": "64"}
                for i in range(20)]
        return {"kind": "feasibility-metadata-freeze-v1",
                "f1_panel": {"rows": rows, "context_plans": {
                    r["sample_path"]: plan_context_windows(64) for r in rows}}}

    def test_accepts_frozen_414_forward_budget_without_waveform_reads(self):
        metadata = self.panel()
        rows = metadata["f1_panel"]["rows"]
        for index, width in enumerate((2, 2, 2, 32, 32)):
            rows[index]["channels"] = str(width)
            metadata["f1_panel"]["context_plans"][rows[index]["sample_path"]] = plan_context_windows(width)
        with patch("numpy.load", side_effect=AssertionError("must not open NPZ")):
            selected, forwards = validate_panel(metadata)
        self.assertEqual(forwards, 414)
        self.assertEqual(len(selected), 20)

    def test_reserved_rejected_before_any_npz_access(self):
        for split in ("val", "test"):
            metadata = self.panel()
            metadata["f1_panel"]["rows"][0]["split"] = split
            with patch("numpy.load", side_effect=AssertionError("must not open NPZ")):
                with self.assertRaisesRegex(ValueError, "train-panel"):
                    validate_panel(metadata)
        metadata = self.panel()
        metadata["f1_panel"]["rows"][0]["group_id"] = "EDAS Polyimide/external"
        with self.assertRaises(ValueError):
            validate_panel(metadata)

    def test_changed_plan_and_budget_are_rejected(self):
        metadata = self.panel()
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_panel(metadata)
        metadata["f1_panel"]["context_plans"]["data/train/0.npz"]["core_start"] = 0
        with self.assertRaisesRegex(ValueError, "plan differs"):
            validate_panel(metadata)


if __name__ == "__main__":
    unittest.main()
