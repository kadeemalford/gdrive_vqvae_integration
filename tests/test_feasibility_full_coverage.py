import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from compression.full_tile_vq_stream import FullTileVQWire
from compression.conditioned_vq_stream import ConditionedVQWire, unpack_context
from compression.partition_bundle import unpack_bundle
from finetuning.robust_escape import encode_robust
from finetuning.run_feasibility_full_coverage import (
    OVERLAY, coverage_guards, decode_coverage, decode_zero_overlay, encode_coverage,
    no_regression, operation_budget, pack_zero_overlay, paired_gate, partition,
    rate_ledger, validate_raw,
)
from finetuning.streaming_physical_summary import StreamingPhysicalSummary
from finetuning.feasibility_metrics import target_supported_spectra
from tests.test_joint_anchor_vq import ToyCodec


class CoverageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = ToyCodec().eval()
        self.sender = FullTileVQWire(self.model, np.ones(2048, np.int64), "0" * 64)
        self.receiver = FullTileVQWire.from_setup(self.model, self.sender.setup_bytes())

    def test_partition_no_overlap_tail_and_budget(self):
        for width in range(1, 1025):
            pieces = partition(width, "part8")
            self.assertEqual([i for a, b in pieces for i in range(a, b)], list(range(width)))
            self.assertTrue(all(0 < b - a <= 8 for a, b in pieces))
        self.assertEqual(partition(19, "part8"), [(0, 8), (8, 16), (16, 19)])
        budget = operation_budget([{"channels": 19}])
        self.assertEqual(budget["encode_calls"], 8)
        self.assertEqual(budget["decoder_forwards_including_encode"], 24)
        with self.assertRaises(ValueError):
            operation_budget([{"channels": 10000}])

    def test_full_and_partition_decode_without_encoder_and_zero_bits(self):
        raw = np.random.default_rng(109).normal(3, 1, (19, 1000)).astype(np.float32)
        raw[0, :2] = [0., -0.]
        original = raw.tobytes()
        for arm in ("full", "part8"):
            bundle = encode_coverage(raw, self.sender, arm)
            shape, fragments = unpack_bundle(bundle)
            self.assertEqual(shape, raw.shape)
            self.assertEqual(len(fragments), len(partition(19, arm)))
            with patch.object(self.sender, "encode", side_effect=AssertionError("encoder forbidden")), \
                    patch("compression.full_tile_vq_stream.exact_branch_forward",
                          side_effect=AssertionError("encoder forward forbidden")):
                first = decode_coverage(bundle, self.receiver)
                second = decode_coverage(bundle, self.receiver)
            self.assertEqual(first.tobytes(), second.tobytes())
            self.assertEqual(first[0, :2].tobytes(), raw[0, :2].tobytes())
            self.assertEqual(raw.tobytes(), original)

    def test_overlay_does_not_change_historical_normalized_input(self):
        raw = np.random.default_rng(12).normal(3, 1, (3, 1000)).astype(np.float32)
        raw[0, :2] = [0., -0.]
        expected = encode_robust(raw).values.copy()
        with patch("compression.full_tile_vq_stream.encode_robust", wraps=encode_robust) as spy:
            bundle = encode_coverage(raw, self.sender, "full")
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[0].tobytes(), raw.tobytes())
        self.assertEqual(expected.tobytes(), encode_robust(raw).values.tobytes())
        self.assertNotEqual(float(expected[0, 0]), 0.)
        _, fragments = unpack_bundle(bundle)
        wrapped = fragments[0][3]
        _, size, count, _ = OVERLAY.unpack_from(wrapped)
        self.assertEqual(count, 2)
        native = self.receiver.decode(wrapped[OVERLAY.size:OVERLAY.size + size])
        restored = decode_coverage(bundle, self.receiver)
        mask = raw != 0
        self.assertEqual(native[mask].tobytes(), restored[mask].tobytes())

    def test_constant_signed_zero_literals(self):
        for value in (0., -0., 2.):
            raw = np.full((9, 1000), value, np.float32)
            for arm in ("full", "part8"):
                restored = decode_coverage(encode_coverage(raw, self.sender, arm), self.receiver)
                self.assertEqual(restored.tobytes(), raw.tobytes())

    def test_film_context_paid_for_both_arms(self):
        config = {"method": "constant", "plan": {"seen": []}}
        sender = ConditionedVQWire.from_setup(self.model, self.sender.setup_bytes(), config)
        receiver = ConditionedVQWire.from_setup(self.model, self.sender.setup_bytes(), config)
        raw = np.ones((9, 1000), np.float32)
        for arm in ("full", "part8"):
            bundle = encode_coverage(raw, sender, arm, "test/group")
            _, fragments = unpack_bundle(bundle)
            for _, _, _, wrapped in fragments:
                _, size, _, _ = OVERLAY.unpack_from(wrapped)
                vector, _ = unpack_context(wrapped[OVERLAY.size:OVERLAY.size + size], "constant")
                self.assertEqual(vector.tobytes(), np.zeros(11, np.float32).tobytes())
            self.assertEqual(decode_coverage(bundle, receiver).tobytes(), raw.tobytes())

    def test_nonfinite_rejected_before_encode_and_corruption(self):
        for value in (np.nan, np.inf, -np.inf):
            raw = np.ones((1, 1000), np.float32)
            raw[0, 10] = value
            with patch.object(self.sender, "encode", side_effect=AssertionError("must not encode")):
                with self.assertRaises(ValueError):
                    encode_coverage(raw, self.sender, "full")
        raw = np.ones((1, 1000), np.float32)
        frame = pack_zero_overlay(self.sender.encode(raw), raw)
        for corrupt in (frame[:5], frame[:-1] + bytes([frame[-1] ^ 1])):
            with self.assertRaises(ValueError):
                decode_zero_overlay(corrupt, self.receiver)
        with self.assertRaises(ValueError):
            validate_raw(raw.astype(np.float64))

    def test_guards_include_all_channels_sse_spectra_missing_phase(self):
        raw = np.sin(np.arange(1000)[None] * .2).astype(np.float32)
        raw = np.repeat(raw, 9, axis=0)
        predicted = np.zeros_like(raw)
        row = {"group_id": "doped/g", "drive_id": "f", "channel_map_id": "m"}
        summary = StreamingPhysicalSummary({"doped/g": .001}, {"m": {"source_channel_indices": list(range(9))}})
        summary.update(row, raw, predicted, 100)
        result = summary.result(10)
        spectra = target_supported_spectra(raw, predicted, 1 / 20000, .001)
        guards = coverage_guards(result, [{"group_id": "doped/g", "spectral_statistics": spectra}])
        group = guards["doped/g"]
        self.assertAlmostEqual(group["sse"], float(np.sum(raw.astype(np.float64) ** 2)))
        supported = [v for v in group["spectral"].values() if v["supported_target_energy"] > 0]
        self.assertTrue(supported)
        self.assertTrue(all(v["missing_phase_energy_fraction"] == 1 for v in supported))
        self.assertEqual(len(result["records"][0]["channels"]), 9)

    def test_complete_rate_ledger_no_double_counting(self):
        ledger = rate_ledger(100, 20, 30, 1000, 200, 10, 1000)
        self.assertEqual(ledger["installed_shared_bytes"], 1240)
        self.assertEqual(ledger["shared_model"]["bytes"], 120)
        self.assertEqual(ledger["self_contained"]["bytes"], 1360)
        self.assertEqual(ledger["self_contained"]["bpc"], 10.88)

    def test_gate_all_required_guards_and_zero_baselines(self):
        baseline = {"macro_families": 1., "families": {"f": 1.},
                    "rate": {"shared_model": {"bytes": 100}, "self_contained": {"bytes": 1000}},
                    "coverage_guards": {"f/g": {"sse": 1., "p90_channel_nrmse_star": 1., "spectral": {
                        "band": {"target_energy": 1., "supported_target_energy": 1.,
                                 "relative_complex_rmse": 1.}}}}}
        candidate = copy.deepcopy(baseline)
        candidate["macro_families"] = .95
        self.assertTrue(paired_gate(baseline, candidate)["passed"])
        for mutate in (
            lambda c: c["families"].update(f=1.011),
            lambda c: c["coverage_guards"]["f/g"].update(p90_channel_nrmse_star=1.011),
            lambda c: c["coverage_guards"]["f/g"].update(sse=1.001),
            lambda c: c["coverage_guards"]["f/g"]["spectral"]["band"].update(relative_complex_rmse=1.011),
            lambda c: c["rate"]["shared_model"].update(bytes=101),
            lambda c: c.update(macro_families=.951),
        ):
            trial = copy.deepcopy(candidate)
            mutate(trial)
            self.assertFalse(paired_gate(baseline, trial)["passed"])
        self.assertTrue(no_regression(0, 0))
        self.assertFalse(no_regression(1e-12, 0))
        self.assertIsNone(no_regression(None, 0))
        self.assertFalse(paired_gate(baseline, candidate)["starts_f4"])


if __name__ == "__main__":
    unittest.main()
