"""Synthetic-only conformance tests for the direct feasibility coder."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

import numpy as np

from compression.feasibility_tables import (
    TOTAL, Tables, bucket, contexts, fit_tables, integer_frequencies,
    quantize, reconstruct, score,
)
from compression.feasibility_stream import FRAME, decode, encode, portable_object, select_support
from compression.feasibility_portable import decode_bytes


class FeasibilityCoderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1701)
        self.x = rng.normal(size=(1031, 3)).astype(np.float32)
        self.epsilon = .03
        self.q, self.pos, self.bits = quantize(self.x, self.epsilon)
        self.ids = [2, 3, 7]

    def model(self, mode="spatiotemporal", k=15):
        return fit_tables(self.q, self.pos, self.epsilon, self.ids, k, mode)

    def test_integer_apportionment_and_ties(self):
        p = np.ones(3) / 3
        f = integer_frequencies(p)
        self.assertEqual(int(f.sum()), TOTAL)
        self.assertEqual(f.tolist(), [349526, 349525, 349525])
        f = integer_frequencies([1., 0., 0.])
        self.assertEqual(f.tolist(), [TOTAL - 2, 1, 1])

    def test_all_modes_identical_q_waveform_and_ledger(self):
        expected = reconstruct(self.q, self.pos, self.bits, self.epsilon)
        for mode in ("marginal", "temporal", "spatiotemporal"):
            model = self.model(mode)
            frozen = Tables.load(model.dump())
            self.assertEqual(model.dump(), frozen.dump())
            for forced in (None, 0, 1, 2):
                data, ledger = encode(self.q, self.pos, self.bits, model, forced)
                q, x = decode(data, model.dump())
                np.testing.assert_array_equal(q, self.q)
                np.testing.assert_array_equal(x.view("u4"), expected.view("u4"))
                pq, px = decode_bytes(data, model.dump())
                self.assertEqual(pq, q.astype("<i4").tobytes())
                self.assertEqual(px, expected.astype("<f4").tobytes())
                self.assertEqual(sum(ledger["components"].values()), len(data))
                self.assertEqual(ledger["table_object_bytes"], len(data) + len(model.dump()))
        self.assertLessEqual(float(np.max(np.abs(self.x.astype(float) - expected))), self.epsilon)

    def test_nearest_even_and_special_payloads(self):
        bits = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
                         0x7fc12345, 0x7fa12345, 0x7f7fffff, 0xff7fffff,
                         1, 0x80000001], dtype=np.uint32)
        x = bits.view("f4").reshape(-1, 1)
        q, pos, raw = quantize(x, 1e-4)
        tables = fit_tables(q, pos, 1e-4, [0], 15, "temporal")
        for forced in (0, 1, 2, None):
            data, _ = encode(q, pos, raw, tables, forced)
            qq, xx = decode(data, tables.dump())
            np.testing.assert_array_equal(qq, q)
            np.testing.assert_array_equal(xx.view("u4").reshape(-1)[pos], bits[pos])
            pq, px = decode_bytes(data, tables.dump())
            self.assertEqual(pq, q.astype("<i4").tobytes())
            self.assertEqual(px, xx.astype("<f4").tobytes())
        q, _, _ = quantize(np.array([[.5], [1.5], [2.5], [-.5]], dtype="f4"), .5)
        self.assertEqual(q[:, 0].tolist(), [0, 2, 2, 0])

    def test_alphabet_escapes_feed_representable_context(self):
        q = np.array([[100000, -200000], [0, 0]], dtype=np.int32)
        pos, bits = np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)
        tables = fit_tables(q, pos, .1, [0, 1], 15, "spatiotemporal")
        data, cost = encode(q, pos, bits, tables, forced=2)
        trace = []
        decoded, _ = decode(data, tables.dump(), trace=trace)
        np.testing.assert_array_equal(decoded, q)
        self.assertEqual(trace[2][1] // 18, bucket(100000))
        self.assertEqual(cost["components"]["alphabet_escapes"], 8)

    def test_causality_and_reset_with_physical_gaps(self):
        tables = self.model()
        trace = []
        data, _ = encode(self.q, self.pos, self.bits, tables, forced=2)
        decode(data, tables.dump(), trace=trace)
        expected = contexts(self.q, self.pos, self.ids, "spatiotemporal").reshape(-1)
        self.assertEqual([r[1] for r in trace], expected.tolist())
        self.assertEqual(trace[3000][1], 17 * 18 + 17)
        self.assertEqual(trace[2][1] % 18, 17)  # physical gap, no adjacent context
        changed = self.q.copy()
        changed[800:] = 999999
        c = contexts(changed, self.pos, self.ids, "spatiotemporal")
        np.testing.assert_array_equal(c[:800], expected.reshape(self.q.shape)[:800])
        data2, _ = encode(changed, self.pos, self.bits, tables, forced=2)
        trace2 = []
        decode(data2, tables.dump(), trace=trace2)
        self.assertEqual(trace[:2400], trace2[:2400])
        # No fitted probabilities are needed by the decoder.
        self.assertIsNone(Tables.load(tables.dump()).probabilities)

    def test_physical_escape_unavailable_context(self):
        x = np.array([[np.nan, 1], [1, 1]], dtype="f4")
        q, pos, bits = quantize(x, .1)
        tables = fit_tables(q, pos, .1, [0, 1], 15, "spatiotemporal")
        data, _ = encode(q, pos, bits, tables, 2)
        trace = []
        decode(data, tables.dump(), trace=trace)
        self.assertEqual(trace[1][1] % 18, 17)
        self.assertEqual(trace[2][1] // 18, 17)

    def test_backoff_fit_only_and_score(self):
        model = self.model("temporal")
        c = contexts(self.q, self.pos, self.ids, "temporal").reshape(-1)
        for ctx in range(18):
            if np.count_nonzero(c == ctx) < 256:
                self.assertEqual(model.mapping[ctx], 0)
        before = model.dump()
        stats = score(model, self.q, self.pos)
        self.assertTrue(np.isfinite(stats["integer_cdf_nll_bits"]))
        encode(self.q * 10, self.pos, self.bits, model)
        self.assertEqual(before, model.dump())

    def test_fit_support_uses_real_cost_and_deterministic_tie(self):
        k, candidates = select_support(self.q[:23], self.pos, self.bits, .03, self.ids)
        self.assertEqual(k, min(candidates, key=lambda x: (x["table_object_bytes"], x["k"]))["k"])
        self.assertEqual(len(candidates), 5)

    def test_corrupted_and_truncated_bytes_rejected(self):
        tables = self.model()
        data, _ = encode(self.q, self.pos, self.bits, tables, forced=2)
        for i in (0, 4, 36, 44, 80, len(data) - 1):
            damaged = bytearray(data)
            damaged[i] ^= 1
            with self.assertRaises(ValueError):
                decode(bytes(damaged), tables.dump())
            with self.assertRaises(ValueError):
                decode_bytes(bytes(damaged), tables.dump())
        for n in (0, 43, 75, len(data) - 1):
            with self.assertRaises(ValueError):
                decode(data[:n], tables.dump())
        damaged = bytearray(tables.dump())
        damaged[-40] ^= 1
        with self.assertRaises(ValueError):
            decode(data, bytes(damaged))
        with self.assertRaises(ValueError):
            decode(data, tables.dump(), max_cells=1)

    def test_rehashed_malformed_ans_rejected(self):
        tables = self.model()
        data, _ = encode(self.q, self.pos, self.bits, tables, forced=2)
        damaged = bytearray(data)
        _, _, nphys, nesc, nbytes, _ = FRAME.unpack_from(damaged, 44)
        cursor = 44 + FRAME.size + nphys * 8 + nesc * 4
        damaged[cursor:cursor + 8] = b"\0" * 8
        end = cursor + nbytes
        struct.pack_into("<I", damaged, end, zlib.crc32(damaged[44:end]))
        damaged[-32:] = hashlib.sha256(damaged[:-32]).digest()
        with self.assertRaises(ValueError):
            decode(bytes(damaged), tables.dump())
        with self.assertRaises(ValueError):
            decode_bytes(bytes(damaged), tables.dump())

    def test_portable_object_without_site_packages(self):
        tables = self.model()
        transport, _ = encode(self.q, self.pos, self.bits, tables)
        obj = portable_object(transport, tables.dump())
        with tempfile.TemporaryDirectory(prefix="feasibility_", dir="/tmp/opencode") as directory:
            root = Path(directory)
            (root / "object.pyz").write_bytes(obj)
            subprocess.run([sys.executable, "-I", "-S", str(root / "object.pyz"),
                            str(root / "q"), str(root / "x")], check=True, capture_output=True)
            self.assertEqual((root / "q").read_bytes(), self.q.astype("<i4").tobytes())
            expected = reconstruct(self.q, self.pos, self.bits, self.epsilon)
            self.assertEqual((root / "x").read_bytes(), expected.astype("<f4").tobytes())

    def test_synthetic_generation_and_fit_calibration(self):
        from finetuning.run_feasibility_synthetic import calibration_sigma, generate
        for kind in ("gaussian_iid", "gaussian_ar1", "conditional_variance"):
            a = generate(kind, 1701, 4096)
            np.testing.assert_array_equal(a, generate(kind, 1701, 4096))
            self.assertFalse(np.array_equal(a, generate(kind, 101701, 4096)))
            self.assertGreater(calibration_sigma(a), 0)
        self.assertIsNone(calibration_sigma(generate("constant", 1701, 13)))
        a = generate("gaussian_ar1", 1701, 16384)
        self.assertAlmostEqual(float(np.corrcoef(a[:-1, 0], a[1:, 0])[0, 1]), .95, delta=.02)

    def test_probe_detects_ar_and_conditional_variance_in_cdf(self):
        from finetuning.run_feasibility_synthetic import calibration_sigma, generate
        for kind in ("gaussian_ar1", "conditional_variance"):
            fit, check = (generate(kind, seed, 8192) for seed in (1701, 101701))
            epsilon = .1 * calibration_sigma(fit)
            fq, fp, _ = quantize(fit, epsilon)
            cq, cp, _ = quantize(check, epsilon)
            marginal = fit_tables(fq, fp, epsilon, range(8), 63, "marginal")
            temporal = fit_tables(fq, fp, epsilon, range(8), 63, "temporal")
            gain = score(marginal, cq, cp)["integer_cdf_nll_bits"] - score(temporal, cq, cp)["integer_cdf_nll_bits"]
            self.assertGreater(gain / cq.size, .05)

    def test_invalid_inputs(self):
        for epsilon in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                quantize(self.x, epsilon)
        for ids in ([0, 0, 1], [2, 1, 0], [0, 1]):
            with self.assertRaises(ValueError):
                contexts(self.q, self.pos, ids, "temporal")

    def test_frozen_fixtures_when_present(self):
        root = Path("results/experiments/feasibility_coder_preflight_20261117")
        if not (root / "manifest.json").exists():
            self.skipTest("Preflight not frozen yet")
        manifest = json.loads((root / "manifest.json").read_text())
        for name, digest in manifest["artifacts"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        fixture = np.load(root / "fixture.npz")
        for mode in ("marginal", "temporal", "spatiotemporal"):
            q, x = decode((root / f"{mode}.fqs").read_bytes(), (root / f"{mode}.ftb").read_bytes())
            np.testing.assert_array_equal(q, fixture["q"])
            np.testing.assert_array_equal(x.view("u4"), fixture["reconstruction_bits"])


if __name__ == "__main__":
    unittest.main()
