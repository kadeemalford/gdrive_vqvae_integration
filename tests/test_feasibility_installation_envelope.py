"""Integrity, fallback and report107 tests without signals or third-party libs."""
import hashlib
import io
import json
import lzma
from pathlib import Path
import random
import unittest
import zipfile

from compression.feasibility_installation_envelope import HEADER, OVERHEAD, pack, unpack
from finetuning.audit_feasibility_installation_envelope import guards


def sealed(magic, flag, original_size, payload, original_digest):
    body = HEADER.pack(magic, flag, original_size, len(payload), original_digest) + payload
    return body + hashlib.sha256(body).digest()


class InstallationEnvelopeTests(unittest.TestCase):
    def test_xz_exact_preset_and_full_zip_bytes(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("table.bin", b"abcd" * 10000)
            archive.comment = b"original zip comment must survive"
        original = stream.getvalue()
        encoded = pack(original)
        self.assertEqual(HEADER.unpack_from(encoded)[1], 1)
        self.assertEqual(encoded[HEADER.size:-32],
                         lzma.compress(original, format=lzma.FORMAT_XZ,
                                       check=lzma.CHECK_CRC64, preset=6))
        self.assertEqual(unpack(encoded), original)
        self.assertEqual(len(encoded), len(encoded[HEADER.size:-32]) + OVERHEAD)

    def test_raw_fallback_including_empty(self):
        for original in (b"", b"a", random.Random(17).randbytes(4096)):
            encoded = pack(original)
            self.assertEqual(HEADER.unpack_from(encoded)[1], 0)
            self.assertEqual(unpack(encoded), original)
            self.assertEqual(len(encoded), len(original) + OVERHEAD)

    def test_every_byte_is_integrity_protected(self):
        encoded = pack(b"hello" * 100)
        for i in range(len(encoded)):
            damaged = bytearray(encoded)
            damaged[i] ^= 1
            with self.assertRaises(ValueError):
                unpack(bytes(damaged))

    def test_truncation_and_extra_bytes_rejected(self):
        encoded = pack(b"z" * 1000)
        for end in range(len(encoded)):
            with self.assertRaises(ValueError):
                unpack(encoded[:end])
        with self.assertRaises(ValueError):
            unpack(encoded + b"x")

    def test_authenticated_invalid_grammar_rejected(self):
        digest = hashlib.sha256(b"a").digest()
        invalid = (
            sealed(b"BAD1", 0, 1, b"a", digest),
            sealed(b"FIE1", 2, 1, b"a", digest),
            sealed(b"FIE1", 0, 2, b"a", digest),
            sealed(b"FIE1", 0, 1, b"b", digest),
            sealed(b"FIE1", 1, 1, b"not xz", digest),
            sealed(b"FIE1", 1, 1, lzma.compress(b"a")[:-1], digest),
            sealed(b"FIE1", 1, 1, lzma.compress(b"a") + b"x", digest),
            sealed(b"FIE1", 1, 1, lzma.compress(b"a") * 2, digest),
            sealed(b"FIE1", 1, 1, lzma.compress(b"a" * 10000), digest),
        )
        for encoded in invalid:
            with self.assertRaises(ValueError):
                unpack(encoded)

    def test_payload_length_and_original_digest(self):
        original = b"abc" * 1000
        encoded = pack(original)
        body = bytearray(encoded[:-32])
        body[13] ^= 1  # authenticated malformed payload length
        with self.assertRaises(ValueError):
            unpack(bytes(body) + hashlib.sha256(body).digest())
        bad = sealed(b"FIE1", 1, len(original), lzma.compress(original), b"\0" * 32)
        with self.assertRaises(ValueError):
            unpack(bad)

    def test_guards_are_joint_and_keep_generalist_blocked(self):
        rows, summaries = [], []
        for ratio in (.03, .1):
            for mode in ("marginal", "temporal", "spatiotemporal"):
                summaries.append({"ratio": ratio, "mode": mode, "cells": 9000,
                                  "shared_once_total_bytes": 9000 if mode == "marginal" else 7200,
                                  "isolated_total_bytes": 9000 if mode == "marginal" else 7200})
                for i in range(9):
                    rows.append({"ratio": ratio, "mode": mode, "family": str(i % 4),
                                 "group_id": str(i), "new_marginal_total_bytes": 1000,
                                 "saving_vs_new_marginal_bytes": 0 if mode == "marginal" else 200})
        outcomes = guards(rows, summaries)
        self.assertEqual(len(outcomes), 8)
        self.assertTrue(all(g["measured_coverage_thresholds_pass"] for g in outcomes))
        self.assertTrue(all(g["generalist_gate"] is None and not g["f4_authorized"] for g in outcomes))
        rows[9]["saving_vs_new_marginal_bytes"] = -11
        self.assertFalse(guards(rows, summaries)[0]["measured_coverage_thresholds_pass"])

    def test_completed_artifacts_when_available(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "results/experiments/feasibility_installation_envelope_20261118"
        source = root / "results/experiments/feasibility_continuous_probe_20261118"
        if not (output / "results.json").exists():
            self.skipTest("Contrast has not executed")
        result = json.loads((output / "results.json").read_text())
        self.assertEqual(result["installation_roundtrips"], 243)
        self.assertEqual(result["zip_artifacts_verified_before_and_after"], 729)
        self.assertEqual(result["waveform_cells_read"], 0)
        self.assertFalse(result["f4_authorized"])
        self.assertEqual((output / "installation_decoder.py").read_bytes(),
                         (root / "compression/feasibility_installation_envelope.py").read_bytes())
        for row in result["rows"]:
            with self.subTest(path=row["envelope_path"]):
                original = (source / row["original_installation_path"]).read_bytes()
                envelope = (output / row["envelope_path"]).read_bytes()
                self.assertEqual(unpack(envelope), original)
                self.assertEqual(hashlib.sha256(original).hexdigest(), row["original_installation_sha256"])
                self.assertEqual(hashlib.sha256(envelope).hexdigest(), row["envelope_sha256"])
                self.assertEqual(row["new_isolated_total_bytes"],
                                 row["transport_bytes"] + len(envelope)
                                 + result["decoder_source"]["bytes"])
        for summary in result["summaries"]:
            self.assertEqual(summary["shared_once_total_bytes"],
                             summary["transport_bytes"] + summary["envelope_bytes"]
                             + summary["shared_decoder_bytes"])
            self.assertEqual(summary["isolated_total_bytes"],
                             summary["transport_bytes"] + summary["envelope_bytes"]
                             + summary["isolated_decoder_bytes"])
        self.assertEqual(result["primary_guards"], guards(result["rows"], result["summaries"]))


if __name__ == "__main__":
    unittest.main()
