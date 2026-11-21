import unittest

import numpy as np

from finetuning.run_external_egs_routing import STARTS, SUBSETS, calibration, paired, rows


class ExternalRoutingTests(unittest.TestCase):
    def test_check_cannot_calibrate(self):
        w = {"raw": np.array([[0, 5], [2, 5]], dtype=np.float32), "role": "fit"}
        self.assertEqual(calibration([w]), 1)
        with self.assertRaises(ValueError):
            calibration([w, {**w, "role": "check"}])

    def test_equal_exposure_by_clip_and_no_run_overlap(self):
        records = rows()
        self.assertEqual(len({r["stamp"][:8] for r in records}), 4)
        fit = [r["session"] for r in records if r["role"] == "fit"]
        clips = [(s, t) for s in fit for t in STARTS]
        for name, indices in SUBSETS.items():
            chosen = [clips[i] for i in indices]
            self.assertEqual(len({t for _, t in chosen}), 2)
            self.assertEqual(len({s for s, _ in chosen}), 1 if name.endswith("only") else 2)
        self.assertEqual(sorted(SUBSETS["A0B1"] + SUBSETS["B0A1"]), list(range(4)))

    def test_equal_sizes_do_not_mask_different_realizations(self):
        rec = dict(sample_path="C/t0", q_sha256="q", original_float32_sha256="raw",
                   waveform_sha256="x")
        with self.assertRaises(ValueError):
            paired({"records": [rec]}, {"records": [{**rec, "waveform_sha256": "y"}]})
