import json
import tempfile
import unittest
from pathlib import Path

from tools.run_semisynthetic_pair import verify_pair


class SemisyntheticPairTests(unittest.TestCase):
    def fixture(self, root):
        for method in ("real", "mixed"):
            folder = root / method
            (folder / "check_000000").mkdir(parents=True)
            (folder / "check_000000/window.frame").write_bytes(b"paid-frame")
            (folder / "results.json").write_text(json.dumps({
                "updates": 2, "cells": 100, "schedule_sha256": "paired",
                "initial_nrmse_star": .4, "initial_bpc": .7,
                "augmented_updates": int(method == "mixed")}))

    def test_pair_accepts_matched_dose_and_rejects_schedule_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            verify_pair(root)
            path = root / "mixed/results.json"
            result = json.loads(path.read_text())
            result["schedule_sha256"] = "different"
            path.write_text(json.dumps(result))
            with self.assertRaises(AssertionError):
                verify_pair(root)

    def test_initial_frame_difference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "mixed/check_000000/window.frame").write_bytes(b"different")
            with self.assertRaises(AssertionError):
                verify_pair(root)
