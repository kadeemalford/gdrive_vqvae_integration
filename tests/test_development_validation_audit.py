import json
import tempfile
import unittest
from pathlib import Path

from finetuning.audit_development_validation import audit


class DevelopmentAuditTests(unittest.TestCase):
    def test_failed_or_partial_runs_cannot_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for exits in ([], [{"returncode": 0}] * 4, [{"returncode": 1}] * 5):
                (root / "EXIT.json").write_text(json.dumps(exits))
                (root / "COMPLETE").touch()
                with self.assertRaises(ValueError):
                    audit(root)

    def test_missing_supervisor_marker_fails_before_reading_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "EXIT.json").write_text(json.dumps([{"returncode": 0}] * 5))
            with self.assertRaises(ValueError):
                audit(root)
