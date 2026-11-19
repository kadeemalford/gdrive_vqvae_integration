import json
from pathlib import Path
import unittest

from tools.audit_feasibility_clock_envelopes import audit


class FeasibilityClockEnvelopeTests(unittest.TestCase):
    def test_frozen_clock_envelope_never_authorizes_signals(self):
        root = Path(__file__).resolve().parents[1] / "results/experiments"
        temporal = root / "feasibility_temporal_preflight_20261117/temporal_preflight.json"
        freeze = root / "feasibility_definition_20261117/metadata_freeze.json"
        if not temporal.is_file() or not freeze.is_file():
            self.skipTest("Frozen experiment metadata is not installed")
        result = audit(json.loads(temporal.read_text()), json.loads(freeze.read_text()))
        self.assertEqual(result["cross_file_potential_intersections"], 534)
        self.assertEqual(result["candidate_role_counts"],
                         {"excluded": 2, "fit": 20, "check": 8, "blocked": 2})
        self.assertFalse(result["contrasts"]["multicore/C2 and C3"]["has_candidate_contrast"])
        self.assertEqual(sum(r["has_candidate_contrast"] for r in result["contrasts"].values()), 8)
        self.assertIsNone(result["executable_plan"])
        self.assertEqual(result["waveform_cells_read"], 0)
