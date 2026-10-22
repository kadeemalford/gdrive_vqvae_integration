from pathlib import Path
import unittest
from finetuning.evaluate_physical_candidates import parse_candidate, score_reference


class PhysicalCandidateParserTests(unittest.TestCase):
    def test_reference_guard_prevents_mixed_denominators(self):
        self.assertEqual(parse_candidate("robust=robust-escape-v1=model.pt")[1], "robust-escape-v1")
        self.assertEqual(score_reference(["robust-escape-v1"]), "robust-escape-v1")
        self.assertEqual(score_reference(["channel-escape-v1", "escape-v1"]), "escape-v1")
        with self.assertRaisesRegex(ValueError, "incomparable"):
            score_reference(["escape-v1", "robust-escape-v1"])
    def test_optional_checkpoint_candidate(self):
        label, policy, path = parse_candidate("dense=escape-v1=models/checkpoint.pt")
        self.assertEqual(label, "dense")
        self.assertEqual(policy, "escape-v1")
        self.assertEqual(path, Path("models/checkpoint.pt").resolve())

    def test_bad_labels_and_representation_refused(self):
        for text in ("../dense=escape-v1=x.pt", "dense=legacy=x.pt", "dense", "dense=escape-v1="):
            with self.assertRaises(ValueError):
                parse_candidate(text)
