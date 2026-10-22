import unittest
from finetuning.run_rate_allocation_confirmation import ROOT, verify_source_fold


class ConfirmationSourceTests(unittest.TestCase):
    def test_fold_specific_source_required(self):
        self.assertEqual(verify_source_fold({"epoch": 9, "finetune": {"samples_dir": "data/finetune/fold_0"}},
                                           ROOT / "data/finetune/fold_0"), 9)
        with self.assertRaisesRegex(ValueError, "potential leakage"):
            verify_source_fold({"epoch": 23, "finetune": {"samples_dir": "data/finetune/fold_1"}},
                               ROOT / "data/finetune/fold_0")

    def test_untraced_source_rejected(self):
        with self.assertRaises(ValueError):
            verify_source_fold({"epoch": 9}, ROOT / "data/finetune/fold_0")
