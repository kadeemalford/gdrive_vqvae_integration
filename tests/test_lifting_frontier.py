import unittest

from finetuning.summarize_lifting_frontier import compare


class LiftingFrontierTests(unittest.TestCase):
    def point(self, label, error, size, group=None):
        return {"label": label, "macro_families": error, "stream_bytes": size,
                "groups": {"g": error if group is None else group}, "shared_bytes": 100}

    def test_never_buys_fidelity_with_unreported_bytes(self):
        reference = self.point("fixed", 1., 100)
        candidate = self.point("learned", .5, 101)
        self.assertIsNone(compare(reference, [candidate])["candidate"])

    def test_no_promotion_and_configuration_guard(self):
        reference = self.point("fixed", 1., 100)
        candidate = self.point("learned", .8, 90, group=1.1)
        result = compare(reference, [candidate])
        self.assertFalse(result["exploratory_five_percent_error_gate"])
        self.assertFalse(result["final_acceptance"])
        self.assertEqual(result["regressing_groups"], ["g"])

    def test_equivalent_error_savings_charged_separately(self):
        reference = self.point("fixed", 1., 100)
        lower_error = self.point("lowest_error", .7, 95)
        cheaper = self.point("cheaper", .9, 80)
        result = compare(reference, [lower_error, cheaper])
        self.assertEqual(result["candidate"], "lowest_error")
        self.assertEqual(result["equivalent_error_candidate"], "cheaper")
        self.assertTrue(result["exploratory_ten_percent_byte_gate"])
        self.assertFalse(result["final_acceptance"])
