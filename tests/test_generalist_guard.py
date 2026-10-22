import unittest

from finetuning.generalist_guard import assess


class GeneralistGuardTests(unittest.TestCase):
    def row(self, **updates):
        return {"family": "a", "configuration": "c", "windows": 10,
                "relative_rmse": 1., "correlation": .5, "ordinary_channel_p90_relative_rmse": 1., **updates}

    def test_baseline_is_eligible_and_mean_gain_cannot_hide_tail(self):
        baseline = [self.row()]
        self.assertTrue(assess(baseline, baseline)["eligible"])
        self.assertFalse(assess(baseline, [self.row(relative_rmse=.8, ordinary_channel_p90_relative_rmse=1.1)])["eligible"])
        self.assertFalse(assess(baseline, [self.row(correlation=.4)])["eligible"])

    def test_missing_duplicate_nonfinite_or_different_windows_refused(self):
        baseline = [self.row()]
        for candidate in ([], baseline * 2, [self.row(windows=11)], [self.row(relative_rmse=float("nan"))]):
            with self.assertRaises(ValueError):
                assess(baseline, candidate)

    def test_macro_equal_families_not_equal_configurations(self):
        baseline = [self.row(), self.row(family="b", configuration="d", relative_rmse=2.),
                    self.row(family="b", configuration="e", relative_rmse=4.)]
        self.assertEqual(assess(baseline, baseline)["macro_relative_rmse"], 2.)
