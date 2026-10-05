import unittest

from finetuning.summarize_sampling_experiment import aggregate


class SummaryTests(unittest.TestCase):
    def test_configurations_not_windows_have_equal_weight(self):
        rows = [dict(checkpoint="window", family="a", configuration="large",
                     correlation=0.0, relative_rmse=1.0, snr_db=0.0) for _ in range(100)]
        rows.append(dict(checkpoint="window", family="a", configuration="small",
                         correlation=1.0, relative_rmse=0.0, snr_db=2.0))
        configurations, families = aggregate(rows)
        self.assertEqual(len(configurations), 2)
        self.assertEqual(families[("window", "a")]["correlation"], 0.5)
        self.assertEqual(families[("window", "a")]["relative_rmse"], 0.5)
        self.assertEqual(families[("window", "a")]["snr_db"], 1.0)


if __name__ == "__main__":
    unittest.main()
