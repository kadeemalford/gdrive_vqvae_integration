import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from finetuning.summarize_numeric_adaptation import compare, family_metrics


class NumericSummaryTests(unittest.TestCase):
    def test_continuation_baseline_is_source_epoch_not_random_initialization(self):
        history = {epoch: {"macro_relative_rmse": value, "macro_correlation": .5,
                           "worst_family_relative_rmse": value} for epoch, value in ((0, 1.), (16, .8), (32, .7))}
        groups = [dict(epoch=epoch, family="family", configuration="one", relative_rmse=value, correlation=.5)
                  for epoch, value in ((0, 1.), (16, .8), (32, .7))]
        summary = dict(validation_hash="same", best_macro_epoch=32, epochs_completed=32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"arms": ["arm"], "initial_epoch": 16}))
            with patch("finetuning.summarize_numeric_adaptation.read_history", return_value=(history, groups, summary)):
                state = compare(root)["arms"]["arm"]
        self.assertEqual(state["initial"]["epoch"], 16)
        self.assertAlmostEqual(state["selected_macro_rmse_delta"], -.1)

    def test_configuration_regression_not_hidden_by_family_mean(self):
        history = {epoch: {"macro_relative_rmse": value, "macro_correlation": .5,
                           "worst_family_relative_rmse": value} for epoch, value in ((0, .5), (1, .45))}
        groups = [dict(epoch=epoch, family="single_mode", configuration=config,
                       relative_rmse=value, correlation=.5)
                  for epoch, config, value in ((0, "a", .5), (0, "b", .5), (1, "a", .3), (1, "b", .6))]
        summary = dict(validation_hash="same", best_macro_epoch=1, epochs_completed=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(json.dumps({"arms": ["arm"]}))
            with patch("finetuning.summarize_numeric_adaptation.read_history", return_value=(history, groups, summary)):
                state = compare(root)["arms"]["arm"]
        self.assertEqual(state["selected_family_rmse_regressions"], [])
        self.assertEqual(state["selected_configuration_rmse_regressions"][0]["configuration"], "b")

    def test_equal_configuration_not_window_weight(self):
        rows = [dict(epoch="0", family="a", configuration="one", relative_rmse="0.2", correlation="0.8", windows="100"),
                dict(epoch="0", family="a", configuration="two", relative_rmse="0.8", correlation="0.2", windows="1")]
        self.assertEqual(family_metrics(rows, 0)["a"]["relative_rmse"], .5)

    def test_duplicate_group_rejected(self):
        row = dict(epoch="0", family="a", configuration="one", relative_rmse="0.2", correlation="0.8")
        with self.assertRaisesRegex(ValueError, "Repeated"):
            family_metrics([row, row], 0)

    def test_missing_epoch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            family_metrics([], 1)
