import unittest

from finetuning.summarize_feasibility_transfer_robustness import summarize


def ledger():
    curves = []
    for ratio in (.03, .1):
        for index in range(8):
            for mode, total in (("marginal", 1000), ("temporal", 990),
                                ("spatiotemporal", 900 if index else 1010)):
                curves.append({"ratio": ratio, "horizon": 20, "group_id": str(index),
                               "mode": mode, "cells": 1000, "files": 1, "windows": 1,
                               "available_windows_by_file": {"f": 1}, "total_bytes": total})
    return {"kind": "feasibility-npz-transfer-results-v1", "fit_calls": 0, "curves": curves}


class TransferRobustnessTests(unittest.TestCase):
    def test_exclusion_uses_paired_byte_totals(self):
        spatial = summarize(ledger())["contrasts"][1]
        self.assertEqual(spatial["configurations_with_saving"], 7)
        self.assertAlmostEqual(spatial["aggregate_saving"], 690 / 8000)
        self.assertAlmostEqual(spatial["saving_without_each_configuration"]["0"], .1)
        self.assertAlmostEqual(spatial["saving_without_each_configuration"]["1"], 590 / 7000)

    def test_changed_coverage_fails(self):
        data = ledger()
        data["curves"][1]["cells"] = 999
        with self.assertRaises(ValueError):
            summarize(data)
