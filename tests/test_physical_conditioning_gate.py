import copy
import unittest

from finetuning.summarize_physical_conditioning import cohort, opportunity


class ConditioningGateTests(unittest.TestCase):
    def fixture(self):
        groups = ["doped/seen", "single_mode/A", "single_mode/B"]

        def result(scores, size):
            return {"records": [{"group_id": g, "drive_id": str(i), "cells": 100, "windows": 1,
                                 "frame_bytes": size, "nrmse_star_macro_channels": score}
                                for i, (g, score) in enumerate(zip(groups, scores))], "setup_bytes": 10}
        return {"constant": result([1., 1., 1.], 100),
                "hardware": result([1., .94, .94], 90),
                "category": result([1., 1.1, 1.1], 101)}

    def test_gate_checks_both_controls_and_seen_groups(self):
        normal = self.fixture()
        self.assertTrue(opportunity(normal, ["single_mode/A", "single_mode/B"], ["doped/seen"])["opportunity_gate"])
        normal["hardware"]["records"][0]["nrmse_star_macro_channels"] = 1.00001
        self.assertFalse(opportunity(normal, ["single_mode/A", "single_mode/B"], ["doped/seen"])["opportunity_gate"])

    def test_subset_pays_entire_setup_and_cannot_drop_a_group(self):
        result = cohort(self.fixture()["hardware"], ["single_mode/A", "single_mode/B"])
        self.assertEqual(result["stream_bytes"], 190)
        with self.assertRaises(ValueError):
            cohort(self.fixture()["hardware"], ["single_mode/absent"])
        normal = self.fixture()
        normal["hardware"]["setup_bytes"] = 200
        self.assertFalse(opportunity(normal, ["single_mode/A", "single_mode/B"], ["doped/seen"])["opportunity_gate"])

    def test_subthreshold_gain_is_not_rounded_up(self):
        normal = copy.deepcopy(self.fixture())
        for row in normal["hardware"]["records"][1:]:
            row["nrmse_star_macro_channels"] = .9500001
        self.assertFalse(opportunity(normal, ["single_mode/A", "single_mode/B"], ["doped/seen"])["opportunity_gate"])
