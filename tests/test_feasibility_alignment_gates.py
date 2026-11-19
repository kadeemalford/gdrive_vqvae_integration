from copy import deepcopy
import unittest

from finetuning.summarize_feasibility_alignment import evaluate_endpoint_comparisons


def arm(error, sizes):
    return {
        "cells": 1000, "macro_families": error, "families": {"doped": error},
        "rate": {"shared_model": {"bytes": sizes[0]}, "self_contained": {"bytes": sizes[1]}},
        "records": [{"drive_id": "f", "group_id": "doped/g", "cells": 1000,
                     "windows": 1, "source_channel_indices": [7]}],
        "coverage_guards": {"doped/g": {"sse": 10 * error, "p90_channel_nrmse_star": error,
            "spectral": {"0_1khz": {"target_energy": 1., "supported_target_energy": 1.,
                                    "relative_complex_rmse": error}}}},
    }


class FeasibilityAlignmentGateTests(unittest.TestCase):
    def test_paired_control_and_mature_gates_are_separate(self):
        result = evaluate_endpoint_comparisons(arm(.8, (20, 1000)), arm(1., (20, 800)),
                                              arm(.7, (20, 800)))
        self.assertTrue(result["comparisons"]["compatibility"]["passed"])
        self.assertFalse(result["comparisons"]["compatibility"]["object_bytes_not_increased_descriptive"])
        self.assertFalse(result["comparisons"]["practical_advance"]["passed"])
        self.assertFalse(result["p3_eligible"])

    def test_population_cannot_change_with_same_cell_count(self):
        reference = arm(1., (20, 800))
        candidate = deepcopy(reference)
        candidate["records"][0]["source_channel_indices"] = [8]
        with self.assertRaises(ValueError):
            evaluate_endpoint_comparisons(candidate, reference, reference)
