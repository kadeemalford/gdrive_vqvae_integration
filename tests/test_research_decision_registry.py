import json
import tempfile
import unittest
from pathlib import Path

from finetuning.consolidate_research_decisions import read_history, validate_cohort


class DecisionRegistryTests(unittest.TestCase):
    def test_completed_history_does_not_accept_resets_or_partial_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text("\n".join(json.dumps({"update": n}) for n in (32, 64)))
            self.assertEqual(read_history(path, 64)["update"], 64)
            with self.assertRaises(ValueError):
                read_history(path, 8192)
            path.write_text("\n".join(json.dumps({"update": n}) for n in (32, 32, 64)))
            with self.assertRaises(ValueError):
                read_history(path, 64)

    def test_cohorts_are_not_merged_by_equal_cell_count(self):
        point = {"cells": 100, "groups": {"a": .5}, "file_population": [("a", "file1", 100)]}
        validate_cohort([point, point])
        with self.assertRaises(ValueError):
            validate_cohort([point, {**point, "file_population": [("a", "file2", 100)]}])
