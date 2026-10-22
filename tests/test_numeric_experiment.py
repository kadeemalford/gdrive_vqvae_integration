import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from finetuning import run_numeric_adaptation_experiment as runner


class NumericExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        checkpoint = self.source / "fold_2/window/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        self.samples = self.root / "data/finetune/fold_2"
        self.samples.mkdir(parents=True)
        with (self.samples / "samples_manifest.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["drive_id", "split"])
            writer.writeheader()
            for key in ["1xlM62wnebrkHFOUcWspz0lzWDDzGCXJj", "1k74OEu0i80YMqpO9U8_JTJ7lJH8V9jZA"]:
                writer.writerow({"drive_id": key, "split": "train"})
        self.output = self.root / "output"

    def invoke(self, extra):
        argv = ["numeric", "--source-experiment", str(self.source), "--output-dir", str(self.output), *extra]
        with patch.object(sys, "argv", argv), patch.object(runner, "ROOT", self.root), patch.object(runner, "check_splits"), contextlib.redirect_stdout(io.StringIO()):
            runner.main()

    def test_dry_run_matched_commands_without_outputs(self):
        with patch.object(runner, "run") as run:
            self.invoke(["--dry-run"])
        self.assertFalse(self.output.exists())
        self.assertEqual(run.call_count, 2)
        first, second = [call.args[0] for call in run.call_args_list]
        self.assertEqual(first[first.index("--normalization") + 1], "legacy")
        self.assertEqual(second[second.index("--normalization") + 1], "escape-v1")
        for flag in ["--checkpoint", "--seed", "--steps-per-epoch", "--learning-rate", "--samples-dir"]:
            self.assertEqual(first[first.index(flag) + 1], second[second.index(flag) + 1])

    def test_refuses_existing_output_without_resume(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.invoke(["--dry-run"])

    def test_refuses_affected_files_in_validation(self):
        path = self.samples / "samples_manifest.csv"
        path.write_text(path.read_text().replace(",train", ",val"))
        with self.assertRaisesRegex(ValueError, "exclusively in training"):
            self.invoke(["--dry-run"])
