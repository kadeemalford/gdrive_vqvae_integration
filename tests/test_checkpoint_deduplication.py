from pathlib import Path
import os
import tempfile
import unittest

import torch

from finetuning.deduplicate_completed_checkpoints import candidates, link_equivalent, payload_digest


class CheckpointDeduplicationTests(unittest.TestCase):
    def test_equivalent_archives_keep_both_loadable_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "best.pt", Path(directory) / "last.pt"
            payload = {"epoch": 2, "model": torch.arange(12), "optimizer": {"lr": .01}, "rng": (1, 2)}
            torch.save(payload, source)
            torch.save(payload, target)
            self.assertEqual(payload_digest(source), payload_digest(target))
            action = link_equivalent(source, target)
            self.assertTrue(os.path.samefile(source, target))
            self.assertEqual(torch.load(target, weights_only=False)["epoch"], 2)
            self.assertGreater(action["reclaimed_allocated_bytes"], 0)
            self.assertIsNone(link_equivalent(source, target))

    def test_distinct_metadata_refused_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "best.pt", Path(directory) / "last.pt"
            torch.save({"model": torch.arange(12), "epoch": 1}, source)
            torch.save({"model": torch.arange(12), "epoch": 2}, target)
            previous = target.read_bytes()
            with self.assertRaisesRegex(ValueError, "differ"):
                link_equivalent(source, target)
            self.assertEqual(target.read_bytes(), previous)

    def test_active_and_unapproved_experiments_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("sampling_pilot_20261005", "preservation_long_20261007", "unknown"):
                folder = root / name
                folder.mkdir()
                for file in ("best.pt", "last.pt"):
                    torch.save({"model": torch.arange(12)}, folder / file)
                if name != "sampling_pilot_20261005":
                    (folder / "COMPLETE").touch()
            self.assertEqual(candidates(root), [])
