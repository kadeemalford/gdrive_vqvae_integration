import tempfile
import unittest
from pathlib import Path

from finetuning.run_scratch_budget_continuation import FILES, fork_history
from finetuning.run_temporal_density_long import digest


class ScratchForkTests(unittest.TestCase):
    def test_copies_are_verified_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir()
            for name in FILES:
                (source / name).write_text(name)
            hashes = {name: digest(source / name) for name in FILES}
            fork_history(source, target, hashes)
            fork_history(source, target, hashes)
            for name in FILES:
                self.assertEqual(digest(source / name), hashes[name])
                self.assertEqual(digest(target / name), hashes[name])
            self.assertFalse((target / "summary.json").exists())
            (target / "last.pt").write_text("training changed this checkpoint")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                fork_history(source, target, hashes)
