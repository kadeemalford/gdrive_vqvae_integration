import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from finetuning.run_sampling_experiment import run


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.log = Path(self.directory.name) / "training.log"

    def test_live_output_and_log_preserve_carriage_returns(self):
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            run(["-c", "import sys; sys.stdout.write('10%\\r100% ✓\\n'); sys.stderr.write('stderr\\n')"], self.log)
        self.assertIn("10%\r100% ✓", console.getvalue())
        self.assertIn("10%\r100% ✓".encode(), self.log.read_bytes())
        self.assertIn("stderr", console.getvalue())

    def test_resume_appends_and_quiet_keeps_detail_in_log(self):
        self.log.write_text("previous epoch\n")
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            run(["-c", "print('new epoch')"], self.log, resume=True, quiet=True)
        self.assertNotIn("new epoch", console.getvalue())
        self.assertIn("previous epoch", self.log.read_text())
        self.assertIn("new epoch", self.log.read_text())

    def test_child_failure_is_not_hidden(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                run(["-c", "raise SystemExit(7)"], self.log)
        self.assertEqual(error.exception.returncode, 7)

    def test_dry_run_does_not_create_log(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run(["-c", "raise SystemExit(7)"], self.log, dry_run=True)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
