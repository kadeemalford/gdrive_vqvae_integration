import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from finetuning.scalar_source_lineage import APPROVED_NONRUNTIME_PAIRS, audit_scalar_history


class ScalarHistoryTests(unittest.TestCase):
    def test_recorded_compatibility_exceptions_have_full_sha256_values(self):
        for pair in APPROVED_NONRUNTIME_PAIRS.values():
            for value in pair:
                self.assertEqual(len(value), 64)
                int(value, 16)

    def test_unreviewed_changes_are_rejected_even_with_matching_old_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'history'; source.mkdir()
            (root / 'runtime.py').write_text('unchanged')
            expected = hashlib.sha256(b'unchanged').hexdigest()
            (source / 'protocol.json').write_text(json.dumps({'code_sha256': {'runtime.py': expected}}))
            self.assertEqual(audit_scalar_history(source, root)['runtime.py']['status'], 'identical')
            (source / 'code_snapshot').mkdir()
            (source / 'code_snapshot/runtime.py').write_text('unchanged')
            (root / 'runtime.py').write_text('unexpected change')
            with self.assertRaises(ValueError):
                audit_scalar_history(source, root)
