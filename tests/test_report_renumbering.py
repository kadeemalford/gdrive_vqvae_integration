import json
import tempfile
import unittest
from pathlib import Path

from tools.renumber_progress_reports import renumber


class ReportRenumberingTests(unittest.TestCase):
    def test_sequential_names_links_historical_anchors_and_noop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "progress_reports"
            directory.mkdir()
            (root / "README.md").write_text("[H2](progress_reports/96-H2.md#report-98)\n")
            (directory / "89-OLD.md").write_text("# Original89\n")
            (directory / "90-REVIEW.md").write_text("# Review\n")
            (directory / "92-H1.md").write_text('# H1\n<a id="report-95"></a>\n')
            (directory / "96-H2.md").write_text('# H2\n<a id="report-98"></a>\n')
            (directory / "README.md").write_text("| [96 — H2](96-H2.md) | 96, 97, 98 |\n")
            (directory / "consolidation_map.json").write_text(json.dumps({"98-RESULT.md": {"target": "96-H2.md", "anchor": "report-98"}}))
            renumber(root)
            self.assertTrue((directory / "96-H2.md").exists())
            result = renumber(root, apply=True)
            self.assertEqual(result["mapping"]["96-H2.md"], "92-H2.md")
            self.assertIn("#report-98", (root / "README.md").read_text())
            self.assertIn("92-H2.md", (root / "README.md").read_text())
            self.assertIn('id="report-98"', (directory / "92-H2.md").read_text())
            self.assertIn("[92 — H2]", (directory / "README.md").read_text())
            self.assertEqual((directory / "89-OLD.md").read_text(), "# Original89\n")
            self.assertEqual(renumber(root, apply=True)["changed"], 0)
