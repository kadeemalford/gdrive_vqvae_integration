import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.consolidate_progress_reports import POST_89_END, POST_89_START, consolidate, historical_part


class ReportConsolidationTests(unittest.TestCase):
    def fixture(self, root):
        directory = root / "progress_reports"
        directory.mkdir()
        (directory / "89-UNCHANGED.md").write_text("# Histórico89\n\nNão alterar.\n")
        for number in range(90, 122):
            (directory / f"{number}-SOURCE.md").write_text(
                f"# Título {number}\n\n## Resultado\n\nEvidência única {number}.\n")
        (directory / "FEBUS_GENERALIST_ROADMAP.md").write_text("[Resultado](121-SOURCE.md)\n")
        (root / "README.md").write_text("Antes.\n" + POST_89_START + " antigo.\n" + POST_89_END + "\nDepois.\n")

    def test_full_merge_preserves_originals_and_earlier_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            expected = (root / "progress_reports/93-SOURCE.md").read_bytes()
            plan = consolidate(root, apply=True)
            self.assertEqual((plan["before"], plan["after"]), (32, 13))
            files = [path for path in (root / "progress_reports").glob("*.md")
                     if path.name.split("-")[0].isdigit() and int(path.name.split("-")[0]) > 89]
            self.assertEqual(len(files), 13)
            self.assertEqual((root / "progress_reports/89-UNCHANGED.md").read_text(), "# Histórico89\n\nNão alterar.\n")
            merged = (root / "progress_reports/92-LEARNED_LIFTING_STUDY.md").read_text()
            self.assertIn("Evidência única 93.", merged)
            self.assertIn('id="report-93"', merged)
            archive = root / "progress_reports/archive/post_89_consolidation_20261115/originals.tar.gz"
            with tarfile.open(archive) as backup:
                self.assertEqual(backup.extractfile("progress_reports/93-SOURCE.md").read(), expected)
            mapping = json.loads((root / "progress_reports/consolidation_map.json").read_text())
            self.assertEqual(mapping["93-SOURCE.md"]["anchor"], "report-93")

    def test_dry_run_and_preexisting_target_do_not_destroy_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            consolidate(root)
            self.assertTrue((root / "progress_reports/93-SOURCE.md").exists())
            target = root / "progress_reports/92-LEARNED_LIFTING_STUDY.md"
            target.write_text("User work")
            with self.assertRaises(FileExistsError):
                consolidate(root, apply=True)
            self.assertEqual(target.read_text(), "User work")
            self.assertTrue((root / "progress_reports/93-SOURCE.md").exists())

    def test_headings_and_internal_links_are_namespaced_but_code_is_not_demoted(self):
        text = "# Teste\n\n## Seção\n\n[Ver](#seção)\n\n```python\n# comment\n```\n"
        merged = historical_part(93, "93-SOURCE.md", text, {})
        self.assertIn('id="report-93-seção"', merged)
        self.assertIn('<a id="report-93"></a>\n\n## Registro', merged)
        self.assertIn('<a id="report-93-seção"></a>\n\n####', merged)
        self.assertIn("[Ver](#report-93-seção)", merged)
        self.assertIn("#### Seção", merged)
        self.assertIn("```python\n# comment\n```", merged)
