"""Tests for metadata-only deterministic feasibility selection."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.freeze_feasibility_metadata import (
    canonical,
    historical_hashes,
    select_f1_panel,
    select_f2_pool,
    selection_key,
    sha256_file,
    write_freeze,
)


def row(file_id, sample="0", split="train", group="group"):
    return {"drive_id": file_id, "sample_path": f"{file_id}/{sample}.npz",
            "split": split, "group_id": group, "block_index": sample}


def provenance(ids):
    return {"files": [{"drive_id": file_id, "group_id": "group",
                       "clock_status": "read", "clock_dtype": "float64",
                       "clock_native": [index]}
                      for index, file_id in enumerate(ids)],
            "complete_block_confirmations": [], "exact_clock_array_groups": []}


class FeasibilityMetadataTests(unittest.TestCase):
    def test_panel_uses_only_historical_rows_and_is_order_invariant(self):
        rows = [row(file_id, str(index)) for file_id in "abcd" for index in range(4)]
        checks = [{**item, "role": "check"} for item in rows if item["block_index"] != "0"]
        expected_ids = sorted("abcd", key=selection_key)[:2]
        expected = [min([item for item in rows if item["drive_id"] == file_id
                         and item["block_index"] != "0"],
                        key=lambda item: selection_key(item["sample_path"]))
                    for file_id in expected_ids]
        self.assertEqual(select_f1_panel(rows, checks), expected)
        self.assertEqual(select_f1_panel(rows[::-1], checks[::-1]), expected)
        self.assertTrue(all("role" not in item for item in expected))
        self.assertEqual(selection_key("é"), hashlib.sha256(
            "feasibility-v1|é".encode("utf-8")).hexdigest())

    def test_transitive_exclusion_uses_all_checks_and_all_relation_types(self):
        rows = [row(file_id) for file_id in "abcdef"]
        data = provenance("abcdef")
        data["files"][1]["clock_native"] = [0]
        data["complete_block_confirmations"] = [
            {"left": ["b", 0], "right": ["c", 2], "complete_block_values_equal": True},
            {"left": ["d", 0], "right": ["e", 0], "complete_block_values_equal": False}]
        data["exact_clock_array_groups"] = [
            {"files": [{"drive_id": "c"}, {"drive_id": "d"}],
             "clock_dtype": "float64", "clock_native": [100]}]
        result = select_f2_pool(rows, [rows[0], rows[5]], data)
        self.assertEqual([item["drive_id"] for item in result["files"]], ["e"])
        self.assertEqual({item["drive_id"] for item in result["exclusion_ledger"]},
                         set("abcdf"))
        component = next(item for item in result["components"] if "a" in item["members"])
        self.assertEqual(component["members"], list("abcd"))
        self.assertEqual(component["component_id"], hashlib.sha256(
            canonical(list("abcd")).encode("utf-8")).hexdigest())
        reversed_data = {**data, "files": data["files"][::-1]}
        self.assertEqual(result, select_f2_pool(rows[::-1], [rows[5], rows[0]], reversed_data))
        self.assertIsNone(result["fit_check_assignment"])
        self.assertFalse(result["independence_claim"])

    def test_clock_dtype_is_part_of_exact_timestamp_key(self):
        rows = [row("a"), row("b")]
        data = provenance("ab")
        data["files"][1].update(clock_dtype="float32", clock_native=[0])
        result = select_f2_pool(rows, [rows[0]], data)
        self.assertEqual([item["drive_id"] for item in result["files"]], ["b"])

    def test_cross_split_ids_are_forbidden(self):
        rows = [row("a"), row("a", "other", split="val"), row("b")]
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            select_f1_panel(rows, [rows[0], rows[2]])
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            select_f2_pool(rows, [rows[0]], provenance("ab"))

    def test_historical_common_fields_must_match_exactly(self):
        rows = [row("a"), row("b")]
        with self.assertRaisesRegex(ValueError, "mismatch"):
            select_f1_panel(rows, [{**rows[0], "block_index": "1"}, rows[1]])
        with self.assertRaisesRegex(ValueError, "missing"):
            select_f1_panel(rows, [row("unknown")])

    def test_provenance_must_cover_train_and_relations_must_reference_known_ids(self):
        rows = [row("a"), row("b")]
        with self.assertRaisesRegex(ValueError, "cover"):
            select_f2_pool(rows, [rows[0]], provenance("a"))
        data = provenance("ab")
        data["exact_clock_array_groups"] = [{
            "files": [{"drive_id": "a"}, {"drive_id": "unknown"}],
            "clock_dtype": "float64", "clock_native": [0]}]
        with self.assertRaisesRegex(ValueError, "absent"):
            select_f2_pool(rows, [rows[0]], data)
        with self.assertRaisesRegex(ValueError, "not train"):
            select_f2_pool([row("a", split="val")], [row("a", split="val")],
                           provenance("a"))

    def test_output_is_idempotent_but_never_overwrites_different_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metadata_freeze.json"
            write_freeze(output, {"value": 1})
            original = output.read_bytes()
            write_freeze(output, {"value": 1})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_freeze(output, {"value": 2})
            self.assertEqual(output.read_bytes(), original)

    def test_source_hashing_refuses_waveforms(self):
        with self.assertRaisesRegex(ValueError, "Disallowed"):
            sha256_file(Path("must_not_be_opened.npz"))

    def test_historical_sequence_must_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "progress_reports").mkdir()
            with self.assertRaisesRegex(ValueError, "complete"):
                historical_hashes(root)
            for index in range(1, 107):
                (root / "progress_reports" / f"{index}-REPORT.md").write_text(
                    f"Report {index}", encoding="utf-8")
            hashes = historical_hashes(root)
            self.assertEqual(list(hashes), [str(index) for index in range(1, 107)])
            (root / "progress_reports/01-DUPLICATE.md").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unambiguous"):
                historical_hashes(root)


if __name__ == "__main__":
    unittest.main()
