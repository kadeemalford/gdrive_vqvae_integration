"""Synthetic metadata tests: no source HDF5 or prepared signal arrays."""

import copy
import tempfile
import unittest
from pathlib import Path

from tools.prepare_feasibility_temporal import (
    ROOT, assign_components, augment_components, build_preflight, conditional_intervals,
    contiguous_runs, interval_edges, plan_file, selection_key, write_preflight,
)


def fixture():
    header = {
        "dataset": "host/Source1/Zone1/Strain Rate [nStrain|s]",
        "shape": [2, 21000, 8], "dtype": "float32",
        "attributes": {"/host/Source1/Zone1": {
            "Spacing": [.2, .05, 1], "Origin": [0, -525, 0],
            "Extent": [100, 107, 0, 20999, 0, 0],
            "BlockOverlap": [5], "BlockRate": [1000]}},
    }
    clock = {"dataset": header["dataset"], "shape": header["shape"],
             "clock_status": "read", "clock_native": [1000., 1001.],
             "clock_path": "/host/Source1/time", "clock_dtype": "float64"}
    rows, maps = [], {}
    for block in range(2):
        for region, channels in enumerate(([0, 1], [4, 5, 6])):
            mid = f"b{block}r{region}"
            rows.append({"drive_id": "file", "split": "train", "group_id": "group",
                         "dataset": header["dataset"], "block_index": str(block),
                         "channel_map_id": mid, "channels": str(len(channels)),
                         "spatial_rows_contiguous": "True", "dt_s": "5e-05",
                         "drive_path": "local source"})
            maps[mid] = {
                "drive_id": "file", "block_index": block, "dataset": header["dataset"],
                "source_channel_indices": list(channels), "source_channel_count": 8,
                "selected_channel_count": len(channels), "selected_rows_contiguous": True,
                "metadata_positions_m": [(100 + c) * .2 for c in channels],
                "dataset_relative_positions_m": [c * .2 for c in channels],
                "source_extent_channel_indices": [100 + c for c in channels]}
    return header, clock, rows, maps


class TemporalTests(unittest.TestCase):
    def test_regions_preserve_physical_gaps_and_source_coordinates(self):
        h, c, rows, maps = fixture()
        timing = conditional_intervals(h, c, .00005)
        plan = plan_file("file", rows, maps, h, timing)
        self.assertEqual(plan["cells"], 25000)
        self.assertEqual([r["source_channel_indices"] for r in plan["regions"]], [[0, 1], [4, 5, 6]])
        self.assertEqual(plan["source_block_reads"], 1)
        self.assertEqual(plan["regions"][1]["source_extent_channel_indices"], [104, 105, 106])
        self.assertEqual(plan["regions"][0]["time_stop_source_index_exclusive"], 5000)
        self.assertEqual(plan, plan_file("file", list(reversed(rows)), maps, h, timing))

    def test_index_gaps_duplicates_and_booleans(self):
        self.assertEqual(contiguous_runs([0, 1, 3]), [[0, 1], [3]])
        for bad in ([1, 0], [1, 1], [True], [-1]):
            with self.assertRaises(ValueError):
                contiguous_runs(bad)

    def test_overlap_not_just_equal_timestamps(self):
        h, c, _, _ = fixture()
        a = {"drive_id": "a", **conditional_intervals(h, c, .00005)}
        shifted = copy.deepcopy(c)
        shifted["clock_native"] = [1001.02, 1002.02]
        b = {"drive_id": "b", **conditional_intervals(h, shifted, .00005)}
        edges = interval_edges([a, b])
        self.assertTrue(any(e["left"][1] == 0 and e["right"][1] == 0 for e in edges))
        self.assertAlmostEqual(a["intervals"][0]["start"], 999.475)
        self.assertAlmostEqual(a["intervals"][0]["stop"], 1000.525)
        self.assertEqual(a["raw_block_duration_s"], 1.05)

    def test_touching_half_open_intervals_do_not_overlap(self):
        records = [{"drive_id": fid, "intervals": [{"block_index": 0, "start": start, "stop": start+1}]}
                   for fid, start in [("a", 0), ("b", 1)]]
        self.assertEqual(interval_edges(records), [])

    def test_overlap_propagates_historical_exclusion_transitively(self):
        components = [{"component_id": x, "members": [x]} for x in "abc"]
        edges = [{"left": ["a", 0], "right": ["b", 0]},
                 {"left": ["b", 0], "right": ["c", 0]}]
        result, remaining, excluded = augment_components(
            components, edges, ["a"], {x: ("train", "g") for x in "abc"})
        self.assertEqual(remaining, [])
        self.assertEqual(len(excluded), 3)
        self.assertEqual(result[0]["check_ids"], ["a"])

    def test_assignment_sha_order_and_single_component(self):
        files = [{"drive_id": fid, "group_id": "g"} for fid in "abc"]
        components = [{"component_id": fid, "members": [fid]} for fid in "abc"]
        result = assign_components(files, components)
        last = max("abc", key=lambda x: (selection_key(x), x))
        self.assertEqual(result["file_roles"][last], "check")
        self.assertEqual(list(result["file_roles"].values()).count("fit"), 2)
        one = assign_components(files[:1], components[:1])
        self.assertEqual(one["groups"]["g"]["status"], "blocked_single_component")
        self.assertEqual(one["file_roles"]["a"], "blocked")

    def test_multigroup_component_never_splits(self):
        ordered = sorted(["a", "b", "c"], key=selection_key)
        lo, mid, hi = ordered
        components = [{"component_id": lo, "members": ["low"]},
                      {"component_id": mid, "members": ["shared1", "shared2"]},
                      {"component_id": hi, "members": ["high"]}]
        files = [{"drive_id": "low", "group_id": "g1"},
                 {"drive_id": "shared1", "group_id": "g1"},
                 {"drive_id": "shared2", "group_id": "g2"},
                 {"drive_id": "high", "group_id": "g2"}]
        result = assign_components(files, components)
        self.assertEqual(result["multigroup_conflicting_components"], [mid])
        self.assertEqual(result["file_roles"]["shared1"], "blocked")
        self.assertEqual(result["file_roles"]["shared2"], "blocked")
        self.assertTrue(all(g["status"] == "blocked_multigroup_assignment_conflict"
                            for g in result["groups"].values()))

    def test_consistent_multigroup_component_is_kept_whole(self):
        components = [{"component_id": "shared", "members": ["a1", "a2"]},
                      {"component_id": "other", "members": ["b1", "b2"]}]
        files = [{"drive_id": fid, "group_id": group}
                 for fid, group in [("a1", "g1"), ("a2", "g2"), ("b1", "g1"), ("b2", "g2")]]
        result = assign_components(files, components)
        self.assertEqual(result["multigroup_conflicting_components"], [])
        self.assertEqual(result["file_roles"]["a1"], result["file_roles"]["a2"])
        self.assertEqual(result["file_roles"]["b1"], result["file_roles"]["b2"])

    def test_malformed_clock_dt_extent_and_shape_fail_closed(self):
        for change in ("dt", "clock", "extent", "shape"):
            h, c, _, _ = fixture()
            dt = .00005
            if change == "dt":
                dt = .0001
            elif change == "clock":
                c["clock_native"] = [1., float("nan")]
            elif change == "extent":
                h["attributes"]["/host/Source1/Zone1"]["Extent"][3] -= 1
            else:
                h["shape"] = [2, 1000, 8]
            with self.assertRaises(ValueError):
                conditional_intervals(h, c, dt)

    def test_bad_map_wrong_split_and_coordinate_fail_closed(self):
        for change in ("split", "position", "indices", "source", "width"):
            h, c, rows, maps = fixture()
            if change == "split":
                rows[0]["split"] = "test"
            else:
                # Damage both blocks; the SHA choice must not bypass integrity.
                for mapping in maps.values():
                    if change == "position":
                        mapping["metadata_positions_m"][0] += 1
                    elif change == "indices":
                        mapping["source_channel_indices"][0] = 99
                    elif change == "source":
                        mapping["drive_id"] = "other"
                    else:
                        mapping["source_channel_count"] = 9
            with self.assertRaises(ValueError):
                plan_file("file", rows, maps, h, conditional_intervals(h, c, .00005))

    def test_idempotence_and_divergent_evidence_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight.json"
            write_preflight(output, {"status": "blocked"})
            original = output.read_bytes()
            write_preflight(output, {"status": "blocked"})
            with self.assertRaises(ValueError):
                write_preflight(output, {"status": "ready"})
            self.assertEqual(output.read_bytes(), original)

    def test_real_metadata_integration_remains_blocked_and_train_only(self):
        result = build_preflight(ROOT)
        self.assertEqual(result["status"], "blocked_clock_semantics_not_certified")
        self.assertEqual(result["executable_plan"]["files"], [])
        self.assertEqual(result["executable_plan"]["authorized_waveform_cells"], 0)
        self.assertEqual(result["candidate_plan"]["file_count"], 32)
        self.assertEqual(result["candidate_plan"]["cell_count"], 18_275_000)
        self.assertTrue(result["budget"]["candidate_within_budget"])
        audit = result["conditional_overlap_audit"]
        self.assertEqual(audit["authorized_train_file_count"], 87)
        self.assertEqual(audit["historical_check_file_count"], 43)
        self.assertEqual(audit["raw_block_interval_count"], 486)
        self.assertEqual(audit["newly_excluded_frozen_pool_ids"], [])
        self.assertEqual(audit["direct_frozen_pool_to_historical_check_intersections"], [])
        self.assertEqual(audit["remaining_file_count"], 32)
        self.assertFalse(audit["frozen_pool_collapses_under_hypothesis"])
        assignment = result["candidate_plan"]["assignment"]
        self.assertEqual(assignment["groups"]["multicore/C1 and C4"]["status"],
                         "blocked_single_component")
        self.assertEqual(assignment["multigroup_conflicting_components"], [])
        self.assertFalse(any(result["access"].values()))
        for plan in result["candidate_plan"]["files"]:
            self.assertFalse(plan["executable"])
            self.assertEqual(plan["source_shape"][1], 21000)
            self.assertTrue(all(r["time_start_source_index"] == 0 and
                                r["time_stop_source_index_exclusive"] == 5000
                                for r in plan["regions"]))


if __name__ == "__main__":
    unittest.main()
