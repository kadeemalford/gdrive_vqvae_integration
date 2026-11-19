"""Revalidate frozen F2 coordinates under explicitly accepted clock assumptions."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from tools.prepare_feasibility_temporal import interval_edges


def revalidate(original):
    audit = original["conditional_overlap_audit"]
    roles = audit["assignment"]["file_roles"]
    excluded = {r["drive_id"] for r in audit["excluded_connected_to_historical_checks"]}
    records = []
    for clock in audit["all_authorized_train_clocks"]:
        duration = clock["raw_block_duration_s"]
        if duration != 1.05:
            raise ValueError("Unexpected raw block duration")
        intervals = [{"block_index": i, "start": tick - duration / 2,
                      "stop": tick - duration / 2 + duration}
                     for i, tick in enumerate(clock["clock_native"])]
        if intervals != clock["intervals"]:
            raise ValueError("Centered convention differs from the frozen hypothesis")
        records.append({"drive_id": clock["drive_id"], "intervals": intervals})
    edges = interval_edges(records)
    forbidden = []
    for edge in edges:
        a, b = edge["left"][0], edge["right"][0]
        ra, rb = roles.get(a), roles.get(b)
        if ({ra, rb} == {"fit", "check"}
                or a in excluded and rb in ("fit", "check")
                or b in excluded and ra in ("fit", "check")):
            forbidden.append(edge)
    if forbidden or audit["newly_excluded_frozen_pool_ids"]:
        raise ValueError("Overlap must be reconciled before waveforms can be read")
    files = []
    clock_by_file = {r["drive_id"]: r["intervals"] for r in records}
    for row in original["candidate_plan"]["files"]:
        if roles[row["drive_id"]] != row["candidate_role"]:
            raise ValueError("Frozen roles changed")
        if row["candidate_role"] not in ("fit", "check"):
            continue
        interval = clock_by_file[row["drive_id"]][row["block_index"]]
        if row["raw_block_conditional_clock_interval_s"] != [interval["start"], interval["stop"]]:
            raise ValueError("Selected block interval differs")
        for region in row["regions"]:
            start = interval["start"] + region["time_start_source_index"] / 20000
            stop = interval["start"] + region["time_stop_source_index_exclusive"] / 20000
            if any(not math.isclose(a, b, rel_tol=0, abs_tol=5e-7)
                   for a, b in zip((start, stop), region["conditional_clock_interval_s"])):
                raise ValueError("Selected region interval differs")
        files.append({**row, "executable": True})
    cells = sum(r["cells"] for r in files)
    if not 0 < cells <= 64_000_000:
        raise ValueError("Cell budget exceeded")
    return {"kind": "feasibility-centered-clock-revalidation-v1",
            "clock_assumptions": "User accepted centered timestamps in seconds on a common timebase, without export offsets. Not vendor certification.",
            "cross_file_block_intersections": len(edges), "forbidden_intersections": 0,
            "role_file_counts": dict(Counter(r["candidate_role"] for r in files)),
            "cells": cells, "waveform_cells_read": 0,
            "limits": "No crossing block boundaries, no session independence claim, C1/C4 still blocked.",
            "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = revalidate(json.loads(raw))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != text:
            raise ValueError("Existing evidence differs")
    else:
        with args.output.open("x") as target:
            target.write(text)
    print(result["role_file_counts"], result["cells"], result["cross_file_block_intersections"])


if __name__ == "__main__":
    main()
