"""Frozen-cohort conditioning comparisons and unrounded prospective opportunity gate."""
import argparse
import json
from pathlib import Path

from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.compare_physical_probe import compare_records
from finetuning.run_lifting_study import macro_scores

METHODS = ("constant", "hardware", "category")


def cohort(result, groups):
    groups = set(groups)
    records = [r for r in result["records"] if r["group_id"] in groups]
    if not groups or {r["group_id"] for r in records} != groups:
        raise ValueError("Missing configuration in conditioning cohort")
    frames = sum(r["frame_bytes"] for r in records)
    cells = sum(r["cells"] for r in records)
    total = frames + result["setup_bytes"]
    return {"records": records, **macro_scores(records), "cells": cells,
            "windows": sum(r["windows"] for r in records), "frame_bytes": frames,
            "setup_bytes": result["setup_bytes"], "stream_bytes": total,
            "stream_bpc": 8 * total / cells}


def opportunity(normal, heldout, seen):
    excluded = {method: cohort(result, heldout) for method, result in normal.items()}
    included = {method: cohort(result, seen) for method, result in normal.items()}
    hardware = excluded["hardware"]
    reductions = {}
    for method in ("constant", "category"):
        control = excluded[method]
        reference = control["macro_families"]
        reductions[method] = 1 - hardware["macro_families"] / reference if reference > 0 else 0.
    checks = {
        "heldout_error_reduction_ge_5pct_vs_both": all(v >= .05 for v in reductions.values()),
        "heldout_complete_bytes_not_greater_vs_both": all(
            hardware["stream_bytes"] <= excluded[method]["stream_bytes"] for method in ("constant", "category")),
        "heldout_no_group_regression_vs_both": all(
            hardware["groups"][g] <= excluded[method]["groups"][g]
            for g in heldout for method in ("constant", "category")),
        "seen_no_group_regression_vs_constant": all(
            included["hardware"]["groups"][g] <= included["constant"]["groups"][g] for g in seen),
    }
    return {"opportunity_gate": all(checks.values()), "checks": checks,
            "heldout_relative_error_reduction": reductions,
            "limits": "Scalar opportunity only; inspect physical guards and missing-context stress. No automatic promotion."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.study / "COMPLETE").exists():
        raise ValueError("All registered arms must complete before comparison")
    phase = args.study / "scientific"
    protocol = json.loads((phase / "constant/protocol.json").read_text())
    seen, heldout = protocol["plan"]["seen"], protocol["plan"]["heldout"]
    normal, missing, hashes = {}, {}, {}
    for method in METHODS:
        folder = phase / method
        if not (folder / "COMPLETE").exists():
            raise ValueError("Incomplete conditioning arm")
        other = json.loads((folder / "protocol.json").read_text())
        if other["plan"] != protocol["plan"] or other["source_sha256"] != protocol["source_sha256"]:
            raise ValueError("Unmatched conditioning plans or source")
        for label, destination in (("normal", normal), ("missing", missing)):
            path = folder / "check_016384" / label / "results.json"
            hashes[str(path)] = sha256(path)
            destination[method] = json.loads(path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    all_groups = seen + heldout
    rows = []
    for label, groups in (("all", all_groups), ("seen", seen), ("heldout", heldout)):
        for method in METHODS:
            for stress, source in (("normal", normal), ("missing", missing)):
                result = cohort(source[method], groups)
                rows.append({"cohort": label, "method": method, "stress": stress,
                             "macro_families": result["macro_families"],
                             "stream_bytes": result["stream_bytes"], "stream_bpc": result["stream_bpc"],
                             "groups": result["groups"], "cells": result["cells"], "windows": result["windows"]})
        for reference in ("constant", "category"):
            left, right = cohort(normal[reference], groups), cohort(normal["hardware"], groups)
            write_json(args.output / f"PHYSICAL_{label}_hardware_VS_{reference}.json", compare_records(left, right))
        for method in METHODS:
            write_json(args.output / f"MISSING_{label}_{method}.json",
                       compare_records(cohort(normal[method], groups), cohort(missing[method], groups)))
    result = opportunity(normal, heldout, seen)
    result.update(rows=rows, input_sha256=hashes, seen=seen, heldout=heldout,
                  automatic_promotion=False, parameter_updates=0, waveforms_reopened=False)
    write_json(args.output / "results.json", result)
    (args.output / "COMPLETE").touch()
    print(json.dumps({k: v for k, v in result.items() if k != "rows" and k != "input_sha256"}), flush=True)


if __name__ == "__main__":
    main()
