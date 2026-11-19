"""Stress-test D2c metadata under any timestamp location within its raw block."""
import argparse
from collections import Counter
import json
from pathlib import Path

from tools.freeze_feasibility_metadata import sha256_file
from tools.prepare_feasibility_temporal import augment_components, interval_edges


def audit(temporal, freeze):
    pool = freeze["f2_candidate_pool"]
    candidate_ids = {row["drive_id"] for row in pool["files"]}
    historical_checks = set(pool["historical_check_ids"])
    plans = {row["drive_id"]: row for row in temporal["candidate_plan"]["files"]}
    if set(plans) != candidate_ids or len(historical_checks) != 43:
        raise ValueError("Mismatched frozen selection")
    by_id = {row["drive_id"]: row["group_id"]
             for row in pool["files"] + pool["exclusion_ledger"]}
    records = temporal["conditional_overlap_audit"]["all_authorized_train_clocks"]
    if {r["drive_id"] for r in records} != set(by_id) or len(records) != 87:
        raise ValueError("Expected exact historical train clock coverage")
    envelopes = []
    for row in records:
        dur = row["raw_block_duration_s"]
        if dur != 1.05:
            raise ValueError("Duration differs from the declared sensitivity analysis")
        envelopes.append({"drive_id": row["drive_id"], "intervals": [
            {"block_index": i, "start": tick - dur, "stop": tick + dur}
            for i, tick in enumerate(row["clock_native"])]})
    edges = interval_edges(envelopes)
    components, remaining, _ = augment_components(
        pool["components"], edges, historical_checks,
        {fid: ("train", group) for fid, group in by_id.items()})
    remaining_ids = {row["drive_id"] for row in remaining}
    # Preserve old assignments. A new edge between fit and check blocks the
    # entire component instead of silently choosing a replacement check.
    roles = {fid: plan["candidate_role"] if fid in remaining_ids else "excluded"
             for fid, plan in plans.items()}
    for c in components:
        members = set(c["members"]) & remaining_ids
        component_roles = {roles[fid] for fid in members}
        if len(component_roles) > 1:
            for fid in members:
                roles[fid] = "blocked_component_role_conflict"
    contrasts = {}
    for group in sorted({row["group_id"] for row in plans.values()}):
        counts = Counter(roles[fid] for fid in plans if plans[fid]["group_id"] == group)
        contrasts[group] = {"role_counts": dict(counts),
                            "has_candidate_contrast": bool(counts["fit"] and counts["check"])}
    return {
        "kind": "feasibility-clock-envelope-sensitivity-v1",
        "assumptions_not_certified": [
            "Ticks are seconds on a comparable timebase across files.",
            "Each tick lies within the support of its 1.05 s stored block.",
        ],
        "envelope": "[tick - raw_block_duration_s, tick + raw_block_duration_s]",
        "train_files": len(records), "raw_blocks": sum(len(r["intervals"]) for r in envelopes),
        "cross_file_potential_intersections": len(edges),
        "direct_pool_to_historical_check_edges": [
            e for e in edges if (e["left"][0] in candidate_ids and e["right"][0] in historical_checks)
            or (e["right"][0] in candidate_ids and e["left"][0] in historical_checks)],
        "remaining_file_count": len(remaining_ids),
        "removed_file_ids": sorted(candidate_ids - remaining_ids),
        "candidate_role_counts": dict(Counter(roles.values())),
        "candidate_roles": roles, "contrasts": contrasts,
        "components": components,
        "clock_semantics_certified": False, "waveform_cells_read": 0,
        "executable_plan": None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temporal", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(json.loads(args.temporal.read_text()), json.loads(args.freeze.read_text()))
    result["input_and_source_sha256"] = {str(path): sha256_file(path) for path in
        (args.temporal, args.freeze, Path(__file__).resolve().relative_to(Path.cwd()),
         Path("tools/prepare_feasibility_temporal.py"))}
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != text:
            raise ValueError("Existing evidence differs; choose a new destination")
    else:
        with args.output.open("x") as out:
            out.write(text)
    print(json.dumps({k: result[k] for k in ("cross_file_potential_intersections",
                     "remaining_file_count", "candidate_role_counts", "removed_file_ids")}))


if __name__ == "__main__":
    main()
