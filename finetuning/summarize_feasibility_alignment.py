"""Apply the two report-111 investment gates to completed endpoint evaluations."""
import argparse
import hashlib
import json
from pathlib import Path

from finetuning.run_feasibility_full_coverage import paired_gate


def evaluate_endpoint_comparisons(full, part8, mature):
    """Separate the compatibility effect from improvement over the mature codec."""
    cells = full["cells"]
    if cells <= 0 or any(s["cells"] != cells for s in (part8, mature)):
        raise ValueError("Only matched physical coverage is comparable")
    identity = {(r["drive_id"], r["group_id"], r["cells"], r["windows"],
                 tuple(r["source_channel_indices"])) for r in full["records"]}
    if any(identity != {(r["drive_id"], r["group_id"], r["cells"], r["windows"],
                         tuple(r["source_channel_indices"])) for r in baseline["records"]}
           for baseline in (part8, mature)):
        raise ValueError("Endpoint populations differ")
    comparisons = {}
    for name, baseline in (("compatibility", part8), ("practical_advance", mature)):
        guard = paired_gate(baseline, full)
        checks = dict(guard["checks"])
        object_result = checks.pop("self_contained_bytes_not_increased")
        comparisons[name] = {
            "passed": all(flag is True for flag in checks.values()),
            "checks": checks, "object_bytes_not_increased_descriptive": object_result,
            "unavailable": guard["unavailable"],
            "candidate_nrmse_macro": full["macro_families"],
            "baseline_nrmse_macro": baseline["macro_families"],
            "candidate_rate": full["rate"], "baseline_rate": baseline["rate"],
        }
    return {"kind": "febus-context-alignment-endpoint-gates-v1",
            "cells_per_arm": cells, "comparisons": comparisons, "model_promoted": False,
            "p3_eligible": False, "confirmation_requires_new_sessions": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("full", "part8", "mature"):
        parser.add_argument("--" + arm, required=True, type=Path,
                            help="completed endpoint evaluation summary JSON")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    arms = [(p.parents[2], key) for p, key in
            ((args.full, "B"), (args.part8, "A"), (args.mature, "mature"))]
    training = []
    for path, key in arms:
        if not (path / "COMPLETE").is_file():
            raise ValueError(f"Expected a completed, unselected endpoint: {path}")
        manifest = json.loads((path / "manifest.json").read_text())
        arm = json.loads((path / key / "results.json").read_text())
        expected = 0 if key == "mature" else 8192
        if manifest["smoke"] or arm["updates"] != expected or arm["endpoint"]["heldout_claim"]:
            raise ValueError("Wrong endpoint dose or scope")
        if key != "mature":
            training.append((manifest["schedule_sha256"], arm["initial_state_sha256"],
                             arm["cells"], arm["coverage"]))
    if training[0] != training[1]:
        raise ValueError("Training arms differ in schedule, initialization, or cell exposure")
    result = evaluate_endpoint_comparisons(
        *[json.loads(p.read_text()) for p in (args.full, args.part8, args.mature)])
    result["evaluations_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (args.full, args.part8, args.mature)}
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
