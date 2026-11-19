"""Artifact-only registry: isolate cohorts and audit the completed research budget."""
import argparse
import json
from pathlib import Path

from finetuning.audit_prospective_preparation import sha256, write_json

BASE = Path("results/experiments")
OUTPUT = BASE / "research_decision_registry_20261115"


def read_history(path, endpoint):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    updates = [row["update"] for row in rows]
    if not updates or updates[-1] != endpoint or any(a >= b for a, b in zip(updates, updates[1:])):
        raise ValueError("Incomplete or nonmonotonic scientific history")
    return rows[-1]


def point(path, name, status):
    result = json.loads(path.read_text())
    return {
        "name": name, "status": status, "artifact": str(path), "artifact_sha256": sha256(path),
        "macro_families_nrmse_star": result["macro_families"], "stream_bpc": result["stream_bpc"],
        "stream_bytes": result["stream_bytes"], "cells": result["cells"],
        "files": len(result["records"]), "groups": result["groups"],
        "file_population": sorted((row["group_id"], row["drive_id"], row["cells"]) for row in result["records"]),
    }


def validate_cohort(points):
    if not points:
        raise ValueError("Empty research cohort")
    first = points[0]
    for row in points:
        if (row["cells"] != first["cells"] or row["file_population"] != first["file_population"]
                or row["groups"].keys() != first["groups"].keys()):
            raise ValueError("Research cohorts must not mix acquired populations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budgets = []
    lifting_audit = json.loads((BASE / "lifting_training_budget_audit_20261115.json").read_text())
    for record in lifting_audit["records"]:
        folder = BASE / f"lifting_p2_{record['rate']}_fold0_20261114" / record["arm"]
        read_history(folder / "history.jsonl", record["updates"])
        budgets.append({"campaign": "H1_lifting", "arm": f"{record['rate']}/{record['arm']}",
                        "updates": record["updates"], "cells": record["cells_seen"],
                        "history_sha256": sha256(folder / "history.jsonl")})
    for campaign, directory, arms, endpoint in [
        ("H2_joint", "joint_anchor_vq_fold0_20261115", ("independent", "closed-loop"), 32768),
        ("encoder_observation", "joint_encoder_observation_fold0_20261115", ("local-analysis", "full-observation"), 8192),
        ("dense_product", "dense_full_tile_product_v2_fold0_20261115", ("flat", "product"), 8192),
        ("local_internal_norm", "local_internal_norm_fold0_20261115", ("global", "local"), 8192),
    ]:
        if not (BASE / directory / "COMPLETE").exists():
            raise ValueError("Cannot consolidate an incomplete campaign")
        for arm in arms:
            history = BASE / directory / "arms" / arm / "history.jsonl"
            last = read_history(history, endpoint)
            budgets.append({"campaign": campaign, "arm": arm, "updates": endpoint,
                "cells": last.get("cells", last.get("cells_seen")), "history_sha256": sha256(history)})
    cohorts = {}
    joint = BASE / "joint_anchor_vq_fold0_20261115/expanded"
    observation = BASE / "joint_encoder_observation_fold0_20261115/expanded"
    cohorts["train_P1_129"] = [
        point(joint / "closed-loop/practical_reference/results.json", "mature_reference", "reference"),
        point(joint / "independent/check_030720/results.json", "H2_independent", "not_promoted"),
        point(joint / "closed-loop/check_016384/results.json", "H2_joint", "not_promoted"),
        point(observation / "local-analysis/check_002048/results.json", "encoder_local", "not_promoted"),
        point(observation / "full-observation/check_002048/results.json", "encoder_full", "not_promoted"),
    ]
    product = BASE / "dense_full_tile_product_v2_fold0_20261115/arms"
    norm = BASE / "local_internal_norm_fold0_20261115/arms"
    cohorts["train_expanded_136"] = [
        point(norm / "local/expanded/mature_reference/check_000000/results.json", "mature_reference", "reference"),
        point(product / "flat/expanded/endpoint/check_008192/results.json", "flat_endpoint", "not_selected"),
        point(product / "product/expanded/selected/check_004096/results.json", "product_selected", "not_promoted"),
        point(product / "product/expanded/endpoint/check_008192/results.json", "product_endpoint", "not_selected"),
        point(norm / "local/expanded/selected/check_004096/results.json", "local_norm_selected", "not_promoted"),
        point(norm / "local/expanded/endpoint/check_008192/results.json", "local_norm_endpoint", "not_selected"),
    ]
    curves = BASE / "frozen_dense_rate_match_fold0_20261115"
    for variant in ("flat", "product"):
        for path in sorted((curves / variant).glob("multiplier_*/results.json")):
            cohorts["train_expanded_136"].append(point(path, f"frozen_rate/{variant}/{path.parent.name}", "unpromoted_rate_probe"))
    validation = BASE / "frozen_development_validation_fold0_20261115"
    cohorts["consumed_development_validation_7220"] = [
        point(validation / name / "results.json", name, "reference" if name == "mature_reference" else "not_promoted")
        for name in ("mature_reference", "independent", "h2_joint", "encoder_local", "encoder_full")]
    for values in cohorts.values():
        validate_cohort(values)
    total = sum(row["updates"] for row in budgets)
    if total != 245760:
        raise ValueError("Unexpected completed optimization budget")
    result = {
        "completed_scientific_optimization_updates_since_report90": total,
        "budget_scope": "Includes fixed H1 quantization-step optimization; excludes technical smokes, historical pretraining and Lloyd iterations. Updates are not equal compute across architectures.",
        "budgets": budgets, "cohorts": cohorts, "global_cross_cohort_ranking": False,
        "parameter_updates_in_consolidation": 0, "waveforms_reopened": False,
        "final_test_access": False, "independent_confirmation": False,
        "decision": "Keep mature high-fidelity reference. No promotion, P3, or automatic LR/gate/seed sweep. Preserve all measured rate points without global dominance claim.",
        "limits": "Historically consumed development; different cohorts are never merged. Missing new-model validation is deliberate after failed gates, not coverage silently omitted.",
    }
    write_json(args.output / "registry.json", result)
    (args.output / "COMPLETE").touch()
    print(json.dumps({"scientific_optimization_updates": total,
                      "cohort_points": {key: len(values) for key, values in cohorts.items()},
                      "waveforms_reopened": False}), flush=True)


if __name__ == "__main__":
    main()
