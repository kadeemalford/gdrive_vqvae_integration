#!/usr/bin/env python3
"""Read-only 4x research registry; compare only matching fold/reference/rate cohorts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .audit_rate_continuation import digest, read_rows, write_csv
    from .generalist_guard import assess, grouped
    from .run_sampling_experiment import ROOT
except ImportError:
    from audit_rate_continuation import digest, read_rows, write_csv
    from generalist_guard import assess, grouped
    from run_sampling_experiment import ROOT


REGISTRY = (
    (0, "source24", "guarded_decoder_pilot_20261019", "full_guarded"),
    (0, "continued40", "robust_rate_continuation_20261018", "continued_rate_4x"),
    (0, "loss38", "frozen_rate_stability_20261018", "loss_selected_4x"),
    (0, "midpoint24_40", "frozen_rate_stability_20261018", "midpoint_4x"),
    (0, "low_lr_full8", "guarded_decoder_pilot_20261019", "full_macro"),
    (0, "low_lr_decoder8", "guarded_decoder_pilot_20261019", "decoder_macro"),
    (0, "channel_aux8", "channel_tail_pilot_20261019", "channel_tail_macro"),
    (1, "source24", "channel_tail_fold1_20261019", "full_guarded"),
    (1, "continued39", "high_rate_continuation_20261015", "continued_rate4"),
    (1, "low_lr_full8", "channel_tail_fold1_20261019", "full_macro"),
    (1, "channel_aux8", "channel_tail_fold1_20261019", "channel_tail_macro"),
)


def metric_vector(rows):
    return np.array([value for _, row in sorted(grouped(rows).items())
                     for value in (float(row["relative_rmse"]), -float(row["correlation"]),
                                   float(row["ordinary_channel_p90_relative_rmse"]))])


def dominates(before, after, epsilon=1e-9):
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Incompatible/nonfinite frontier vectors")
    return bool(np.all(before <= after + epsilon) and np.any(before < after - epsilon))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profiles, hashes = [], {}
    for fold, name, experiment, label in REGISTRY:
        root = ROOT / "results/experiments" / experiment
        full = root / "full_validation"
        if not (root / "COMPLETE").is_file() or not (full / "COMPLETE").is_file():
            raise ValueError(f"Wait for complete evidence: {experiment}")
        protocol = json.loads((full / "protocol.json").read_text())
        summary = next(r for r in read_rows(full / "summary.csv") if r["label"] == label)
        groups = read_rows(full / f"{label}.groups.csv")
        source = next(c for c in protocol["candidates"] if c["label"] == label)
        if protocol["fold"] != fold or protocol["split"] != "val":
            raise ValueError("Registry fold/split mismatch")
        checkpoint = Path(source["checkpoint"])
        available = checkpoint.is_file()
        if available and digest(checkpoint) != source["sha256"]:
            raise ValueError(f"Checkpoint changed after measurement: {checkpoint}")
        for path in (full / "protocol.json", full / "summary.csv", full / f"{label}.groups.csv"):
            hashes[str(path)] = digest(path)
        cohort = (fold, protocol["validation_hash"], protocol.get("score_reference", "escape-v1"),
                  summary["latent_geometry"], summary["model_parameters"], summary["num_embeddings"],
                  summary["input_cells"], summary["fixed_index_bits"], summary["side_bytes"])
        profiles.append({"id": f"fold{fold}_{name}", "fold": fold, "name": name, "experiment": experiment,
                         "label": label, "reported_epoch": int(summary["selected_checkpoint_epoch"]),
                         "policy": summary["policy"], "checkpoint": str(checkpoint), "checkpoint_available": available,
                         "checkpoint_sha256": source["sha256"],
                         "macro_relative_rmse": float(summary["macro_relative_rmse"]),
                         "macro_correlation": float(summary["macro_correlation"]),
                         "macro_channel_p90_relative_rmse": float(summary["macro_ordinary_channel_p90_relative_rmse"]),
                         "macro_reconstruction_rms_ratio": float(summary["macro_reconstruction_rms_ratio"]),
                         "macro_fft_phase_alignment": float(summary["macro_fft_phase_alignment"]),
                         "pooled_index_plus_side_ratio": float(summary["pooled_index_plus_side_ratio"]),
                         "worst_configuration_rmse": max(float(r["relative_rmse"]) for r in groups),
                         "_groups": groups, "_vector": metric_vector(groups), "_cohort": cohort})
    decisions, choices = {}, {}
    for profile in profiles:
        baseline = next(p for p in profiles if p["fold"] == profile["fold"] and p["name"] == "source24")
        if profile["_cohort"] != baseline["_cohort"]:
            raise ValueError("Reference/rate mismatch: do not pool incompatible candidates")
        decision = assess(baseline["_groups"], profile["_groups"])
        decisions[profile["id"]] = decision
        profile.update(guard_eligible=decision["eligible"], guard_violation_count=len(decision["violations"]),
                       dominated_by="|".join(other["id"] for other in profiles if other["id"] != profile["id"]
                           and other["_cohort"] == profile["_cohort"] and other["checkpoint_available"]
                           and dominates(other["_vector"], profile["_vector"])))
        profile["frontier_available"] = profile["checkpoint_available"] and not profile["dominated_by"]
    for fold in sorted({p["fold"] for p in profiles}):
        eligible = [p for p in profiles if p["fold"] == fold and p["checkpoint_available"] and p["guard_eligible"]]
        choices[str(fold)] = min(eligible, key=lambda p: p["macro_relative_rmse"])["id"] if eligible else None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    public = [{k: v for k, v in p.items() if not k.startswith("_")} for p in profiles]
    write_csv(output / "profiles.csv", public)
    warning = ("Retrospective registry, not a prospective selector win, final promotion or one model validated across folds. "
               "Only matching fold/reference/rate cohorts compared; 18 group metrics exclude amplitude/frequency acceptance. "
               "Unequal pretraining/optimizer/compute histories; no causal attribution from profile ranking. "
               "Missing checkpoints remain historical evidence, not deployable candidates. Independent acquisitions and physical tolerances absent.")
    (output / "research_choices.json").write_text(json.dumps({"choices_by_fold": choices, "warning": warning,
                                                             "guard_decisions": decisions}, indent=2) + "\n")
    hashes[str(Path(__file__))] = digest(Path(__file__))
    hashes[str(Path(__file__).with_name("generalist_guard.py"))] = digest(Path(__file__).with_name("generalist_guard.py"))
    (output / "protocol.json").write_text(json.dumps({"kind": "retrospective_generalist_4x_registry_v1",
        "inputs_sha256": hashes, "training_updates": 0, "inference_windows": 0,
        "epsilon": 1e-9, "warning": warning}, indent=2) + "\n")
    (output / "COMPLETE").write_text("Read-only 4x candidate registry complete; no promotion or new training/inference.\n")
    print(json.dumps(choices, indent=2))
    for p in public:
        print(p["id"], "eligible", p["guard_eligible"], "available", p["checkpoint_available"], "frontier", p["frontier_available"])


if __name__ == "__main__":
    main()
