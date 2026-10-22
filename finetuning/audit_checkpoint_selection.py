#!/usr/bin/env python3
"""Retrospective selection/sampling diagnostics on existing validation outputs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import numpy as np

try:
    from .audit_rate_continuation import ROOT, INVARIANTS, digest, keyed, read_rows, write_csv
except ImportError:
    from audit_rate_continuation import ROOT, INVARIANTS, digest, keyed, read_rows, write_csv


METRICS = ("relative_rmse", "correlation", "ordinary_channel_p90_relative_rmse")
LABELS = ("initial", "continued", "loss_selected", "midpoint")
SIGN = np.array([1., -1., 1.])
EPS = 1e-9  # Numerical comparison only, not an application acceptance tolerance.


def sample_indices(metadata, per_file, seed, design):
    if per_file < 1 or design not in ("random", "time_stratified"):
        raise ValueError("Invalid sampling design/budget")
    files = defaultdict(list)
    for index, row in enumerate(metadata):
        files[row["drive_id"]].append(index)
    rng, selected = random.Random(seed), []
    for file_id in sorted(files):
        indices = sorted(files[file_id], key=lambda i: metadata[i]["sample_path"])
        count = min(per_file, len(indices))
        if design == "random":
            selected.extend(rng.sample(indices, count))
        else:
            indices.sort(key=lambda i: (int(metadata[i]["block_index"]), int(metadata[i]["time_start"]),
                                        metadata[i]["sample_path"]))
            selected.extend(rng.choice(list(chunk)) for chunk in np.array_split(indices, count))
    return np.array(sorted(selected), dtype=int)


def group_scores(values, indices, group_indices):
    """Candidate x window x metric -> candidate x configuration x metric."""
    indices = np.asarray(indices)
    scores = []
    for group in sorted(set(group_indices)):
        chosen = indices[group_indices[indices] == group]
        if not len(chosen):
            raise ValueError("Selection/complement missing a configuration")
        scores.append(values[:, chosen, :].mean(axis=1))
    return np.stack(scores, axis=1)


def choose(scores, weights, guarded):
    regret = (scores - scores[0:1]) * SIGN
    feasible = np.all(regret <= EPS, axis=(1, 2))
    macro = scores[:, :, 0] @ weights
    eligible = np.flatnonzero(feasible) if guarded else np.arange(len(scores))
    return int(eligible[np.argmin(macro[eligible])]), feasible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    full = experiment / "full_validation"
    if not all(p.is_file() for p in (experiment / "COMPLETE", full / "COMPLETE")):
        raise ValueError("Completed frozen experiment required")
    source_protocol = json.loads((full / "protocol.json").read_text())
    if source_protocol["fold"] != 0 or source_protocol["split"] != "val" or not source_protocol.get("score_reference", "").startswith("robust-escape-v1"):
        raise ValueError("Common robust fold-0 validation required")
    manifest = ROOT / "data/finetune/fold_0/samples_manifest.csv"
    manifest_rows = keyed([r for r in read_rows(manifest) if r["split"] == "val"])
    paths = sorted(manifest_rows)
    metadata = [{**manifest_rows[p], "sample_path": p} for p in paths]
    if len(paths) != source_protocol["windows"]:
        raise ValueError("Manifest/window count mismatch")
    groups = sorted({(r["family"], r["configuration"]) for r in metadata})
    group_indices = np.array([groups.index((r["family"], r["configuration"])) for r in metadata])
    families = Counter(f for f, _ in groups)
    weights = np.array([1 / len(families) / families[f] for f, _ in groups])
    hashes = {str(p): digest(p) for p in (manifest, full / "protocol.json", Path(__file__),
                                        Path(__file__).with_name("audit_rate_continuation.py"))}
    all_trials, detection_trials = [], []
    for arm in ("2x", "4x"):
        candidates, reference = [], None
        for label in LABELS:
            name = f"{label}_{arm}"
            summary = full / f"{name}.json"
            if not summary.is_file():
                raise ValueError("Missing completed candidate summary")
            attempts = [(p, read_rows(p)) for p in sorted(full.glob(f"{name}.windows.attempt_*.csv"))]
            complete = [(p, rows) for p, rows in attempts if len(rows) == len(paths)]
            if len(complete) != 1:
                raise ValueError("Expected one complete attempt per candidate")
            path, rows = complete[0]
            rows = keyed(rows)
            if rows.keys() != manifest_rows.keys():
                raise ValueError("Candidate/manifest coordinates differ")
            for p in paths:
                if any(rows[p][k] != manifest_rows[p][k] for k in ("family", "configuration", "drive_id")):
                    raise ValueError("Candidate/manifest provenance differs")
                if reference and any(rows[p][k] != reference[p][k] for k in INVARIANTS):
                    raise ValueError("Candidate reference/rate mismatch")
            reference = reference or rows
            candidates.append([[float(rows[p][metric]) for metric in METRICS] for p in paths])
            hashes[str(path)] = digest(path)
            hashes[str(summary)] = digest(summary)
        values = np.array(candidates)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite candidate metrics")
        whole = group_scores(values, np.arange(len(paths)), group_indices)
        for design in ("random", "time_stratified"):
            for budget in (2, 4, 8, 16):
                for seed in range(2027, 2077):
                    selected = sample_indices(metadata, budget, seed, design)
                    remaining = np.setdiff1d(np.arange(len(paths)), selected)
                    sample = group_scores(values, selected, group_indices)
                    complement = group_scores(values, remaining, group_indices)
                    for guarded in (False, True):
                        chosen, feasible = choose(sample, weights, guarded)
                        full_delta = (whole[chosen] - whole[0]) * SIGN
                        delta = (complement[chosen] - complement[0]) * SIGN
                        all_trials.append({"arm": arm, "design": design, "per_file": budget, "seed": seed,
                            "selector": "guarded" if guarded else "macro", "selected": LABELS[chosen],
                            "selection_windows": len(selected), "complement_windows": len(remaining),
                            "sample_safe": int(feasible[chosen]),
                            "complement_any_regression": int(np.any(delta > EPS)),
                            "full_any_regression": int(np.any(full_delta > EPS)),
                            "false_safe_on_complement": int(feasible[chosen] and np.any(delta > EPS)),
                            "complement_macro_rmse_delta": float(delta[:, 0] @ weights),
                            "complement_worst_rmse_delta": float(delta[:, 0].max()),
                            "complement_worst_correlation_loss": float(delta[:, 1].max()),
                            "complement_worst_channel_p90_delta": float(delta[:, 2].max())})
                    # Detection diagnostics for the fixed continued checkpoint, independent of selection.
                    for gi, (family, configuration) in enumerate(groups):
                        detection_trials.append({"arm": arm, "design": design, "per_file": budget,
                            "seed": seed, "family": family, "configuration": configuration,
                            "sample_rmse_regression": int(sample[1, gi, 0] - sample[0, gi, 0] > EPS),
                            "full_rmse_regression": int(whole[1, gi, 0] - whole[0, gi, 0] > EPS)})
    summaries = []
    grouped = defaultdict(list)
    for row in all_trials:
        grouped[(row["arm"], row["design"], row["per_file"], row["selector"])].append(row)
    for (arm, design, budget, selector), rows in sorted(grouped.items()):
        summary = {"arm": arm, "design": design, "per_file": budget, "selector": selector,
                   "trials": len(rows), "selection_windows": rows[0]["selection_windows"]}
        for key in ("sample_safe", "complement_any_regression", "full_any_regression", "false_safe_on_complement",
                    "complement_macro_rmse_delta", "complement_worst_rmse_delta",
                    "complement_worst_correlation_loss", "complement_worst_channel_p90_delta"):
            summary[f"mean_{key}"] = float(np.mean([r[key] for r in rows]))
        summary.update({f"selected_{label}_fraction": sum(r["selected"] == label for r in rows) / len(rows)
                        for label in LABELS})
        summaries.append(summary)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"kind": "retrospective_checkpoint_selection_audit_v1", "training_updates": 0,
                "seeds": [2027, 2076], "per_file_budgets": [2, 4, 8, 16], "candidates": LABELS,
                "guard_metrics": METRICS, "guard_epsilon": EPS, "inputs_sha256": hashes,
                "warning": "Retrospective reused validation, including candidate/hypothesis discovery. Complement is not untouched test evidence. Seed frequencies are not independent-acquisition probabilities. Strict zero-regression guard is diagnostic, not an application tolerance or final selector recommendation."}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name, rows in (("trials.csv", all_trials), ("summary.csv", summaries), ("detection.csv", detection_trials)):
        write_csv(output / name, rows)
    (output / "COMPLETE").write_text("Retrospective selection audit complete; no training/inference/test access.\n")
    print(f"{len(all_trials)} selector trials; {len(detection_trials)} configuration detection rows; {output}")


if __name__ == "__main__":
    main()
