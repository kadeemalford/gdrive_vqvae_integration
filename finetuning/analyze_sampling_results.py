#!/usr/bin/env python3
"""Audit completed sampling CV and export metadata-aware, paired test summaries."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("correlation", "relative_rmse", "snr_db")


def nominal_frequency(title: str) -> str:
    match = re.search(r"([\d.]+)\s*Hz", title, re.IGNORECASE)
    return match.group(1) if match else "unknown"


def grouped_metrics(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{**dict(zip(keys, key)), "windows": len(values),
             **{metric: mean(float(v[metric]) for v in values) for metric in METRICS}}
            for key, values in sorted(groups.items())]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.experiment_dir / "COMPLETE").exists():
        raise ValueError("Experiment must be complete before auditing all folds")
    rows, training = [], []
    for fold in range(4):
        samples = Path(f"data/finetune/fold_{fold}")
        with (samples / "samples_manifest.csv").open() as stream:
            metadata = {Path(r["sample_path"]).resolve(): r for r in csv.DictReader(stream)}
        folder = args.experiment_dir / f"fold_{fold}"
        with (folder / "evaluation/per_sample_metrics.csv").open() as stream:
            values = list(csv.DictReader(stream))
        labels_by_path = defaultdict(set)
        for row in values:
            if row["checkpoint"] in labels_by_path[row["sample_path"]]:
                raise ValueError("Duplicate checkpoint/window evaluation row")
            labels_by_path[row["sample_path"]].add(row["checkpoint"])
            meta = metadata[Path(row["sample_path"]).resolve()]
            if meta["split"] != "test" or int(meta["fold"]) != fold:
                raise ValueError("Evaluation contains a non-test window")
            rows.append({**row, "fold": fold, "drive_id": meta["drive_id"], "title": meta["title"],
                         "nominal_frequency_hz": nominal_frequency(meta["title"]),
                         "dataset": meta["dataset"], "dt_s": meta["dt_s"], "dx_m": meta["dx_m"],
                         "channels": meta["channels"]})
        if any(labels != {"window", "family-group"} for labels in labels_by_path.values()):
            raise ValueError("Evaluation must pair both checkpoints on each window")
        expected = {path for path, meta in metadata.items() if meta["split"] == "test"}
        if {Path(path).resolve() for path in labels_by_path} != expected:
            raise ValueError("Evaluation does not cover all prepared test windows")
        for mode in ("window", "family-group"):
            summary = json.loads((folder / mode / "summary.json").read_text())
            with (folder / mode / "metrics.csv").open() as stream:
                epochs = list(csv.DictReader(stream))
            if [int(r["epoch"]) for r in epochs] != list(range(1, summary["epochs_completed"] + 1)):
                raise ValueError(f"Duplicate or missing epochs in {folder / mode}")
            training.append({"fold": fold, "sampling": mode, "epochs_completed": summary["epochs_completed"],
                             "best_epoch": summary["best_epoch"], "best_val_loss": summary["best_val_loss"],
                             "last_train_loss": float(epochs[-1]["train_loss"]),
                             "last_val_loss": float(epochs[-1]["val_loss"]),
                             "last_val_correlation": float(epochs[-1]["val_correlation"]),
                             "epoch_seconds": sum(float(e["elapsed_sec"]) for e in epochs)})
    configs = grouped_metrics(rows, ("fold", "checkpoint", "family", "configuration"))
    files = grouped_metrics(rows, ("fold", "checkpoint", "family", "configuration", "drive_id", "title",
                                   "nominal_frequency_hz", "dataset", "dt_s", "dx_m", "channels"))
    families = grouped_metrics(configs, ("checkpoint", "family"))
    overall = grouped_metrics(families, ("checkpoint",))
    weighted = grouped_metrics(rows, ("checkpoint",))
    paired = defaultdict(dict)
    for row in configs:
        paired[(row["fold"], row["family"], row["configuration"])][row["checkpoint"]] = row
    deltas = []
    for (fold, family, configuration), arms in sorted(paired.items()):
        if set(arms) != {"window", "family-group"}:
            raise ValueError("Unpaired evaluation configuration")
        deltas.append({"fold": fold, "family": family, "configuration": configuration,
                       **{f"delta_{m}": arms["family-group"][m] - arms["window"][m] for m in METRICS}})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("configurations", configs), ("files", files), ("training", training), ("paired_deltas", deltas)):
        write_csv(output / f"{name}.csv", data)
    (output / "summary.json").write_text(json.dumps({"macro_families": families, "macro_overall": overall,
        "window_weighted": weighted, "configurations": len(paired), "correlation_configuration_wins":
        sum(r["delta_correlation"] > 0 for r in deltas), "rmse_configuration_wins":
        sum(r["delta_relative_rmse"] < 0 for r in deltas)}, indent=2) + "\n")
    print(f"[audit] {len(paired)} configurations, {len(rows)} paired metric rows -> {output}")


if __name__ == "__main__":
    main()
