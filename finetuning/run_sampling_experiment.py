#!/usr/bin/env python3
"""Paired, isolated CV runs. Pilot evaluates validation only, full evaluates test."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], log: Path) -> None:
    print(f"[experiment] {log}", flush=True)
    with log.open("w") as stream:
        subprocess.run([sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def check_splits(directory: Path, test_fold: int) -> None:
    with (directory / "samples_manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        key = (row["family"], row["configuration"])
        groups.setdefault(key, set()).add(row["split"])
        if (int(row["fold"]) == test_fold) != (row["split"] == "test"):
            raise ValueError("Manifest test fold mismatch")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Configuration crosses split boundaries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # Refuse accidental overwrite of checkpoints and logs.
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"pilot": args.pilot, "seed": 2026, "epochs": 2 if args.pilot else 10,
                "steps_per_epoch": 128 if args.pilot else 0, "sampling": ["window", "family-group"],
                "evaluation_split": "val" if args.pilot else "test", "selection": "minimum window-weighted val_loss",
                "lr": 1e-5, "base_checkpoint": "models/febus_finetune_base/best.pt"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for fold in range(4):
        samples = Path(f"data/finetune/fold_{fold}")
        check_splits(ROOT / samples, fold)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir()
        for mode in protocol["sampling"]:
            target = fold_dir / mode
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", protocol["base_checkpoint"], "--samples-dir", str(samples),
                       "--output-dir", str(target), "--sampling", mode, "--epochs", str(protocol["epochs"]),
                       "--steps-per-epoch", str(protocol["steps_per_epoch"]), "--device", "cuda"]
            if args.pilot:
                command += ["--max-val-samples", "128"]
            run(command, fold_dir / f"{mode}.log")
        command = ["finetuning/evaluate_checkpoints.py", "--repo-root", args.repo_root,
                   "--samples-dir", str(samples / protocol["evaluation_split"]),
                   "--output-dir", str(fold_dir / "evaluation"), "--device", "cuda"]
        for mode in protocol["sampling"]:
            command += ["--checkpoint", f"{mode}={fold_dir / mode / 'best.pt'}"]
        if args.pilot:
            command += ["--manifest", str(samples / "samples_manifest.csv"), "--max-samples-per-group", "16"]
        run(command, fold_dir / "evaluation.log")
    run(["finetuning/summarize_sampling_experiment.py", "--experiment-dir", str(output)], output / "report.log")
    (output / "COMPLETE").write_text("Training and evaluation completed for all four folds.\n")
    print(f"[experiment] complete: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
