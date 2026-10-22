#!/usr/bin/env python3
"""Manual paired pilot: fold 2 has escapes in train, none in validation.

Fresh optimizers, matched windows/update budget and initialization; no test inference.
"""
import argparse
import csv
import fcntl
import json
import shutil
from pathlib import Path

try:
    from .run_sampling_experiment import ROOT, check_splits, run, training_plan
except ImportError:
    from run_sampling_experiment import ROOT, check_splits, run, training_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--source-experiment", type=Path, default=ROOT / "results/experiments/sampling_full_20261005")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    checkpoint = args.source_experiment.resolve() / "fold_2/window/best.pt"
    samples = ROOT / "data/finetune/fold_2"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    check_splits(samples, 2)
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    affected_ids = {"1xlM62wnebrkHFOUcWspz0lzWDDzGCXJj", "1k74OEu0i80YMqpO9U8_JTJ7lJH8V9jZA"}
    if {r["drive_id"] for r in rows if r["drive_id"] in affected_ids} != affected_ids or any(
            r["split"] != "train" for r in rows if r["drive_id"] in affected_ids):
        raise ValueError("Expected numerically affected files exclusively in training for this fold")
    protocol = {"kind": "numeric_conservative_pilot_v1", "fold": 2, "epochs": 3, "steps_per_epoch": 2048,
                "learning_rate": 1e-6, "weight_decay": 1e-5, "seed": 2026, "val_samples_per_file": 2,
                "source_checkpoint": str(checkpoint), "repo_root": str(Path(args.repo_root).resolve()),
                "arms": ["legacy", "escape-v1"], "sampling": "window", "evaluate_initial": True,
                "evaluation_split": "val", "note": "exploratory pilot; full validation/new acquisition holdout required before promotion"}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume must preserve protocol")
        if (output / "COMPLETE").exists():
            print(f"[numeric] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use a new output directory or --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < 5 * 1024 ** 3:
            raise RuntimeError("Reserve at least 5 GiB free for two branches and checkpoint-save headroom")
    lock = None
    try:
        if not args.dry_run:
            lock = (output.parent / f".{output.name}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not args.resume:
                output.mkdir()
                (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        summaries = {}
        for arm in protocol["arms"]:
            target = output / arm
            epochs, last = training_plan(target, "window", protocol, samples) if args.resume else (3, None)
            if epochs:
                command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                           "--checkpoint", str(checkpoint), "--samples-dir", str(samples), "--output-dir", str(target),
                           "--epochs", str(epochs), "--steps-per-epoch", "2048", "--learning-rate", "1e-6",
                           "--weight-decay", "1e-5", "--seed", "2026", "--sampling", "window", "--device", "cuda",
                           "--normalization", arm, "--validation-details", "--val-samples-per-file", "2", "--evaluate-initial"]
                if last:
                    command += ["--resume", str(last)]
                run(command, output / f"{arm}.log", resume=args.resume, dry_run=args.dry_run)
            if not args.dry_run:
                if not (target / "best_macro.pt").is_file():
                    raise ValueError(f"Missing macro checkpoint: {target}")
                summary_path = target / "summary.json"
                if not summary_path.is_file():
                    import torch
                    saved = torch.load(target / "last.pt", map_location="cpu", weights_only=False)
                    state = saved["finetune"]
                    summaries[arm] = {"epochs_completed": saved["epoch"], "best_macro_rmse": state["best_macro_rmse"],
                                      "best_macro_epoch": state["best_macro_epoch"], "normalization": state["normalization"],
                                      "note": "Summary recovered from completed/early-stopped checkpoint after interruption"}
                else:
                    summaries[arm] = json.loads(summary_path.read_text())
        if not args.dry_run:
            (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            (output / "COMPLETE").write_text("Paired validation-only pilot complete. No automatic model promotion.\n")
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
