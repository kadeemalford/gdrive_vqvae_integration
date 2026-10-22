#!/usr/bin/env python3
"""Paired six-epoch representation pilot; common restored-space validation selector."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil

try:
    from .run_sampling_experiment import ROOT, check_splits, run
except ImportError:
    from run_sampling_experiment import ROOT, check_splits, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = ROOT / "results/experiments/preservation_long_20261007/full/best_macro.pt"
    samples = ROOT / "data/finetune/fold_1"
    if not (source.parent.parent / "COMPLETE").is_file() or not source.is_file():
        raise ValueError("Completed preservation experiment/checkpoint required")
    check_splits(samples, 1)
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if {r["family"] for r in rows if r["split"] == "val"} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    protocol = {"kind": "channel_representation_pilot_v1", "fold": 1, "arms": ["global", "channel"],
                "normalization": {"global": "escape-v1", "channel": "channel-escape-v1"},
                "source_checkpoint": str(source), "repo_root": str(Path(args.repo_root).resolve()),
                "epochs": 6, "steps_per_epoch": 2048, "learning_rate": 1e-5, "weight_decay": 1e-5, "seed": 2026,
                "val_samples_per_file": 2, "trainable": "full", "physical_validation": True,
                "initial_validation_expected_equal": False, "evaluate_initial": True, "patience": 7,
                "evaluation_split": "val", "selector": "macro relative RMSE after restoring both arms onto SAME global-normalized ordinary target",
                "note": "Same budget, no early stopping before six epochs; channel policy changes input AND per-cell weighting in training. Nominal index+side rate recorded."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol mismatch")
        if (output / "COMPLETE").exists():
            print(f"[channel] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use a new output directory or --resume")
    lock = None
    try:
        if args.dry_run:
            print(json.dumps(protocol, indent=2))
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(output.parent).free < 5 * 1024 ** 3:
                raise RuntimeError("At least 5 GiB free required; no old data/checkpoints removed")
            lock = (output.parent / f".{output.name}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not args.resume:
                output.mkdir()
                (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        for arm in protocol["arms"]:
            target = output / arm
            remaining, last = 6, None
            if args.resume and (target / "last.pt").is_file():
                import torch
                saved = torch.load(target / "last.pt", map_location="cpu", weights_only=False)
                state = saved["finetune"]
                if state["normalization"] != protocol["normalization"][arm] or not state.get("physical_validation"):
                    raise ValueError("Checkpoint representation/selector mismatch")
                if not 0 <= saved["epoch"] <= 6:
                    raise ValueError("Checkpoint outside epoch budget")
                remaining, last = 6 - saved["epoch"], target / "last.pt"
                del saved
            elif args.resume and target.exists() and any(target.iterdir()):
                raise ValueError("Nonempty branch without resumable checkpoint")
            if not remaining:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(source), "--samples-dir", str(samples), "--output-dir", str(target),
                       "--epochs", str(remaining), "--steps-per-epoch", "2048", "--learning-rate", "1e-5", "--weight-decay", "1e-5",
                       "--seed", "2026", "--device", "cuda", "--sampling", "window", "--normalization", protocol["normalization"][arm],
                       "--physical-validation", "--validation-details", "--val-samples-per-file", "2", "--evaluate-initial",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "7", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, dry_run=args.dry_run)
        if not args.dry_run:
            (output / "COMPLETE").write_text("Paired common-physical-target validation pilot complete; no automatic promotion.\n")
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
