#!/usr/bin/env python3
"""Paired temporal-rate pilot with unchanged spatial context and target samples."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil

try:
    from .latent_geometry import checkpoint_geometry, POLICIES
    from .run_sampling_experiment import ROOT, check_splits, run
except ImportError:
    from latent_geometry import checkpoint_geometry, POLICIES
    from run_sampling_experiment import ROOT, check_splits, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = ROOT / "results/experiments/channel_representation_pilot_20261008/global/best_macro.pt"
    smoke = ROOT / "results/experiments/temporal_geometry_smoke_20261008"
    if not source.is_file() or not (source.parent.parent / "COMPLETE").is_file():
        raise ValueError("Completed source channel experiment required")
    if not (smoke / "COMPLETE").is_file():
        raise ValueError("Native maximum-geometry forward/backward smoke must pass first")
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if {row["family"] for row in rows if row["split"] == "val"} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    protocol = {"kind": "temporal_latent_density_pilot_v1", "fold": 1, "arms": list(POLICIES),
                "source_checkpoint": str(source), "repo_root": str(Path(args.repo_root).resolve()),
                "normalization": "escape-v1", "physical_validation": True, "trainable": "full",
                "epochs": 6, "steps_per_epoch": 2048, "learning_rate": 1e-5, "weight_decay": 1e-5,
                "seed": 2026, "sampling": "window", "val_samples_per_file": 2,
                "evaluate_initial": True, "initial_validation_expected_equal": False,
                "evaluation_split": "val", "patience": 7,
                "train_windows": sum(row["split"] == "train" for row in rows),
                "full_val_windows": sum(row["split"] == "val" for row in rows),
                "selector": "macro restored-space relative RMSE; report actual index count + side information",
                "note": "Standard vs temporal stride 1 in last one/two encoder levels and matching first decoder levels. Same parameter shapes, no temporal decimation, fixed spatial stride. More bits AND changed temporal receptive field, not same-rate or pure-rate causal comparison."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[temporal-density] already complete: {output}")
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
                raise RuntimeError("At least 5 GiB free required; no old checkpoints removed")
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
                if checkpoint_geometry(saved) != arm or not saved["finetune"].get("physical_validation"):
                    raise ValueError("Checkpoint geometry/selector mismatch")
                if not 0 <= saved["epoch"] <= 6:
                    raise ValueError("Checkpoint outside budget")
                remaining, last = 6 - int(saved["epoch"]), target / "last.pt"
                del saved
            elif args.resume and target.exists() and any(target.iterdir()):
                raise ValueError("Nonempty branch without last checkpoint")
            if not remaining:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(source), "--samples-dir", str(samples), "--output-dir", str(target),
                       "--epochs", str(remaining), "--steps-per-epoch", "2048", "--learning-rate", "1e-5", "--weight-decay", "1e-5",
                       "--seed", "2026", "--device", "cuda", "--sampling", "window", "--normalization", "escape-v1",
                       "--latent-geometry", arm, "--physical-validation", "--validation-details", "--val-samples-per-file", "2",
                       "--evaluate-initial", "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "7", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, dry_run=args.dry_run)
        if not args.dry_run:
            (output / "COMPLETE").write_text("Temporal density pilot complete; no automatic promotion or test access.\n")
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
