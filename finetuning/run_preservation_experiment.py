#!/usr/bin/env python3
"""Long paired full-corpus adaptation, full four-family validation, no test inference."""
import argparse
import csv
import fcntl
import json
import shutil
from pathlib import Path

try:
    from .run_sampling_experiment import ROOT, check_splits, run
except ImportError:
    from run_sampling_experiment import ROOT, check_splits, run


def remaining(target, protocol):
    import torch
    last = target / "last.pt"
    if not last.exists():
        if target.exists() and any(target.iterdir()):
            raise ValueError("Nonempty branch without last checkpoint")
        return protocol["epochs"], None
    saved = torch.load(last, map_location="cpu", weights_only=False)
    state = saved["finetune"]
    if state.get("trainable") != target.name or state.get("early_stopping_metric") != "macro-rmse":
        raise ValueError("Checkpoint branch/stopping policy mismatch")
    epoch = int(saved["epoch"])
    if not 0 <= epoch <= protocol["epochs"]:
        raise ValueError("Checkpoint outside budget")
    for name in ("best.pt", "best_macro.pt"):
        if not (target / name).is_file():
            raise ValueError(f"Missing checkpoint: {target / name}")
    finished = (target / "summary.json").exists() or state["epochs_without_improvement"] >= protocol["patience"]
    return (0 if finished else protocol["epochs"] - epoch), last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, default=ROOT / "results/experiments/sampling_full_20261005")
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = args.source_experiment.resolve() / "fold_1/window/best.pt"
    samples = ROOT / "data/finetune/fold_1"
    if not source.is_file():
        raise FileNotFoundError(source)
    check_splits(samples, 1)
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    families = {r["family"] for r in rows if r["split"] == "val"}
    if families != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Full four-family validation required")
    protocol = {"kind": "preservation_full_corpus_v1", "fold": 1, "arms": ["full", "decoder"],
                "source_checkpoint": str(source), "repo_root": str(Path(args.repo_root).resolve()),
                "epochs": 8, "steps_per_epoch": 0, "learning_rate": 1e-6, "weight_decay": 1e-5,
                "seed": 2026, "normalization": "escape-v1", "patience": 3, "min_delta": 1e-4,
                "early_stopping_metric": "macro-rmse", "evaluate_initial": True, "val_samples_per_file": 0,
                "train_windows": sum(r["split"] == "train" for r in rows),
                "val_windows": sum(r["split"] == "val" for r in rows), "evaluation_split": "val",
                "note": "Exploratory reserved-configuration validation; no final model or automatic promotion"}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol mismatch")
        if (output / "COMPLETE").exists():
            print(f"[preservation] already complete: {output}")
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
                raise RuntimeError("At least 5 GiB free required; no old files will be removed")
            lock = (output.parent / f".{output.name}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not args.resume:
                output.mkdir()
                (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        for arm in protocol["arms"]:
            target = output / arm
            epochs, last = remaining(target, protocol) if args.resume else (protocol["epochs"], None)
            if not epochs:
                print(f"[preservation] skip completed branch: {arm}")
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(source), "--samples-dir", str(samples), "--output-dir", str(target),
                       "--epochs", str(epochs), "--learning-rate", "1e-6", "--weight-decay", "1e-5",
                       "--seed", "2026", "--sampling", "window", "--device", "cuda",
                       "--normalization", "escape-v1", "--trainable", arm, "--validation-details", "--evaluate-initial",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "3", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, dry_run=args.dry_run)
        if not args.dry_run:
            (output / "COMPLETE").write_text("Paired training finished; full validation only. Evaluate per family before any promotion.\n")
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
