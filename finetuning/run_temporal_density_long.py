#!/usr/bin/env python3
"""Full-corpus continuation of promising temporal 2x/4x candidates; val only."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

try:
    from .latent_geometry import checkpoint_geometry
    from .run_sampling_experiment import ROOT, check_splits, run
except ImportError:
    from latent_geometry import checkpoint_geometry
    from run_sampling_experiment import ROOT, check_splits, run


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check nvidia-smi and the installed/loaded NVIDIA driver versions before resuming; checkpoints are unchanged and CPU fallback is not automatic.")


def remaining_epochs(target, protocol, arm):
    import torch
    last = target / "last.pt"
    if not last.is_file():
        if target.exists() and any(target.iterdir()):
            raise ValueError("Nonempty branch without last checkpoint")
        return protocol["epochs"], None
    saved = torch.load(last, map_location="cpu", weights_only=False)
    state = saved["finetune"]
    if (checkpoint_geometry(saved) != arm or not state.get("physical_validation")
            or state.get("normalization") != protocol.get("normalization", "escape-v1") or state.get("early_stopping_metric") != "macro-rmse"):
        raise ValueError("Checkpoint geometry/representation/selector mismatch")
    epoch = int(saved["epoch"])
    if not 0 <= epoch <= protocol["epochs"]:
        raise ValueError("Checkpoint outside budget")
    for name in ("best.pt", "best_macro.pt"):
        if not (target / name).is_file():
            raise ValueError(f"Missing selected checkpoint: {target / name}")
    completed = epoch == protocol["epochs"] or state["epochs_without_improvement"] >= protocol["patience"]
    summary = target / "summary.json"
    if summary.is_file():
        if json.loads(summary.read_text())["epochs_completed"] != epoch:
            raise ValueError("Summary/checkpoint epoch mismatch")
    elif completed:
        # A restart can happen after atomic last.pt but before summary.json.
        # Recover only scalar selection/provenance fields; do not claim new metrics.
        recovered = {key: state[key] for key in ("best_epoch", "best_macro_epoch", "best_macro_rmse", "validation_hash",
                    "normalization", "physical_validation", "latent_geometry", "evaluate_initial", "trainable", "early_stopping_metric") if key in state}
        recovered.update(epochs_completed=epoch, summary_recovered_from_completed_last=True)
        temporary = target / "summary.json.recovered.tmp"
        temporary.write_text(json.dumps(recovered, indent=2) + "\n")
        temporary.replace(summary)
    return (0 if completed else protocol["epochs"] - epoch), last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    pilot = ROOT / "results/experiments/temporal_density_pilot_20261008"
    full_validation = ROOT / "results/experiments/temporal_density_full_validation_20261008"
    if not (pilot / "COMPLETE").is_file() or not (full_validation / "COMPLETE").is_file():
        raise ValueError("Pilot and full candidate validation must be complete")
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if {row["family"] for row in rows if row["split"] == "val"} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    # Do not silently run on a partial prepared dataset after manual pruning.
    if any(not (ROOT / row["sample_path"]).is_file() for row in rows if row["split"] in ("train", "val")):
        raise FileNotFoundError("Complete fold_1 train/validation arrays required")
    arms = ["temporal-2x", "temporal-4x"]
    sources = {arm: pilot / arm / "best_macro.pt" for arm in arms}
    protocol = {"kind": "temporal_density_full_corpus_continuation_v1", "fold": 1, "arms": arms,
                "sources": {arm: {"checkpoint": str(path), "sha256": digest(path)} for arm, path in sources.items()},
                "repo_root": str(Path(args.repo_root).resolve()), "epochs": 8, "steps_per_epoch": 0,
                "learning_rate": 3e-6, "weight_decay": 1e-5, "seed": 2026, "patience": 3, "min_delta": 1e-4,
                "normalization": "escape-v1", "physical_validation": True, "trainable": "full", "sampling": "window",
                "val_samples_per_file": 0, "evaluate_initial": True, "initial_validation_expected_equal": False,
                "train_windows": sum(row["split"] == "train" for row in rows),
                "val_windows": sum(row["split"] == "val" for row in rows), "evaluation_split": "val",
                "note": "Each arm starts from its own selected six-epoch pilot, fresh AdamW; same full-data order/max epoch budget, different bitrate and compute. No test selection or final-model promotion."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source checksum mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[density-long] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use a new output directory or --resume")
    lock = None
    try:
        if args.dry_run:
            print(json.dumps(protocol, indent=2))
        else:
            require_cuda()
            output.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(output.parent).free < 5 * 1024 ** 3:
                raise RuntimeError("At least 5 GiB free required; no automatic deletions")
            lock = (output.parent / f".{output.name}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not args.resume:
                output.mkdir()
                (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        for arm in arms:
            target = output / arm
            epochs, last = remaining_epochs(target, protocol, arm) if args.resume else (8, None)
            if not epochs:
                print(f"[density-long] skip finished branch: {arm}")
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(sources[arm]), "--samples-dir", str(samples), "--output-dir", str(target),
                       "--epochs", str(epochs), "--learning-rate", "3e-6", "--weight-decay", "1e-5",
                       "--seed", "2026", "--device", "cuda", "--sampling", "window", "--normalization", "escape-v1",
                       "--latent-geometry", arm, "--physical-validation", "--validation-details", "--evaluate-initial",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "3", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, dry_run=args.dry_run)
        if not args.dry_run:
            (output / "COMPLETE").write_text("Full-corpus temporal-density continuation complete; val only, no automatic promotion.\n")
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
