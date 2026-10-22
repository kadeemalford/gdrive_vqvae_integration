#!/usr/bin/env python3
"""Paired temporal/spectral branches of uniform checkpoints; validation only."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

try:
    from .run_sampling_experiment import ROOT, check_splits, run, training_plan
except ImportError:
    from run_sampling_experiment import ROOT, check_splits, run, training_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, default=Path("results/experiments/sampling_full_20261005"))
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--pilot", action="store_true", help="3 epochs x 256 updates, two validation windows per file. Without this flag: 10 full epochs and full validation.")
    parser.add_argument("--evaluate-initial", action="store_true", help="Include branch initialization as epoch 0 checkpoint candidate; recommended for warm starts.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = args.source_experiment.resolve()
    if not (source / "COMPLETE").exists():
        raise ValueError("Source sampling experiment must be complete")
    protocol = {"kind": "spectral", "pilot": args.pilot, "seed": 2026,
                "source_experiment": str(source), "initialization": "same fold uniform best.pt; fresh AdamW in both branches",
                "epochs": 3 if args.pilot else 10, "steps_per_epoch": 256 if args.pilot else 0,
                "val_samples_per_file": 2 if args.pilot else 0, "learning_rate": 3e-6,
                "arms": {"temporal": 0.0, "spectral": 0.1}, "sampling": "window",
                "validation_details": True, "evaluate_initial": args.evaluate_initial, "spectral_fft_sizes": [128, 256, 512],
                "evaluation_split": "validation", "selectors": ["temporal+VQ val_loss", "family/config macro rRMSE"]}
    if args.resume:
        saved_protocol = json.loads((output / "protocol.json").read_text())
        saved_protocol.setdefault("evaluate_initial", False)
        if saved_protocol != protocol:
            raise ValueError("Resume protocol differs; retain original --pilot and source experiment")
        if (output / "COMPLETE").exists():
            print(f"[spectral] already complete: {output / 'REPORT.md'}", flush=True)
            return
    elif output.exists():
        raise FileExistsError(f"{output} exists; use --resume")
    lock = None
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        lock = (output.parent / f".{output.name}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another orchestrator is running this experiment") from None
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for fold in range(4):
        samples = Path(f"data/finetune/fold_{fold}")
        check_splits(ROOT / samples, fold)
        folder = output / f"fold_{fold}"
        if not args.dry_run:
            folder.mkdir(exist_ok=args.resume)
        for index, (arm, weight) in enumerate(protocol["arms"].items()):
            print(f"\n[spectral {fold * 2 + index + 1}/9] fold {fold}, {arm}, weight={weight}", flush=True)
            target = folder / arm
            epochs, checkpoint = training_plan(target, "window", protocol, samples) if args.resume else (protocol["epochs"], None)
            if not epochs:
                if not (target / "best_macro.pt").exists():
                    raise ValueError(f"Completed training without macro checkpoint: {target}")
                print(f"[spectral] skip completed training: {target}", flush=True)
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(source / f"fold_{fold}/window/best.pt"),
                       "--samples-dir", str(samples), "--output-dir", str(target),
                       "--epochs", str(epochs), "--steps-per-epoch", str(protocol["steps_per_epoch"]),
                       "--learning-rate", str(protocol["learning_rate"]), "--spectral-weight", str(weight),
                       "--validation-details", "--val-samples-per-file", str(protocol["val_samples_per_file"]),
                       "--seed", str(protocol["seed"]), "--sampling", "window", "--device", "cuda"]
            if checkpoint:
                command += ["--resume", str(checkpoint)]
            if args.evaluate_initial:
                command += ["--evaluate-initial"]
            run(command, folder / f"{arm}.log", resume=args.resume, dry_run=args.dry_run, quiet=args.quiet)
    print("\n[spectral 9/9] validation comparison", flush=True)
    run(["finetuning/summarize_spectral_experiment.py", "--experiment-dir", str(output)],
        output / "report.log", resume=args.resume, dry_run=args.dry_run, quiet=args.quiet)
    if not args.dry_run:
        (output / "COMPLETE").write_text("Both branches completed; validation-only comparison generated.\n")
    if lock:
        lock.close()
    print(f"[spectral] {'plan checked' if args.dry_run else 'complete'}: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
