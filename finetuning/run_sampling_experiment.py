#!/usr/bin/env python3
"""Paired, isolated CV runs. Pilot evaluates validation only, full evaluates test."""
from __future__ import annotations

import argparse
import codecs
import csv
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], log: Path, *, resume: bool = False, dry_run: bool = False, quiet: bool = False) -> None:
    print(f"[experiment] {log}", flush=True)
    if dry_run:
        print("[dry-run] " + " ".join([sys.executable, *command]), flush=True)
        return
    with log.open("ab" if resume else "wb") as stream:
        if resume:
            stream.write(b"\n[experiment] resumed invocation\n")
            stream.flush()
        # Read raw chunks rather than lines: tqdm redraws with carriage returns.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        process = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        try:
            while chunk := process.stdout.read(8192):
                stream.write(chunk)
                stream.flush()
                if not quiet:
                    sys.stdout.write(decoder.decode(chunk))
                    sys.stdout.flush()
            if not quiet:
                sys.stdout.write(decoder.decode(b"", final=True))
                sys.stdout.flush()
            status = process.wait()
        except BaseException:
            # Do not leave a trainer running after its orchestrator is interrupted.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
        if status:
            raise subprocess.CalledProcessError(status, process.args)


def training_plan(target: Path, mode: str, protocol: dict, samples: Path) -> tuple[int, Path | None]:
    """Return remaining epochs and checkpoint; zero means training already finished."""
    last = target / "last.pt"
    if not last.exists():
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Nonempty training directory without last.pt: {target}")
        return protocol["epochs"], None
    import torch
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    state = checkpoint["finetune"]
    if state.get("sampling", "window") != mode:
        raise ValueError(f"Checkpoint sampling mismatch: {last}")
    if (ROOT / state["samples_dir"]).resolve() != (ROOT / samples).resolve():
        raise ValueError(f"Checkpoint sample directory mismatch: {last}")
    epoch = int(checkpoint["epoch"])
    if epoch < 0 or epoch > protocol["epochs"]:
        raise ValueError(f"Checkpoint epoch outside protocol budget: {last}")
    if not (target / "best.pt").exists():
        raise ValueError(f"Missing best.pt beside {last}; refusing an incomplete checkpoint pair")
    summary = target / "summary.json"
    if summary.exists():
        saved = json.loads(summary.read_text())
        if saved["epochs_completed"] != epoch:
            raise ValueError(f"Summary/checkpoint epoch mismatch: {target}")
        return 0, last
    # The trainer can save last.pt and stop before writing summary.json.
    stopped = state.get("epochs_without_improvement", 0) >= 3 and state["best_epoch"] != epoch
    return (0 if stopped else protocol["epochs"] - epoch), last


def check_splits(directory: Path, test_fold: int) -> None:
    marker = directory / "PRUNED_ARRAYS.json"
    if marker.is_file() and json.loads(marker.read_text()).get("status") != "restored":
        raise ValueError(f"Prepared arrays were pruned/incomplete: {directory}; use restore_pruned_samples.py before this run")
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
    parser.add_argument("--resume", action="store_true", help="Continue an existing experiment, preserving logs and total epoch budgets.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect the plan without training or modifying experiment files.")
    parser.add_argument("--quiet", action="store_true", help="Only print stage summaries; detailed progress is still saved in logs.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protocol = {"pilot": args.pilot, "seed": 2026, "epochs": 2 if args.pilot else 10,
                "steps_per_epoch": 128 if args.pilot else 0, "sampling": ["window", "family-group"],
                "evaluation_split": "val" if args.pilot else "test", "selection": "minimum window-weighted val_loss",
                "lr": 1e-5, "base_checkpoint": "models/febus_finetune_base/best.pt"}
    if args.resume:
        saved_protocol = json.loads((output / "protocol.json").read_text())
        if saved_protocol != protocol:
            raise ValueError("Resume protocol differs from saved protocol; preserve --pilot and original settings")
        if (output / "COMPLETE").exists():
            print(f"[experiment] already complete: {output / 'REPORT.md'}", flush=True)
            return
    elif output.exists():
        raise FileExistsError(f"Experiment already exists: {output}. Use --resume to continue.")
    # Hold an advisory process lock for the entire sequence. OS releases it on exit.
    lock = None
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        lock = (output.parent / f".{output.name}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError("Another orchestrator is already running this experiment") from None
        if not args.resume:
            output.mkdir(parents=True, exist_ok=False)
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for fold in range(4):
        samples = Path(f"data/finetune/fold_{fold}")
        check_splits(ROOT / samples, fold)
        fold_dir = output / f"fold_{fold}"
        if not args.dry_run:
            fold_dir.mkdir(exist_ok=args.resume)
        for mode in protocol["sampling"]:
            step = fold * 3 + protocol["sampling"].index(mode) + 1
            print(f"\n[experiment {step}/13] Fold {fold} — training {mode}", flush=True)
            target = fold_dir / mode
            epochs, checkpoint = training_plan(target, mode, protocol, samples) if args.resume else (protocol["epochs"], None)
            if not epochs:
                print(f"[experiment] skip completed training: {target}", flush=True)
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", protocol["base_checkpoint"], "--samples-dir", str(samples),
                       "--output-dir", str(target), "--sampling", mode, "--epochs", str(epochs),
                       "--steps-per-epoch", str(protocol["steps_per_epoch"]), "--device", "cuda"]
            if args.pilot:
                command += ["--max-val-samples", "128"]
            if checkpoint:
                command += ["--resume", str(checkpoint)]
                print(f"[experiment] resume {target}: {epochs} additional epoch(s)", flush=True)
            run(command, fold_dir / f"{mode}.log", resume=args.resume, dry_run=args.dry_run, quiet=args.quiet)
            if not args.dry_run:
                print(f"\n[experiment {step}/13] Training completed: fold {fold}, {mode}", flush=True)
        print(f"\n[experiment {fold * 3 + 3}/13] Fold {fold} — evaluation", flush=True)
        evaluation_complete = fold_dir / "evaluation/COMPLETE"
        if args.resume and evaluation_complete.exists():
            print(f"[experiment] skip completed evaluation: {fold_dir}", flush=True)
            continue
        command = ["finetuning/evaluate_checkpoints.py", "--repo-root", args.repo_root,
                   "--samples-dir", str(samples / protocol["evaluation_split"]),
                   "--output-dir", str(fold_dir / "evaluation"), "--device", "cuda"]
        for mode in protocol["sampling"]:
            command += ["--checkpoint", f"{mode}={fold_dir / mode / 'best.pt'}"]
        if args.pilot:
            command += ["--manifest", str(samples / "samples_manifest.csv"), "--max-samples-per-group", "16"]
        run(command, fold_dir / "evaluation.log", resume=args.resume, dry_run=args.dry_run, quiet=args.quiet)
        if not args.dry_run:
            evaluation_complete.write_text("Evaluation completed successfully.\n")
    print("\n[experiment 13/13] Comparative report", flush=True)
    run(["finetuning/summarize_sampling_experiment.py", "--experiment-dir", str(output)], output / "report.log", resume=args.resume, dry_run=args.dry_run, quiet=args.quiet)
    if not args.dry_run:
        (output / "COMPLETE").write_text("Training and evaluation completed for all four folds.\n")
    if lock:
        lock.close()
    print(f"[experiment] {'plan checked' if args.dry_run else 'complete'}: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
