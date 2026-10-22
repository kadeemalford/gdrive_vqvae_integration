#!/usr/bin/env python3
"""Paired long temporal-codebook ablation with a fixed source and frozen reference."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil

import torch

try:
    from .run_sampling_experiment import ROOT, check_splits, run
    from .run_temporal_density_long import digest, require_cuda, remaining_epochs
except ImportError:
    from run_sampling_experiment import ROOT, check_splits, run
    from run_temporal_density_long import digest, require_cuda, remaining_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    previous = ROOT / "results/experiments/scratch_budget_continuation_20261011"
    if not (previous / "COMPLETE").is_file():
        raise ValueError("Completed scratch budget comparison required")
    source = previous / "scratch_temporal/best_macro.pt"
    smoke = ROOT / "results/experiments/codebook_refresh_smoke_20261011"
    diagnostic = ROOT / "results/experiments/quantization_gap_20261011"
    if not (smoke / "COMPLETE").is_file() or not (diagnostic / "COMPLETE").is_file():
        raise ValueError("Completed quantization diagnostic and refresh GPU smoke required")
    if json.loads((smoke / "summary.json").read_text())["source_sha256"] != digest(source):
        raise ValueError("Refresh smoke/source mismatch")
    reference = ROOT / "results/experiments/architecture_initialization_pilot_20261010/transferred_4x/best_macro.pt"
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if any(not (ROOT / r["sample_path"]).is_file() for r in rows if r["split"] in ("train", "val")):
        raise FileNotFoundError("Complete prepared dataset required")
    for split in ("train", "val"):
        if {r["family"] for r in rows if r["split"] == split} != {"doped", "multicore", "multimode", "single_mode"}:
            raise ValueError("Four-family corpus required")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    if saved["config"].get("architecture") != "temporal-dilated-v1" or saved["epoch"] != 31:
        raise ValueError("Unexpected selected temporal source")
    del saved
    code = [ROOT / f"finetuning/{name}.py" for name in ("run_codebook_refresh_experiment", "finetune_vqvae",
            "codebook_refresh", "model_factory", "latent_geometry", "reconstruction_objective", "physical_validation",
            "sampling", "reversible_normalization", "evaluate_physical_candidates", "summarize_numeric_adaptation")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "paired_codebook_refresh_v1", "fold": 1, "arms": ["control", "refresh"],
                "source": str(source), "source_sha256": digest(source), "source_epoch": 31,
                "reference": str(reference), "reference_sha256": digest(reference),
                "smoke_sha256": digest(smoke / "summary.json"),
                "diagnostic_protocol_sha256": digest(diagnostic / "protocol.json"),
                "repo_root": str(Path(args.repo_root).resolve()), "seed": 2026, "epochs": 24, "steps_per_epoch": 4096,
                "learning_rate": 3e-5, "weight_decay": 1e-5, "optimizer": "fresh AdamW in both arms",
                "patience": 25, "min_delta": 1e-4, "sampling": "window", "trainable": "full",
                "normalization": "escape-v1", "latent_geometry": "temporal-4x", "architecture": "temporal-dilated-v1",
                "physical_validation": True, "evaluate_initial": True, "initial_validation_expected_equal": True,
                "codebook_diagnostics": True, "channel_diagnostics": True, "val_samples_per_file": 2,
                "evaluation_split": "val", "manifest_sha256": digest(samples / "samples_manifest.csv"),
                "refresh_policy": {"age_epochs": 3, "cap_per_epoch": 32, "sample_every_train_steps": 64,
                                   "vectors_per_sample_max": 128, "reservoir_samples_max": 64,
                                   "replacement": "deterministic random training encoder latents; reset replaced Adam moments"},
                "code_sha256": {str(p): digest(p) for p in code},
                "note": "Common selected source and fresh optimizer/LR 3e-5. Only refresh differs; 98,304 additional updates per arm, no early stop. Control gains do not isolate optimizer reset vs LR/exposure; refresh/control contrast isolates the refresh intervention. Frozen transferred comparison has unequal compute history. No new dataset, loss, test or promotion."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[codebook refresh] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use new directory or --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 8 * 1024 ** 3:
        raise RuntimeError("8 GiB free required; no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        for arm in protocol["arms"]:
            target = output / arm
            epochs, last = remaining_epochs(target, protocol, "temporal-4x") if args.resume else (24, None)
            if not epochs:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root, "--checkpoint", str(source),
                       "--samples-dir", str(samples), "--output-dir", str(target), "--epochs", str(epochs),
                       "--steps-per-epoch", "4096", "--learning-rate", "3e-5", "--weight-decay", "1e-5",
                       "--seed", "2026", "--device", "cuda", "--sampling", "window", "--normalization", "escape-v1",
                       "--latent-geometry", "temporal-4x", "--physical-validation", "--validation-details",
                       "--evaluate-initial", "--val-samples-per-file", "2", "--codebook-diagnostics", "--channel-diagnostics",
                       "--codebook-refresh", "dead-v1" if arm == "refresh" else "none",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "25", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, quiet=True)
        audit = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--output-dir", str(audit),
                   "--candidate", f"frozen_transferred=escape-v1={reference}", "--candidate", f"source_temporal=escape-v1={source}"]
        for arm in protocol["arms"]:
            command += ["--candidate", f"{arm}=escape-v1={output / arm / 'best_macro.pt'}"]
        if audit.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("Paired codebook refresh/full val complete, source/reference retained, no promotion.\n")
        run(["finetuning/summarize_numeric_adaptation.py", "--experiment-dir", str(output)], output / "comparison.log", quiet=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
