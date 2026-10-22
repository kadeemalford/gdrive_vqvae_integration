#!/usr/bin/env python3
"""Fold-0/seed-2027 geometry confirmation, never initialize from fold-1 FEBUS weights."""
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


def verify_source_fold(checkpoint, samples):
    source_samples = checkpoint.get("finetune", {}).get("samples_dir")
    if not source_samples or (ROOT / source_samples).resolve() != samples.resolve():
        raise ValueError("Source was not trained on the requested fold; potential leakage")
    return int(checkpoint["epoch"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    previous = ROOT / "results/experiments/rate_allocation_pilot_20261012"
    if not (previous / "COMPLETE").is_file():
        raise ValueError("Completed allocation comparison required")
    source = ROOT / "results/experiments/sampling_full_20261005/fold_0/window/best.pt"
    samples = ROOT / "data/finetune/fold_0"
    saved = torch.load(source, map_location="cpu", weights_only=False)
    source_epoch = verify_source_fold(saved, samples)
    del saved
    manifest = samples / "samples_manifest.csv"
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    # Check provenance before restoration; test arrays may be restored for integrity
    # by the archived-folder tool, but are NEVER scored or used for updates here.
    groups = {}
    for row in rows:
        key = row["family"], row["configuration"]
        groups.setdefault(key, set()).add(row["split"])
        if (int(row["fold"]) == 0) != (row["split"] == "test"):
            raise ValueError("Fold provenance mismatch")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Configuration crosses split boundaries")
    if {r["family"] for r in rows if r["split"] == "val"} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    arms = {"temporal": "temporal-4x", "balanced_grid": "balanced-2x", "spatial": "spatial-4x"}
    code = [ROOT / f"finetuning/{name}.py" for name in ("run_rate_allocation_confirmation", "restore_pruned_samples",
            "finetune_vqvae", "model_factory", "latent_geometry", "physical_validation", "reconstruction_objective",
            "sampling", "codebook_refresh", "domain_routing", "evaluate_physical_candidates",
            "summarize_numeric_adaptation", "reversible_normalization")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "spatiotemporal_allocation_fold_confirmation_v1", "fold": 0, "seed": 2027,
                "arms": list(arms), "geometries": arms, "source": str(source), "source_sha256": digest(source),
                "source_epoch": source_epoch, "manifest_sha256": digest(manifest),
                "repo_root": str(Path(args.repo_root).resolve()), "epochs": 24, "steps_per_epoch": 4096,
                "learning_rate": 3e-5, "weight_decay": 1e-5, "patience": 25, "min_delta": 1e-4,
                "sampling": "window", "normalization": "escape-v1", "trainable": "full", "physical_validation": True,
                "evaluate_initial": True, "initial_validation_expected_equal": False, "val_samples_per_file": 2,
                "codebook_refresh": "none", "evaluation_split": "val",
                "train_windows": sum(r["split"] == "train" for r in rows), "val_windows": sum(r["split"] == "val" for r in rows),
                "code_sha256": {str(p): digest(p) for p in code},
                "note": "Different fold/configurations and data-order seed; common fold-0-only source, fresh AdamW. Not identical warm-start quality to fold 1; compare within-fold contrasts. Previously inspected development corpus, not nested CV/external audit. Restoration includes archived test files for folder integrity only, never model test access."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[allocation confirmation] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use new output directory or --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Approximately 9 GiB restored arrays plus checkpoints/logs; retain safety margin.
    marker = samples / "PRUNED_ARRAYS.json"
    needs_restore = marker.is_file() and json.loads(marker.read_text()).get("status") != "restored"
    if shutil.disk_usage(output.parent).free < (18 if needs_restore else 10) * 1024 ** 3:
        raise RuntimeError("Insufficient restoration/training disk headroom; no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        if needs_restore:
            run(["finetuning/restore_pruned_samples.py", "--samples-dir", str(samples)], output / "restoration.log", resume=args.resume, quiet=True)
        check_splits(samples, 0)
        if any(not (ROOT / r["sample_path"]).is_file() for r in rows if r["split"] in ("train", "val")):
            raise FileNotFoundError("Incomplete restored train/val arrays")
        for arm, geometry in arms.items():
            target = output / arm
            epochs, last = remaining_epochs(target, protocol, geometry) if args.resume else (24, None)
            if not epochs:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root, "--checkpoint", str(source),
                       "--samples-dir", str(samples), "--output-dir", str(target), "--epochs", str(epochs),
                       "--steps-per-epoch", "4096", "--learning-rate", "3e-5", "--weight-decay", "1e-5",
                       "--seed", "2027", "--device", "cuda", "--sampling", "window", "--normalization", "escape-v1",
                       "--latent-geometry", geometry, "--physical-validation", "--validation-details", "--evaluate-initial",
                       "--val-samples-per-file", "2", "--codebook-diagnostics", "--channel-diagnostics",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "25", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, quiet=True)
        audit = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--fold", "0", "--output-dir", str(audit)]
        for arm in arms:
            command += ["--candidate", f"{arm}=escape-v1={output / arm / 'best_macro.pt'}"]
        if audit.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("Fold-0 allocation confirmation and full val complete; no test score or promotion.\n")
        run(["finetuning/summarize_numeric_adaptation.py", "--experiment-dir", str(output)], output / "comparison.log", quiet=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
