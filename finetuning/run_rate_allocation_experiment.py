#!/usr/bin/env python3
"""Compare temporal, balanced and spatial allocation at equal/nearly equal VQ rates."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil

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
    previous = ROOT / "results/experiments/domain_specialization_pilot_20261012"
    if not (previous / "COMPLETE").is_file():
        raise ValueError("Completed specialization comparison required")
    source = ROOT / "results/experiments/architecture_initialization_pilot_20261010/transferred_4x/best_macro.pt"
    smoke = ROOT / "results/experiments/rate_allocation_smoke_20261012"
    if not (smoke / "COMPLETE").is_file() or json.loads((smoke / "summary.json").read_text())["source_sha256"] != digest(source):
        raise ValueError("Maximum-shape source-matched smoke required")
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if any(not (ROOT / r["sample_path"]).is_file() for r in rows if r["split"] in ("train", "val")):
        raise FileNotFoundError("Complete corpus required")
    if {r["family"] for r in rows if r["split"] == "val"} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    arms = {"temporal": "temporal-4x", "balanced_grid": "balanced-2x", "spatial": "spatial-4x"}
    code = [ROOT / f"finetuning/{name}.py" for name in ("run_rate_allocation_experiment", "finetune_vqvae",
            "model_factory", "latent_geometry", "physical_validation", "reconstruction_objective", "sampling",
            "codebook_refresh", "domain_routing", "evaluate_physical_candidates", "summarize_numeric_adaptation",
            "reversible_normalization")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "spatiotemporal_rate_allocation_v1", "fold": 1, "arms": list(arms), "geometries": arms,
                "source": str(source), "source_sha256": digest(source), "source_epoch": 13,
                "repo_root": str(Path(args.repo_root).resolve()), "manifest_sha256": digest(samples / "samples_manifest.csv"),
                "smoke_sha256": digest(smoke / "summary.json"), "seed": 2026, "epochs": 24,
                "steps_per_epoch": 4096, "learning_rate": 3e-5, "weight_decay": 1e-5, "patience": 25, "min_delta": 1e-4,
                "sampling": "window", "normalization": "escape-v1", "trainable": "full", "physical_validation": True,
                "evaluate_initial": True, "initial_validation_expected_equal": False, "val_samples_per_file": 2,
                "codebook_refresh": "none", "codebook_diagnostics": True, "channel_diagnostics": True,
                "evaluation_split": "val", "code_sha256": {str(p): digest(p) for p in code},
                "note": "Same weights, fresh AdamW/LR, raw targets, data order, parameter count and 98,304 updates per arm. Balanced grid equals temporal index counts for all shapes; spatial grid uses 0.8% more indices at T=1000. Spatial axes add 8 zero rows for L=4 solely to compensate stride-1 shrink, decoder crops them. Allocation, receptive fields and boundary context change; no claim of equal receptive fields, convergence or promotion."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[rate allocation] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use new directory or --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 10 * 1024 ** 3:
        raise RuntimeError("10 GiB free required; no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        for arm, geometry in arms.items():
            target = output / arm
            epochs, last = remaining_epochs(target, protocol, geometry) if args.resume else (24, None)
            if not epochs:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root, "--checkpoint", str(source),
                       "--samples-dir", str(samples), "--output-dir", str(target), "--epochs", str(epochs),
                       "--steps-per-epoch", "4096", "--learning-rate", "3e-5", "--weight-decay", "1e-5",
                       "--seed", "2026", "--device", "cuda", "--sampling", "window", "--normalization", "escape-v1",
                       "--latent-geometry", geometry, "--physical-validation", "--validation-details", "--evaluate-initial",
                       "--val-samples-per-file", "2", "--codebook-diagnostics", "--channel-diagnostics",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "25", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, quiet=True)
        audit = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--output-dir", str(audit),
                   "--candidate", f"frozen_generalist=escape-v1={source}"]
        for arm in arms:
            command += ["--candidate", f"{arm}=escape-v1={output / arm / 'best_macro.pt'}"]
        if audit.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("Rate-allocation comparison/full validation complete, no promotion.\n")
        run(["finetuning/summarize_numeric_adaptation.py", "--experiment-dir", str(output)], output / "comparison.log", quiet=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
