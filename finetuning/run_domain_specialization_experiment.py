#!/usr/bin/env python3
"""Four metadata-routed family specialists, shared training control and frozen anchor."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil

try:
    from .domain_routing import FAMILIES, family_paths
    from .run_sampling_experiment import ROOT, check_splits, run
    from .run_temporal_density_long import digest, require_cuda, remaining_epochs
except ImportError:
    from domain_routing import FAMILIES, family_paths
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
    previous = ROOT / "results/experiments/codebook_refresh_long_20261011"
    if not (previous / "COMPLETE").is_file():
        raise ValueError("Finish quantization comparison before specialization")
    source = ROOT / "results/experiments/architecture_initialization_pilot_20261010/transferred_4x/best_macro.pt"
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    manifest = samples / "samples_manifest.csv"
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    if any(not (ROOT / r["sample_path"]).is_file() for r in rows if r["split"] in ("train", "val")):
        raise FileNotFoundError("Complete corpus required")
    counts = {}
    for family in FAMILIES:
        counts[family] = {split: len(family_paths(sorted((samples / split).glob("*.npz")), manifest, family, split)) for split in ("train", "val")}
    arms = {"shared": None, "balanced": None, **{f"expert_{f}": f for f in FAMILIES}}
    budgets = {arm: {"steps": 4096 if arm == "balanced" else 1024,
                     "sampling": "family-window" if arm == "balanced" else "window"} for arm in arms}
    code = [ROOT / f"finetuning/{name}.py" for name in ("run_domain_specialization_experiment", "domain_routing",
            "summarize_domain_routing", "finetune_vqvae", "codebook_refresh", "model_factory", "latent_geometry",
            "physical_validation", "reconstruction_objective", "reversible_normalization", "sampling", "evaluate_physical_candidates")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "metadata_domain_specialization_v1", "fold": 1, "arms": arms,
                "branch_budgets": budgets,
                "source": str(source), "source_sha256": digest(source), "source_epoch": 13,
                "repo_root": str(Path(args.repo_root).resolve()), "manifest_sha256": digest(manifest), "family_windows": counts,
                "seed": 2026, "epochs": 16, "steps_per_epoch": 1024, "learning_rate": 3e-5, "weight_decay": 1e-5,
                "patience": 17, "min_delta": 1e-4, "sampling": "window", "normalization": "escape-v1",
                "latent_geometry": "temporal-4x", "physical_validation": True, "evaluate_initial": True,
                "val_samples_per_file": 2, "trainable": "full", "codebook_refresh": "none", "evaluation_split": "val",
                "route_by": "manifest family only, fixed before training; unknown/absent family -> shared",
                "route_bytes_per_window": 1, "code_sha256": {str(p): digest(p) for p in code},
                "note": "All start from same native transferred anchor, fresh AdamW LR 3e-5. Experts and uniform shared get 16,384 updates each. Balanced shared gets 65,536 updates, exactly 16,384 per family, matching the four experts' combined update count/dose (not wall-clock, order or optimizer state). Routed system also retains uniform fallback, additional compute/storage. No error oracle, test access or promotion."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[domain specialists] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use new directory or --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 12 * 1024 ** 3:
        raise RuntimeError("12 GiB free required; no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        for arm, family in arms.items():
            target = output / arm
            epochs, last = remaining_epochs(target, protocol, "temporal-4x") if args.resume else (16, None)
            if not epochs:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root, "--checkpoint", str(source),
                       "--samples-dir", str(samples), "--output-dir", str(target), "--epochs", str(epochs),
                       "--steps-per-epoch", str(budgets[arm]["steps"]), "--learning-rate", "3e-5", "--weight-decay", "1e-5",
                       "--seed", "2026", "--device", "cuda", "--sampling", budgets[arm]["sampling"], "--normalization", "escape-v1",
                       "--latent-geometry", "temporal-4x", "--physical-validation", "--validation-details",
                       "--evaluate-initial", "--val-samples-per-file", "2", "--codebook-diagnostics", "--channel-diagnostics",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "17", "--early-stopping-min-delta", "1e-4"]
            if family:
                command += ["--domain-family", family]
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
        run(["finetuning/summarize_domain_routing.py", "--experiment-dir", str(output)], output / "routing.log", quiet=True)
        (output / "COMPLETE").write_text("Family-specialist/routed full validation complete; no error oracle or promotion.\n")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
