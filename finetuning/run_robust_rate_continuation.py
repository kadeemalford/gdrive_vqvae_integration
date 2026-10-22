#!/usr/bin/env python3
"""Bounded fold-0 continuation of both still-improving robust rate branches."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil

import torch

try:
    from .run_sampling_experiment import ROOT, check_splits, run
    from .run_scratch_budget_continuation import FILES, fork_history
    from .run_temporal_density_long import digest, require_cuda, remaining_epochs
except ImportError:
    from run_sampling_experiment import ROOT, check_splits, run
    from run_scratch_budget_continuation import FILES, fork_history
    from run_temporal_density_long import digest, require_cuda, remaining_epochs


ARMS = {"rate_2x": "spatial4-temporal2", "rate_4x": "spatial4-temporal4"}


def validate_source(saved, geometry, samples):
    state = saved["finetune"]
    expected = {"latent_geometry": geometry, "normalization": "robust-escape-v1",
                "physical_validation": True, "learning_rate": 3e-5, "weight_decay": 1e-5,
                "steps_per_epoch": 4096, "sampling": "window", "spectral_weight": 0.0,
                "codebook_refresh": "none", "domain_family": None,
                "early_stopping_metric": "macro-rmse", "trainable": "full"}
    if saved["epoch"] != 24 or any(state.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected source epoch/training policy")
    if Path(state["samples_dir"]).resolve() != samples.resolve():
        raise ValueError("Source fold mismatch")
    if "rng_state" not in state or "optimizer_state_dict" not in saved:
        raise ValueError("Saved optimizer/RNG required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    previous = ROOT / "results/experiments/high_rate_fold0_robust_20261017"
    if not all((previous / p).is_file() for p in ("COMPLETE", "full_validation/COMPLETE")):
        raise ValueError("Completed fold-0 confirmation required")
    old = json.loads((previous / "protocol.json").read_text())
    samples = ROOT / "data/finetune/fold_0"
    check_splits(samples, 0)
    if old["fold"] != 0 or old["seed"] != 2027 or digest(samples / "samples_manifest.csv") != old["manifest_sha256"]:
        raise ValueError("Source dataset manifest/fold changed")
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if any(not (ROOT / r["sample_path"]).is_file() for r in rows if r["split"] in ("train", "val")):
        raise FileNotFoundError("Complete train/validation arrays required")
    hashes = {arm: {name: digest(previous / arm / name) for name in FILES} for arm in ARMS}
    for arm, geometry in ARMS.items():
        saved = torch.load(previous / arm / "last.pt", map_location="cpu", weights_only=False)
        validate_source(saved, geometry, samples)
        del saved
    references = {"frozen_rate1": previous / "rate_1x/best_macro.pt",
                  **{f"initial_{arm}": previous / arm / "best_macro.pt" for arm in ARMS}}
    code = [ROOT / f"finetuning/{name}.py" for name in (
        "run_robust_rate_continuation", "run_scratch_budget_continuation", "run_temporal_density_long",
        "run_sampling_experiment", "finetune_vqvae", "model_factory", "latent_geometry", "physical_validation",
        "robust_escape", "channel_normalization", "reversible_normalization", "reconstruction_objective",
        "sampling", "codebook_refresh", "domain_routing", "evaluate_physical_candidates", "summarize_numeric_adaptation")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {**old, "kind": "robust_rate_budget_continuation_v1", "arms": list(ARMS), "geometries": ARMS,
                "source_experiment": str(previous), "source_protocol_sha256": digest(previous / "protocol.json"),
                "source_files_sha256": hashes, "initial_epoch": 24, "epochs": 40, "additional_epochs": 16,
                "patience": 41, "repo_root": str(Path(args.repo_root).resolve()),
                "reference_sha256": {label: digest(path) for label, path in references.items()},
                "code_sha256": {str(p): digest(p) for p in code},
                "note": "Fork immutable fold-0 histories. Continue both improving 2x/4x branches for 65,536 extra updates each, preserving AdamW/RNG and robust reference. Same rate per branch; no new geometry/loss/data. Frozen 1x has less training compute; compare continuation within each rate. No test access or promotion."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[robust rate continuation] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("New directory or --resume required")
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
        for arm, geometry in ARMS.items():
            target = output / arm
            marker = output / f"{arm}.FORKED"
            if not marker.exists():
                fork_history(previous / arm, target, hashes[arm])
                marker.write_text("Source history copied and verified.\n")
            epochs, last = remaining_epochs(target, protocol, geometry)
            if not epochs:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(previous / arm / "last.pt"), "--resume", str(last),
                       "--samples-dir", str(samples), "--output-dir", str(target), "--epochs", str(epochs),
                       "--steps-per-epoch", "4096", "--learning-rate", "3e-5", "--weight-decay", "1e-5",
                       "--seed", "2027", "--device", "cuda", "--sampling", "window", "--normalization", "robust-escape-v1",
                       "--latent-geometry", geometry, "--physical-validation", "--validation-details", "--evaluate-initial",
                       "--val-samples-per-file", "2", "--codebook-diagnostics", "--channel-diagnostics",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "41", "--early-stopping-min-delta", "1e-4"]
            run(command, output / f"{arm}.log", resume=args.resume, quiet=True)
        full = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root,
                   "--fold", "0", "--output-dir", str(full)]
        for label, path in {**references, **{f"continued_{arm}": output / arm / "best_macro.pt" for arm in ARMS}}.items():
            command += ["--candidate", f"{label}=robust-escape-v1={path}"]
        if full.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("Robust fold-0 2x/4x continuation and full validation complete; no promotion.\n")
        run(["finetuning/summarize_numeric_adaptation.py", "--experiment-dir", str(output)], output / "comparison.log", quiet=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
