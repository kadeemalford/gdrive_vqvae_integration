#!/usr/bin/env python3
"""Continue both scratch arms without changing optimizer/RNG, loss or quantizer."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil

import torch

try:
    from .model_factory import architecture
    from .run_sampling_experiment import ROOT, check_splits, run
    from .run_temporal_density_long import digest, require_cuda, remaining_epochs
except ImportError:
    from model_factory import architecture
    from run_sampling_experiment import ROOT, check_splits, run
    from run_temporal_density_long import digest, require_cuda, remaining_epochs

FILES = ("last.pt", "best.pt", "best_macro.pt", "metrics.csv", "validation_metrics.csv", "validation_groups.csv")
ARMS = ("scratch_4x", "scratch_temporal")


def fork_history(source, target, expected):
    """Atomic copies of immutable source history; never overwrite divergent data."""
    target.mkdir(exist_ok=True)
    for name in FILES:
        destination = target / name
        if destination.exists():
            if digest(destination) != expected[name]:
                raise ValueError("Partial fork differs from source; refusing overwrite")
            continue
        temporary = target / f"{name}.fork.tmp"
        shutil.copy2(source / name, temporary)
        if digest(temporary) != expected[name]:
            raise ValueError("Fork checksum mismatch")
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = ROOT / "results/experiments/architecture_initialization_pilot_20261010"
    if not (source / "COMPLETE").is_file():
        raise ValueError("Completed screening/full validation required")
    old = json.loads((source / "protocol.json").read_text())
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    if digest(samples / "samples_manifest.csv") != old["manifest_sha256"]:
        raise ValueError("Source dataset manifest changed")
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if any(not (ROOT / row["sample_path"]).is_file() for row in rows if row["split"] in ("train", "val")):
        raise FileNotFoundError("Complete train/validation arrays required")
    hashes = {arm: {name: digest(source / arm / name) for name in FILES} for arm in ARMS}
    for arm in ARMS:
        saved = torch.load(source / arm / "last.pt", map_location="cpu", weights_only=False)
        state = saved["finetune"]
        if saved["epoch"] != 16 or architecture(saved["config"]) != old["architectures"][arm]:
            raise ValueError("Unexpected source epoch/architecture")
        if state["learning_rate"] != 1e-4 or state["steps_per_epoch"] != 4096:
            raise ValueError("Unexpected source optimizer/exposure policy")
        if "optimizer_state_dict" not in saved or "rng_state" not in state:
            raise ValueError("Optimizer/RNG required; not a fresh branch")
        del saved
    reference = source / "transferred_4x/best_macro.pt"
    code = [ROOT / f"finetuning/{name}.py" for name in
            ("run_scratch_budget_continuation", "finetune_vqvae", "model_factory", "latent_geometry",
             "physical_validation", "reconstruction_objective", "sampling", "evaluate_physical_candidates",
             "summarize_numeric_adaptation", "reversible_normalization")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {**old, "kind": "scratch_budget_continuation_v1", "arms": list(ARMS),
                "source_experiment": str(source), "source_protocol_sha256": digest(source / "protocol.json"),
                "source_files_sha256": hashes, "reference": str(reference), "reference_sha256": digest(reference),
                "epochs": 32, "initial_epoch": 16, "additional_epochs": 16, "patience": 33,
                "code_sha256": {str(path): digest(path) for path in code},
                "note": "Fork source histories without replacing originals. Resume epochs 17–32 with saved AdamW/RNG; same LR, objective, codebook, geometry, dataset and validation. Frozen transferred reference has different compute history. Fixed added budget 65,536 updates per scratch arm, no early stop or test access."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[scratch budget] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use a new directory or --resume")
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
        # Explicit fork markers permit recovery of partial initial copying without
        # mistaking modified training checkpoints for a partial fork.
        for arm in ARMS:
            target = output / arm
            marker = output / f"{arm}.FORKED"
            if not marker.exists():
                fork_history(source / arm, target, hashes[arm])
                marker.write_text("Immutable source history copied and verified.\n")
            epochs, last = remaining_epochs(target, protocol, "temporal-4x")
            if not epochs:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(source / arm / "last.pt"), "--resume", str(last),
                       "--samples-dir", str(samples), "--output-dir", str(target), "--epochs", str(epochs),
                       "--steps-per-epoch", "4096", "--learning-rate", "1e-4", "--weight-decay", "1e-5",
                       "--seed", "2026", "--device", "cuda", "--sampling", "window", "--normalization", "escape-v1",
                       "--latent-geometry", "temporal-4x", "--physical-validation", "--validation-details",
                       "--evaluate-initial", "--val-samples-per-file", "2", "--codebook-diagnostics", "--channel-diagnostics",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "33", "--early-stopping-min-delta", "1e-4"]
            run(command, output / f"{arm}.log", resume=args.resume, quiet=True)
        audit = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--output-dir", str(audit),
                   "--candidate", f"frozen_transferred=escape-v1={reference}"]
        for arm in ARMS:
            command += ["--candidate", f"{arm}=escape-v1={output / arm / 'best_macro.pt'}"]
        if audit.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("Scratch budget continuation and full val complete; frozen reference, no promotion.\n")
        run(["finetuning/summarize_numeric_adaptation.py", "--experiment-dir", str(output)], output / "comparison.log", quiet=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
