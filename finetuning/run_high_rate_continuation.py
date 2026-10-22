#!/usr/bin/env python3
"""Bounded continuation of the improving highest-rate branch; preserve source history."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    previous = ROOT / "results/experiments/neural_rate_sweep_20261014"
    if not (previous / "COMPLETE").is_file():
        raise ValueError("Completed rate sweep required")
    source = previous / "rate_4x"
    hashes = {name:digest(source / name) for name in FILES}
    saved = torch.load(source / "last.pt", map_location="cpu", weights_only=False)
    state = saved["finetune"]
    if (saved["epoch"] != 24 or state["latent_geometry"] != "spatial4-temporal4"
            or state["learning_rate"] != 3e-5 or state["steps_per_epoch"] != 4096
            or "rng_state" not in state or "optimizer_state_dict" not in saved):
        raise ValueError("Unexpected source continuation state")
    del saved
    old = json.loads((previous / "protocol.json").read_text())
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    if digest(samples / "samples_manifest.csv") != old["manifest_sha256"]:
        raise ValueError("Dataset manifest differs")
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if any(not (ROOT / r["sample_path"]).is_file() for r in rows if r["split"] in ("train", "val")):
        raise FileNotFoundError("Complete train/validation arrays required")
    code = [ROOT / f"finetuning/{name}.py" for name in ("run_high_rate_continuation", "run_scratch_budget_continuation",
            "finetune_vqvae", "model_factory", "latent_geometry", "physical_validation", "reconstruction_objective",
            "sampling", "codebook_refresh", "domain_routing", "reversible_normalization",
            "evaluate_physical_candidates", "summarize_numeric_adaptation")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {**old, "kind": "high_rate_budget_continuation_v1", "arms": ["rate_4x"],
                "source": str(source / "last.pt"), "source_sha256": hashes["last.pt"],
                "source_experiment": str(previous), "source_files_sha256": hashes, "initial_epoch": 24,
                "epochs": 40, "additional_epochs": 16, "patience": 41,
                "initial_validation_expected_equal": False, "repo_root": str(Path(args.repo_root).resolve()),
                "code_sha256": {str(p):digest(p) for p in code},
                "note": "Resume high-rate epochs 25–40 with saved AdamW/RNG and LR 3e-5. Source checkpoints/history immutable; selected epoch24 retained. Fixed extra 65,536 updates, same bitrate/geometry/corpus/objective. Other rate arms frozen for comparison; unequal added training compute. No test/promotion."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[high rate continuation] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use new directory or --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 6 * 1024 ** 3:
        raise RuntimeError("6 GiB free required, no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        target = output / "rate_4x"
        marker = output / "rate_4x.FORKED"
        if not marker.exists():
            fork_history(source, target, hashes)
            marker.write_text("Source history copied and verified.\n")
        epochs, last = remaining_epochs(target, protocol, "spatial4-temporal4")
        if epochs:
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(source / "last.pt"), "--resume", str(last), "--samples-dir", str(samples),
                       "--output-dir", str(target), "--epochs", str(epochs), "--steps-per-epoch", "4096",
                       "--learning-rate", "3e-5", "--weight-decay", "1e-5", "--seed", "2026", "--device", "cuda",
                       "--sampling", "window", "--normalization", "escape-v1", "--latent-geometry", "spatial4-temporal4",
                       "--physical-validation", "--validation-details", "--evaluate-initial", "--val-samples-per-file", "2",
                       "--codebook-diagnostics", "--channel-diagnostics", "--early-stopping-metric", "macro-rmse",
                       "--early-stopping-patience", "41", "--early-stopping-min-delta", "1e-4"]
            run(command, output / "rate_4x.log", resume=args.resume, quiet=True)
        full = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--output-dir", str(full)]
        for label, path in (("frozen_rate1", previous / "rate_1x/best_macro.pt"),
                            ("frozen_rate2", previous / "rate_2x/best_macro.pt"),
                            ("initial_rate4", source / "best_macro.pt"), ("continued_rate4", target / "best_macro.pt")):
            command += ["--candidate", f"{label}=escape-v1={path}"]
        if full.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("High-rate continuation/full validation complete, prior rates frozen; no promotion.\n")
        run(["finetuning/summarize_numeric_adaptation.py", "--experiment-dir", str(output)], output / "comparison.log", quiet=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
