#!/usr/bin/env python3
"""Fold-0 rate curve, common robust reference, and no fold-1 weight leakage."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil
import torch

try:
    from .finetune_vqvae import save_checkpoint
    from .run_rate_allocation_confirmation import verify_source_fold
    from .run_sampling_experiment import ROOT, check_splits, run
    from .run_temporal_density_long import digest, require_cuda, remaining_epochs
except ImportError:
    from finetune_vqvae import save_checkpoint
    from run_rate_allocation_confirmation import verify_source_fold
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
    prior = ROOT / "results/experiments/high_rate_continuation_20261015"
    if not (prior / "COMPLETE").is_file():
        raise ValueError("Completed high-rate study required")
    source = ROOT / "results/experiments/rate_allocation_fold0_seed2027/spatial/best_macro.pt"
    smoke = ROOT / "results/experiments/high_rate_fold0_robust_smoke_20261017"
    if not (smoke / "COMPLETE").is_file() or json.loads((smoke / "summary.json").read_text())["source_sha256"] != digest(source):
        raise ValueError("Robust affected-case gradient/escape smoke required")
    samples = ROOT / "data/finetune/fold_0"
    check_splits(samples, 0)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    source_epoch = verify_source_fold(checkpoint, samples)
    if checkpoint["finetune"]["latent_geometry"] != "spatial-4x":
        raise ValueError("Fixed spatial source required")
    del checkpoint
    audit = ROOT / "results/experiments/remaining_dynamic_range_audit_20261014"
    if not (audit / "COMPLETE").is_file():
        raise ValueError("Common-reference range audit required")
    with (audit / "summary.csv").open() as stream:
        if any(int(r["windows_with_additional_escapes"]) != 0 for r in csv.DictReader(stream) if r["split"] == "train"):
            raise ValueError("This confirmation expects unchanged train representations")
    manifest = samples / "samples_manifest.csv"
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    if any(not (ROOT / r["sample_path"]).is_file() for r in rows if r["split"] in ("train", "val")):
        raise FileNotFoundError("Complete dataset required")
    arms = {"rate_1x": "spatial-4x", "rate_2x": "spatial4-temporal2", "rate_4x": "spatial4-temporal4"}
    code = [ROOT / f"finetuning/{name}.py" for name in ("run_high_rate_fold_confirmation", "finetune_vqvae", "model_factory",
            "latent_geometry", "robust_escape", "reversible_normalization", "physical_validation", "reconstruction_objective",
            "sampling", "codebook_refresh", "domain_routing", "run_temporal_density_long", "evaluate_physical_candidates",
            "summarize_numeric_adaptation", "run_rate_allocation_confirmation")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "high_rate_fold_confirmation_robust_v1", "fold": 0, "seed": 2027,
                "arms": list(arms), "geometries": arms, "source": str(source), "source_sha256": digest(source),
                "source_epoch": source_epoch, "repo_root": str(Path(args.repo_root).resolve()),
                "manifest_sha256": digest(manifest), "audit_protocol_sha256": digest(audit / "protocol.json"),
                "smoke_sha256": digest(smoke / "summary.json"),
                "epochs": 24, "steps_per_epoch": 4096, "learning_rate": 3e-5, "weight_decay": 1e-5,
                "patience": 25, "min_delta": 1e-4, "normalization": "robust-escape-v1", "physical_validation": True,
                "sampling": "window", "trainable": "full", "codebook_refresh": "none", "evaluate_initial": True,
                "initial_validation_expected_equal": False, "val_samples_per_file": 2, "evaluation_split": "val",
                "code_sha256": {str(p):digest(p) for p in code},
                "note": "Common fold-0 spatial source, weights unchanged when tagging robust preprocessing; train input unchanged by audited policy, val40 corrected. All arms use same robust ordinary mask/mean/std, not legacy denominators. Fresh AdamW, 98,304 updates each. Different rates/receptive fields; reused development, no test access or promotion. Compare within-fold contrasts, not absolute cross-fold scores."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[high rate confirmation] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("New directory or --resume required")
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
        wrapped = output / "source_robust.pt"
        if not wrapped.exists():
            saved = torch.load(source, map_location="cpu", weights_only=False)
            save_checkpoint({"config": saved["config"], "model_state_dict": saved["model_state_dict"], "epoch": 0,
                             "finetune": {"normalization": "robust-escape-v1", "latent_geometry": "spatial-4x", "samples_dir": str(samples)},
                             "source_sha256": digest(source), "source_epoch": source_epoch, "reencoding_only_no_weight_updates": True}, wrapped)
            del saved
        initial = torch.load(wrapped, map_location="cpu", weights_only=False)
        if initial.get("source_sha256") != digest(source) or initial["finetune"]["normalization"] != "robust-escape-v1":
            raise ValueError("Robust source provenance mismatch")
        original = torch.load(source, map_location="cpu", weights_only=False)
        if initial["config"] != original["config"] or initial["model_state_dict"].keys() != original["model_state_dict"].keys():
            raise ValueError("Robust source changed architecture/state keys")
        if any(not torch.equal(value, original["model_state_dict"][key]) for key, value in initial["model_state_dict"].items()):
            raise ValueError("Robust reencoding must not change source weights")
        del original
        del initial
        for arm, geometry in arms.items():
            target = output / arm
            epochs, last = remaining_epochs(target, protocol, geometry) if args.resume else (24, None)
            if not epochs:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root, "--checkpoint", str(wrapped),
                       "--samples-dir", str(samples), "--output-dir", str(target), "--epochs", str(epochs),
                       "--steps-per-epoch", "4096", "--learning-rate", "3e-5", "--weight-decay", "1e-5",
                       "--seed", "2027", "--device", "cuda", "--sampling", "window", "--normalization", "robust-escape-v1",
                       "--latent-geometry", geometry, "--physical-validation", "--validation-details", "--evaluate-initial",
                       "--val-samples-per-file", "2", "--codebook-diagnostics", "--channel-diagnostics",
                       "--early-stopping-metric", "macro-rmse", "--early-stopping-patience", "25", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, quiet=True)
        full = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--fold", "0", "--output-dir", str(full),
                   "--candidate", f"frozen_spatial=robust-escape-v1={wrapped}"]
        for arm in arms:
            command += ["--candidate", f"{arm}=robust-escape-v1={output / arm / 'best_macro.pt'}"]
        if full.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("Fold-0 high-rate/common-robust confirmation complete; no test/promotion.\n")
        run(["finetuning/summarize_numeric_adaptation.py", "--experiment-dir", str(output)], output / "comparison.log", quiet=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
