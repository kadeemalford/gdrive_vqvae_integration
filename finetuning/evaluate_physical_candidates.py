#!/usr/bin/env python3
"""Full validation of already selected candidates; no optimization or test access."""
import argparse
import csv
import fcntl
from functools import partial
import hashlib
import json
import math
import re
from pathlib import Path
import sys

import torch
from tqdm import tqdm

try:
    from .model_factory import build_model, architecture
    from .latent_geometry import apply_geometry, checkpoint_geometry, latent_positions
    from .finetune_vqvae import load_sample, load_valid_mask
    from .physical_validation import restore_for_validation
    from .reconstruction_objective import evaluate_detailed, validation_metadata, path_fingerprint
    from .analyze_sampling_results import write_csv
    from .run_sampling_experiment import ROOT, check_splits
except ImportError:
    from model_factory import build_model, architecture
    from latent_geometry import apply_geometry, checkpoint_geometry, latent_positions
    from finetune_vqvae import load_sample, load_valid_mask
    from physical_validation import restore_for_validation
    from reconstruction_objective import evaluate_detailed, validation_metadata, path_fingerprint
    from analyze_sampling_results import write_csv
    from run_sampling_experiment import ROOT, check_splits


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidate(text):
    parts = text.split("=", 2)
    if len(parts) != 3 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", parts[0]):
        raise ValueError("Candidate must be safe-label=escape-policy=checkpoint-path")
    label, policy, path = parts
    if policy not in ("escape-v1", "channel-escape-v1", "robust-escape-v1") or not path:
        raise ValueError("Unknown candidate representation or empty path")
    return label, policy, Path(path).resolve()


def score_reference(policies):
    policies = set(policies)
    if "robust-escape-v1" in policies:
        if policies != {"robust-escape-v1"}:
            raise ValueError("Mixed robust/legacy score references are incomparable; use the frozen common-reference evaluator")
        return "robust-escape-v1"
    return "escape-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--fold", type=int, choices=range(4), default=1, help="Prepared fold to evaluate; validation only.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--candidate", action="append", help="Optional repeated label=policy=checkpoint; defaults to the three normalization candidates. Recorded checkpoint geometry is loaded automatically.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    samples = ROOT / f"data/finetune/fold_{args.fold}"
    check_splits(samples, args.fold)
    paths = sorted((samples / "val").glob("*.npz"))
    metadata = validation_metadata(paths, samples / "samples_manifest.csv")
    if {row["family"] for row in metadata.values()} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    specs = [
        ("preservation_full", "escape-v1", ROOT / "results/experiments/preservation_long_20261007/full/best_macro.pt"),
        ("pilot_global", "escape-v1", ROOT / "results/experiments/channel_representation_pilot_20261008/global/best_macro.pt"),
        ("pilot_channel", "channel-escape-v1", ROOT / "results/experiments/channel_representation_pilot_20261008/channel/best_macro.pt"),
    ]
    if args.candidate:
        specs = [parse_candidate(text) for text in args.candidate]
        if len({label for label, _, _ in specs}) != len(specs):
            raise ValueError("Candidate labels must be unique")
    reference_policy = score_reference(policy for _, policy, _ in specs)
    protocol = {"kind": "selected_candidates_full_physical_validation_v1", "fold": args.fold, "split": "val", "windows": len(paths),
                "validation_hash": path_fingerprint(paths), "training_updates": 0, "repo_root": str(Path(args.repo_root).resolve()),
                "candidates": [{"label": label, "policy": policy, "checkpoint": str(path), "sha256": file_hash(path)} for label, policy, path in specs],
                "weak_channel_rule": "diagnostic only: channel temporal std < max(1e-6, window ordinary centered RMS*1e-6). No cells removed from main score.",
                "rate_note": "fixed-width padded indices + explicit normalization/escape bytes only; not total deployed codec rate",
                 "warning": "Selected and expanded validation share development groups; not independent test evidence or automatic promotion"}
    if reference_policy == "robust-escape-v1":
        protocol["score_reference"] = "robust-escape-v1 common ordinary mask/mean/std for all candidates"
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/checkpoint mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[physical-validation] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use a new output directory or --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        sys.path.insert(0, args.repo_root)
        from vqvae.evaluate_vqvae import load_checkpoint_config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summaries = []
        for label, policy, path in specs:
            summary_file = output / f"{label}.json"
            if summary_file.is_file():
                summaries.append(json.loads(summary_file.read_text()))
                continue
            checkpoint, config = load_checkpoint_config(path, device)
            if checkpoint.get("finetune", {}).get("normalization", "legacy") != policy:
                raise ValueError("Candidate normalization does not match checkpoint")
            model = build_model(config, checkpoint, device).eval()
            apply_geometry(model, checkpoint_geometry(checkpoint))
            epoch = int(checkpoint["epoch"])
            del checkpoint
            attempt = len(list(output.glob(f"{label}.windows.attempt_*.csv"))) + 1
            progress = tqdm(total=len(paths), desc=f"[full validation] {label}", unit="window")
            totals = {"input_cells": 0, "fixed_index_bits": 0, "side_bytes": 0}
            with (output / f"{label}.windows.attempt_{attempt:03d}.csv").open("x", newline="") as stream:
                writer = None
                def on_sample(sample_path, metrics):
                    nonlocal writer
                    meta = metadata[sample_path]
                    channels, time_samples = int(meta["channels"]), int(meta["time_samples"])
                    cells = channels * time_samples
                    index_bits = latent_positions(model, channels, time_samples) * math.ceil(math.log2(model.num_embeddings))
                    side_bytes = round(metrics["side_bits_per_cell"] * cells / 8)
                    row = {"sample_path": str(sample_path), "drive_id": meta["drive_id"], "title": meta["title"],
                           "input_cells": cells, "fixed_index_bits": index_bits, "side_bytes": side_bytes, **metrics}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    for key, value in (("input_cells", cells), ("fixed_index_bits", index_bits), ("side_bytes", side_bytes)):
                        totals[key] += value
                    progress.update()
                macro, groups = evaluate_detailed(model, paths, metadata, device, partial(load_sample, normalization=policy),
                    load_valid_mask=partial(load_valid_mask, normalization=policy),
                    fidelity_transform=partial(restore_for_validation, policy=policy, model=model, detailed_channels=True), on_sample=on_sample)
            progress.close()
            row = {"label": label, "policy": policy, "selected_checkpoint_epoch": epoch,
                   "architecture": architecture(config), "latent_geometry": model.latent_geometry,
                   "model_parameters": sum(p.numel() for p in model.parameters()),
                   "num_embeddings": model.num_embeddings, **macro, **totals,
                   "pooled_index_plus_side_ratio": 32 * totals["input_cells"] / (totals["fixed_index_bits"] + 8 * totals["side_bytes"])}
            write_csv(output / f"{label}.groups.csv", groups)
            temporary = output / f"{label}.json.tmp"
            temporary.write_text(json.dumps(row, indent=2) + "\n")
            temporary.replace(summary_file)
            summaries.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_csv(output / "summary.csv", summaries)
        (output / "summary.json").write_text(json.dumps({"protocol": protocol, "candidates": summaries}, indent=2) + "\n")
        (output / "COMPLETE").write_text("Full selected-candidate validation complete; no training or test access.\n")
        print(output / "summary.json")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
