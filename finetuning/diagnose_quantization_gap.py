#!/usr/bin/env python3
"""Stratified train/val diagnostic; continuous latents are NOT an equal-rate codec."""
import argparse
import json
from pathlib import Path
import random
import sys

import torch
from tqdm import tqdm

try:
    from .analyze_sampling_results import write_csv
    from .diagnose_reconstruction import diagnostic_forward, select_validation
    from .evaluate_checkpoints import metrics
    from .finetune_vqvae import load_sample, load_valid_mask
    from .model_factory import build_model
    from .physical_validation import restore_for_validation
    from .reconstruction_objective import aggregate_validation
    from .run_sampling_experiment import ROOT
    from .run_temporal_density_long import digest, require_cuda
except ImportError:
    from analyze_sampling_results import write_csv
    from diagnose_reconstruction import diagnostic_forward, select_validation
    from evaluate_checkpoints import metrics
    from finetune_vqvae import load_sample, load_valid_mask
    from model_factory import build_model
    from physical_validation import restore_for_validation
    from reconstruction_objective import aggregate_validation
    from run_sampling_experiment import ROOT
    from run_temporal_density_long import digest, require_cuda


def main():
    import csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use new output directory")
    require_cuda()
    manifest = ROOT / "data/finetune/fold_1/samples_manifest.csv"
    with manifest.open() as stream:
        all_rows = list(csv.DictReader(stream))
    selected = select_validation(manifest, 2, 2026)
    rng = random.Random(2026)
    for family in ("doped", "multicore", "multimode", "single_mode"):
        candidates = sorted((r for r in all_rows if r["split"] == "train" and r["family"] == family), key=lambda r:r["sample_path"])
        selected += rng.sample(candidates, 8)
    source = ROOT / "results/experiments/scratch_budget_continuation_20261011"
    sources = {"scratch_temporal": source / "scratch_temporal/best_macro.pt",
               "transferred": ROOT / "results/experiments/architecture_initialization_pilot_20261010/transferred_4x/best_macro.pt"}
    sys.path.insert(0, args.repo_root)
    windows, summaries, groups = [], [], []
    for label, path in sources.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(checkpoint["config"], checkpoint, "cuda").eval()
        del checkpoint
        rows = {"train": [], "val": []}
        with torch.no_grad():
            for meta in tqdm(selected, desc=f"[quantization gap] {label}"):
                path = (ROOT / meta["sample_path"]).resolve()
                tensor = torch.from_numpy(load_sample(path, normalization="escape-v1")).cuda()[None, None]
                discrete, continuous, indices, perplexity, _, latent_mse = diagnostic_forward(model, tensor)
                a, b, _ = restore_for_validation(path, discrete[0, 0].cpu().numpy(), "escape-v1", model)
                _, c, _ = restore_for_validation(path, continuous[0, 0].cpu().numpy(), "escape-v1", model)
                valid = load_valid_mask(path)
                discrete_metrics, continuous_metrics = metrics(a[valid], b[valid]), metrics(a[valid], c[valid])
                row = {"family": meta["family"], "configuration": meta["configuration"], **discrete_metrics,
                       "continuous_relative_rmse": continuous_metrics["relative_rmse"],
                       "continuous_correlation": continuous_metrics["correlation"],
                       "quantization_rmse_gap": discrete_metrics["relative_rmse"] - continuous_metrics["relative_rmse"],
                       "latent_quantization_mse": latent_mse, "per_window_perplexity": float(perplexity.item()),
                       "per_window_unique_codes": int(torch.unique(indices).numel())}
                rows[meta["split"]].append(row)
                windows.append({"label": label, "split": meta["split"], "sample_path": str(path), **row})
        for split, values in rows.items():
            macro, grouped = aggregate_validation(values)
            summaries.append({"label": label, "split": split, **macro})
            groups += [{"label": label, "split": split, **r} for r in grouped]
        del model
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "windows.csv", windows)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "groups.csv", groups)
    (args.output_dir / "protocol.json").write_text(json.dumps({"source_sha256": {k:digest(p) for k,p in sources.items()},
        "manifest_sha256": digest(manifest), "selected": selected, "training_updates": 0,
        "warning": "Continuous decoder input is out of its quantized training distribution; not an equal-bitrate codec, performance bound or causal proof."}, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("Quantization gap diagnostic complete, train/val only.\n")


if __name__ == "__main__":
    main()
