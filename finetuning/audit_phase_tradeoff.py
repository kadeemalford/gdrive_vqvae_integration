#!/usr/bin/env python3
"""Paired phase diagnostics with target-only weights; no training or test access."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

try:
    from .analyze_sampling_results import write_csv
    from .diagnose_reconstruction import select_validation, spectral_metrics
    from .evaluate_physical_candidates import file_hash
    from .finetune_vqvae import load_sample, load_valid_mask
    from .latent_geometry import apply_geometry, checkpoint_geometry
    from .physical_validation import restore_for_validation
    from .reconstruction_objective import aggregate_validation
    from .run_sampling_experiment import ROOT, check_splits
    from .run_temporal_density_long import require_cuda
    from .evaluate_checkpoints import metrics
except ImportError:
    from analyze_sampling_results import write_csv
    from diagnose_reconstruction import select_validation, spectral_metrics
    from evaluate_physical_candidates import file_hash
    from finetune_vqvae import load_sample, load_valid_mask
    from latent_geometry import apply_geometry, checkpoint_geometry
    from physical_validation import restore_for_validation
    from reconstruction_objective import aggregate_validation
    from run_sampling_experiment import ROOT, check_splits
    from run_temporal_density_long import require_cuda
    from evaluate_checkpoints import metrics


def target_weighted_phase(original, predicted):
    """Non-DC temporal FFT; weights never depend on reconstruction amplitude.

    Undefined predicted phases contribute zero to the unconditional score;
    conditional alignment and energy coverage expose that convention explicitly.
    These are diagnostic metrics, not a differentiable training objective.
    """
    original, predicted = np.asarray(original, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if original.shape != predicted.shape or original.ndim != 2:
        raise ValueError("Matching channel-by-time arrays required")
    a, b = np.fft.rfft(original, axis=-1), np.fft.rfft(predicted, axis=-1)
    amplitude, recovered = np.abs(a), np.abs(b)
    weight = amplitude ** 2
    threshold = max(float(amplitude.max()) * 1e-8, 1e-12)
    usable = (amplitude > threshold) & (recovered > threshold)
    cosine = np.zeros_like(amplitude)
    cosine[usable] = np.real(a[usable] * np.conj(b[usable])) / (amplitude[usable] * recovered[usable])
    frequency = np.fft.rfftfreq(original.shape[-1]) / .5
    result = {}
    for band, low, high in (("all", 0., 1.), ("low", 0., .1), ("mid", .1, .5), ("high", .5, 1.)):
        selected = (frequency > low) & (frequency <= high)
        energy = float(weight[:, selected].sum())
        covered = float((weight * usable)[:, selected].sum())
        numerator = float((weight * cosine)[:, selected].sum())
        result[f"target_phase_{band}"] = numerator / energy if energy > 1e-24 else float("nan")
        result[f"target_phase_coverage_{band}"] = covered / energy if energy > 1e-24 else float("nan")
        result[f"target_phase_conditional_{band}"] = numerator / covered if covered > 1e-24 else float("nan")
        error = float(np.square(np.abs(b[:, selected] - a[:, selected])).sum())
        result[f"complex_fft_relative_error_{band}"] = float(np.sqrt(error / energy)) if energy > 1e-24 else float("nan")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory, preserving previous evidence")
    require_cuda()
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    manifest = samples / "samples_manifest.csv"
    selected = select_validation(manifest, 2, 2026)
    if {row["family"] for row in selected} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    sources = {"initial_4x": ROOT / "results/experiments/temporal_density_pilot_20261008/temporal-4x/best_macro.pt",
               "long_4x": ROOT / "results/experiments/temporal_density_long_20261008/temporal-4x/best_macro.pt"}
    protocol = {"kind": "target_weighted_temporal_phase_audit_v1", "split": "val", "training_updates": 0,
                "seed": 2026, "windows": len(selected), "selected": selected,
                "manifest_sha256": file_hash(manifest),
                "candidates": {label: {"checkpoint": str(path), "sha256": file_hash(path)} for label, path in sources.items()},
                "phase_threshold": "max(max target FFT amplitude * 1e-8, 1e-12), per window; non-DC; target squared-amplitude weights",
                "warning": "Exploratory diagnostics on reused development configurations, normalized Nyquist bands not physical bands; no automatic promotion."}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    sys.path.insert(0, args.repo_root)
    from vqvae.evaluate_vqvae import build_model, load_checkpoint_config
    device = torch.device("cuda")
    windows, summaries, groups = [], [], []
    for label, path in sources.items():
        checkpoint, config = load_checkpoint_config(path, device)
        if checkpoint["finetune"]["normalization"] != "escape-v1":
            raise ValueError("Expected global escape-v1 checkpoint")
        model = build_model(config, checkpoint, device).eval()
        apply_geometry(model, checkpoint_geometry(checkpoint))
        del checkpoint
        rows = []
        with torch.no_grad():
            for meta in tqdm(selected, desc=f"[target phase] {label}", unit="window"):
                sample = (ROOT / meta["sample_path"]).resolve()
                values = load_sample(sample, normalization="escape-v1")
                tensor = torch.from_numpy(values).to(device)[None, None]
                decoded = model(tensor)[0][0, 0].cpu().numpy()
                original, predicted, _ = restore_for_validation(sample, decoded, "escape-v1", model)
                valid = load_valid_mask(sample)
                row = {"family": meta["family"], "configuration": meta["configuration"],
                       **metrics(original[valid], predicted[valid]), **spectral_metrics(original, predicted),
                       **target_weighted_phase(original, predicted)}
                rows.append(row)
                windows.append({"label": label, "sample_path": str(sample), "drive_id": meta["drive_id"], **row})
        macro, grouped = aggregate_validation(rows)
        summaries.append({"label": label, **macro})
        groups.extend({"label": label, **row} for row in grouped)
        del model
        torch.cuda.empty_cache()
    write_csv(args.output_dir / "windows.csv", windows)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "groups.csv", groups)
    (args.output_dir / "COMPLETE").write_text("Target-weighted phase audit complete; no training or test access.\n")


if __name__ == "__main__":
    main()
