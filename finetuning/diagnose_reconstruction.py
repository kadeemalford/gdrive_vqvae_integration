#!/usr/bin/env python3
"""Validation-only stratified spectral/latent diagnostics; no training or model selection."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .latent_geometry import require_standard_geometry
    from .evaluate_checkpoints import load, metrics
    from .analyze_sampling_results import nominal_frequency, write_csv
except ImportError:
    from latent_geometry import require_standard_geometry
    from evaluate_checkpoints import load, metrics
    from analyze_sampling_results import nominal_frequency, write_csv


def spectral_metrics(values, decoded):
    original = np.fft.rfft(values.astype(np.float64), axis=-1)
    recon = np.fft.rfft(decoded.astype(np.float64), axis=-1)
    amplitude = np.abs(original)
    predicted = np.abs(recon)
    denominator = np.linalg.norm(amplitude)
    result = {"fft_magnitude_relative_error": float(np.linalg.norm(predicted - amplitude) / (denominator + 1e-12)),
              "reconstruction_rms_ratio": float(np.linalg.norm(decoded) / (np.linalg.norm(values) + 1e-12))}
    # Weighted phase alignment: 1 means aligned; unreliable where recon has no energy.
    weight = float(np.sum(amplitude * predicted))
    result["fft_phase_alignment"] = float(np.sum(np.real(original * np.conj(recon))) / weight) if weight > 1e-12 else float("nan")
    # Normalized frequency bands avoid inventing sub-resolution physical bands.
    frequencies = np.fft.rfftfreq(values.shape[-1]) / 0.5
    for label, low, high in (("low", 0.0, 0.1), ("mid", 0.1, 0.5), ("high", 0.5, 1.0)):
        mask = (frequencies > low) & (frequencies <= high)
        signal = float(np.sum(amplitude[:, mask] ** 2))
        recovered = float(np.sum(predicted[:, mask] ** 2))
        result[f"{label}_nyquist_power_ratio"] = recovered / signal if signal > 1e-12 else float("nan")
    a = np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=-1))
    b = np.sqrt(np.mean(np.square(decoded, dtype=np.float64), axis=-1))
    result["spatial_rms_correlation"] = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
    return result


def diagnostic_forward(model, tensor):
    """Same pad/crop as VQVAE.forward plus a non-codec continuous-latent ablation."""
    height, width = tensor.shape[-2:]
    multiple = 2 ** model.downsample_levels
    padded = F.pad(tensor, (0, (-width) % multiple, 0, (-height) % multiple))
    latent = model.encoder(padded)
    quantized, vq_loss, indices, perplexity = model.quantizer(latent)
    decoded = model.decoder(quantized)[..., :height, :width]
    continuous = model.decoder(latent)[..., :height, :width]
    return decoded, continuous, indices, perplexity, vq_loss, float(F.mse_loss(latent, quantized).item())


def select_validation(manifest, per_file, seed):
    with manifest.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == "val"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["drive_id"]].append(row)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda r: r["sample_path"])
        selected.extend(rng.sample(candidates, min(per_file, len(candidates))))
    return selected


def save_figure(path, values, decoded, continuous):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scale = max(float(np.percentile(np.abs(values), 99)), 1e-6)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, array, title in zip(axes.flat, (values, decoded, continuous, values - decoded),
                                ("Original normalizado", "VQ-VAE quantizado", "Latente contínuo (diagnóstico)", "Erro quantizado")):
        ax.imshow(array, aspect="auto", cmap="seismic", vmin=-scale, vmax=scale)
        ax.set_title(title)
        ax.set_xlabel("Amostra temporal")
        ax.set_ylabel("Canal")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/home/murilo-barroso/source/smartgrids"))
    parser.add_argument("--samples-per-file", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples_per_file < 1:
        raise ValueError("samples-per-file must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "figures").mkdir()
    sys.path.insert(0, str(args.repo_root))
    from vqvae.evaluate_vqvae import build_model, load_checkpoint_config
    device = torch.device(args.device)
    rows, selections, code_rows, checkpoint_rows = [], [], [], []
    for fold in range(4):
        selected = select_validation(Path(f"data/finetune/fold_{fold}/samples_manifest.csv"), args.samples_per_file, args.seed)
        selections.extend({"fold": fold, **r} for r in selected)
        specs = {"base": Path("models/febus_finetune_base/best.pt")}
        for mode in ("window", "family-group"):
            for kind in ("best", "last"):
                specs[f"{mode}:{kind}"] = args.experiment_dir / f"fold_{fold}/{mode}/{kind}.pt"
        for label, checkpoint_path in specs.items():
            checkpoint, config = load_checkpoint_config(checkpoint_path, device)
            require_standard_geometry(checkpoint)
            model = build_model(config, checkpoint, device)
            checkpoint_rows.append({"fold": fold, "checkpoint": label, "epoch": checkpoint.get("epoch"),
                                    "num_embeddings": model.num_embeddings, "downsample_levels": model.downsample_levels})
            del checkpoint
            histograms = defaultdict(lambda: np.zeros(model.num_embeddings, dtype=np.int64))
            figures = set()
            with torch.no_grad():
                for meta in tqdm(selected, desc=f"[diagnosis] fold {fold} {label}"):
                    values, _ = load(Path(meta["sample_path"]))
                    tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device)
                    recon, continuous, indices, perplexity, vq_loss, latent_mse = diagnostic_forward(model, tensor)
                    decoded = recon[0, 0].cpu().numpy()
                    unquantized = continuous[0, 0].cpu().numpy()
                    continuous_metrics = metrics(values, unquantized)
                    row = {"fold": fold, "checkpoint": label, "sample_path": meta["sample_path"],
                           "family": meta["family"], "configuration": meta["configuration"],
                           "drive_id": meta["drive_id"], "title": meta["title"],
                           "nominal_frequency_hz": nominal_frequency(meta["title"]), "dataset": meta["dataset"],
                           "channels": values.shape[0], "dt_s": meta["dt_s"], "dx_m": meta["dx_m"],
                           "duration_s": values.shape[-1] * float(meta["dt_s"]),
                           "fft_resolution_hz": 1 / (values.shape[-1] * float(meta["dt_s"])),
                           **metrics(values, decoded), **spectral_metrics(values, decoded),
                           "continuous_relative_rmse": continuous_metrics["relative_rmse"],
                           "continuous_correlation": continuous_metrics["correlation"],
                           "quantization_rmse_penalty": metrics(values, decoded)["relative_rmse"] - continuous_metrics["relative_rmse"],
                           "latent_quantization_mse": latent_mse, "per_window_perplexity": float(perplexity.item()),
                           "vq_loss": float(vq_loss.item()),
                           "val_loss": float((0.5 * F.mse_loss(recon, tensor) + 0.5 * F.l1_loss(recon, tensor) + vq_loss).item())}
                    rows.append(row)
                    histograms[meta["family"]] += np.bincount(indices.cpu().numpy(), minlength=model.num_embeddings)
                    group = (meta["family"], meta["configuration"])
                    if label.endswith(":best") and group not in figures:
                        filename = f"fold_{fold}_{label.replace(':', '_')}_{meta['family']}_{meta['configuration'].replace(' ', '_')}.png"
                        save_figure(args.output_dir / "figures" / filename, values, decoded, unquantized)
                        figures.add(group)
            for family, histogram in histograms.items():
                probabilities = histogram[histogram > 0] / histogram.sum()
                code_rows.append({"fold": fold, "checkpoint": label, "family": family,
                                  "active_codes": int(np.count_nonzero(histogram)), "num_embeddings": model.num_embeddings,
                                  "code_usage": float(np.count_nonzero(histogram) / model.num_embeddings),
                                  "aggregate_perplexity": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
                                  "tokens": int(histogram.sum())})
            del model
    write_csv(args.output_dir / "per_sample.csv", rows)
    write_csv(args.output_dir / "selection.csv", selections)
    write_csv(args.output_dir / "codebook.csv", code_rows)
    write_csv(args.output_dir / "checkpoints.csv", checkpoint_rows)
    # Configuration means followed by family means; don't let larger files dominate.
    excluded = {"fold", "checkpoint", "sample_path", "family", "configuration", "drive_id", "title",
                "nominal_frequency_hz", "dataset", "channels", "dt_s", "dx_m"}
    metric_keys = [key for key in rows[0] if key not in excluded]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["family"], row["configuration"])].append(row)
    family_values = defaultdict(list)
    for (label, family, _), values in groups.items():
        family_values[(label, family)].append({key: float(np.nanmean([v[key] for v in values])) for key in metric_keys})
    summary = {f"{label}|{family}": {key: float(np.nanmean([v[key] for v in values])) for key in metric_keys}
               for (label, family), values in family_values.items()}
    (args.output_dir / "summary.json").write_text(json.dumps({"split": "validation", "samples_per_file": args.samples_per_file,
        "seed": args.seed, "selected_windows": len(selections), "model_evaluations": len(rows), "family_macro": summary}, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("Validation diagnostics complete; no training performed.\n")
    print(f"[diagnosis] complete -> {args.output_dir}")


if __name__ == "__main__":
    main()
