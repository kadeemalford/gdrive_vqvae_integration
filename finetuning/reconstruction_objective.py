"""Optional spectral objective and training-time, split-safe generalist validation."""
from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .latent_geometry import latent_positions
    from .diagnose_reconstruction import spectral_metrics
    from .evaluate_checkpoints import metrics
except ImportError:
    from latent_geometry import latent_positions
    from diagnose_reconstruction import spectral_metrics
    from evaluate_checkpoints import metrics

FFT_SIZES = (128, 256, 512)


def multiresolution_spectral_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of spectral convergence + log1p magnitude error at three STFT scales.

    Per-channel STFT, Hann, hop=N/4, center=False; no conversion of physical units.
    Magnitudes are normalized by window L2 norm. Temporal MSE/L1 must remain present.
    """
    flat_recon = recon.reshape(-1, recon.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    scales = []
    for size in FFT_SIZES:
        if size > target.shape[-1]:
            continue
        window = torch.hann_window(size, dtype=target.dtype, device=target.device)
        original = torch.stft(flat_target, size, hop_length=size // 4, window=window,
                              center=False, return_complex=True).abs() / window.norm()
        predicted = torch.stft(flat_recon, size, hop_length=size // 4, window=window,
                               center=False, return_complex=True).abs() / window.norm()
        convergence = (predicted - original).norm() / original.norm().clamp_min(1e-6)
        log_magnitude = (torch.log1p(predicted) - torch.log1p(original)).abs().mean()
        scales.append(0.5 * convergence + 0.5 * log_magnitude)
    if not scales:
        raise ValueError(f"Spectral objective needs at least {min(FFT_SIZES)} time samples")
    return torch.stack(scales).mean()


def validation_metadata(paths: list[Path], manifest: Path) -> dict[Path, dict]:
    with manifest.open(newline="") as stream:
        rows = {Path(r["sample_path"]).resolve(): r for r in csv.DictReader(stream)}
    selected = {}
    for path in paths:
        meta = rows[path.resolve()]
        if meta["split"] != "val":
            raise ValueError(f"Non-validation sample in validation metrics: {path}")
        selected[path] = meta
    return selected


def stratified_validation(paths: list[Path], metadata: dict, per_file: int, seed: int) -> list[Path]:
    if not per_file:
        return paths
    groups = defaultdict(list)
    for path in sorted(paths):
        groups[metadata[path]["drive_id"]].append(path)
    rng = random.Random(seed)
    return sorted(path for key in sorted(groups) for path in rng.sample(groups[key], min(per_file, len(groups[key]))))


def path_fingerprint(paths: list[Path]) -> str:
    return hashlib.sha256("\n".join(str(p.resolve()) for p in sorted(paths)).encode()).hexdigest()


def aggregate_validation(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Equal config weights within family, then equal family weights. No VQ in RMSE."""
    if not rows:
        raise ValueError("Validation is empty")
    keys = [k for k in rows[0] if k not in ("family", "configuration")]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["configuration"])].append(row)
    groups = []
    families = defaultdict(list)
    for (family, configuration), values in sorted(grouped.items()):
        averages = {key: float(np.mean([v[key] for v in values if np.isfinite(v[key])]))
                    if any(np.isfinite(v[key]) for v in values) else float("nan") for key in keys}
        groups.append({"family": family, "configuration": configuration, "windows": len(values), **averages})
        families[family].append(averages)
    macro = {}
    for key in keys:
        family_means = [float(np.nanmean([v[key] for v in values])) for values in families.values()]
        macro[f"macro_{key}"] = float(np.nanmean(family_means))
    macro["worst_family_relative_rmse"] = max(float(np.mean([v["relative_rmse"] for v in values])) for values in families.values())
    macro["worst_family_correlation"] = min(float(np.mean([v["correlation"] for v in values])) for values in families.values())
    macro["val_families"] = len(families)
    macro["val_configurations"] = len(grouped)
    return macro, groups


def evaluate_detailed(model, paths, metadata, device, load_sample, spectral_weight=0.0, load_valid_mask=None, fidelity_transform=None, on_sample=None):
    model.eval()
    rows = []
    with torch.no_grad():
        for path in paths:
            values = load_sample(path)
            tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device)
            recon, vq_loss, indices, _ = model(tensor)
            if getattr(model, "latent_geometry", "standard") != "standard":
                if indices is None or indices.numel() != latent_positions(model, *values.shape):
                    raise AssertionError("Temporal geometry index accounting differs from actual forward")
            valid = load_valid_mask(path) if load_valid_mask else None
            if valid is None:
                temporal = 0.5 * F.mse_loss(recon, tensor) + 0.5 * F.l1_loss(recon, tensor)
            else:
                mask = torch.from_numpy(valid).to(device)[None, None]
                temporal = 0.5 * F.mse_loss(recon[mask], tensor[mask]) + 0.5 * F.l1_loss(recon[mask], tensor[mask])
            spectral = multiresolution_spectral_loss(recon, tensor)
            decoded = recon[0, 0].cpu().numpy()
            metric_values, metric_decoded, extras = (fidelity_transform(path, decoded) if fidelity_transform
                                                       else (values, decoded, {}))
            meta = metadata[path]
            rows.append({"family": meta["family"], "configuration": meta["configuration"],
                         "val_loss": float((temporal + vq_loss).item()),
                         "temporal_loss": float(temporal.item()), "vq_loss": float(vq_loss.item()),
                         "spectral_loss": float(spectral.item()),
                         "objective": float((temporal + vq_loss + spectral_weight * spectral).item()),
                          **metrics(metric_values if valid is None else metric_values[valid], metric_decoded if valid is None else metric_decoded[valid]),
                          **spectral_metrics(metric_values, metric_decoded), **extras,
                          **({"escaped_cell_fraction": float((~valid).mean())} if valid is not None else {})})
            if on_sample:
                on_sample(path, rows[-1])
    macro, groups = aggregate_validation(rows)
    weighted = {key: float(np.mean([row[key] for row in rows])) for key in ("val_loss", "correlation", "temporal_loss", "vq_loss", "spectral_loss", "objective")}
    return {**weighted, **macro}, groups


def append_metrics(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as stream:
            if next(csv.reader(stream)) != fields:
                raise ValueError(f"Incompatible metric columns: {path}")
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
