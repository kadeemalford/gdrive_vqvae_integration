#!/usr/bin/env python3
"""Compute time, spatial and spectral fidelity metrics from saved reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def band_error(raw: np.ndarray, recon: np.ndarray, sample_rate: float, low: float, high: float) -> float:
    frequencies = np.fft.rfftfreq(raw.shape[-1], d=1.0 / sample_rate)
    mask = (frequencies >= low) & (frequencies < min(high, sample_rate / 2.0))
    if not np.any(mask):
        return float("nan")
    valid = np.isfinite(raw).all(axis=-1) & np.isfinite(recon).all(axis=-1)
    if not np.any(valid):
        return float("nan")
    raw_power = np.mean(np.abs(np.fft.rfft(raw[valid], axis=-1)[..., mask]) ** 2)
    recon_power = np.mean(np.abs(np.fft.rfft(recon[valid], axis=-1)[..., mask]) ** 2)
    return float(abs(raw_power - recon_power) / (raw_power + 1e-12))


def correlation(raw: np.ndarray, recon: np.ndarray) -> float:
    x = raw.reshape(-1).astype(np.float64)
    y = recon.reshape(-1).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if not x.size:
        return float("nan")
    x -= x.mean(); y -= y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon-dir", type=Path, default=Path("results/reconstructions"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics/fidelity_metrics.csv"))
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    paths = sorted(args.recon_dir.glob("*.npz"))
    print(f"[metrics] starting: {len(paths)} reconstruction(s) from {args.recon_dir}")
    for path in paths:
        print(f"[metrics] calculating: {path}")
        with np.load(path, allow_pickle=False) as payload:
            raw = np.asarray(payload["raw"], dtype=np.float32)
            recon = np.asarray(payload["recon"], dtype=np.float32)
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        dt_s = float(metadata.get("dt_s", 0) or 0)
        sample_rate = 1.0 / dt_s if dt_s > 0 else float(metadata.get("frequency_hz", 0) or 0)
        spatial_axes = (0, 2) if raw.ndim == 3 else (1,)
        raw_valid = np.isfinite(raw)
        recon_valid = np.isfinite(recon)
        raw_count = raw_valid.sum(axis=spatial_axes)
        recon_count = recon_valid.sum(axis=spatial_axes)
        spatial_raw = np.sqrt(np.divide(np.where(raw_valid, raw * raw, 0.0).sum(axis=spatial_axes), raw_count, out=np.full(raw_count.shape, np.nan), where=raw_count > 0))
        spatial_recon = np.sqrt(np.divide(np.where(recon_valid, recon * recon, 0.0).sum(axis=spatial_axes), recon_count, out=np.full(recon_count.shape, np.nan), where=recon_count > 0))
        spatial_valid = np.isfinite(spatial_raw) & np.isfinite(spatial_recon)
        rows.append({
            **metadata,
            "file_name": metadata.get("local_filename", path.name),
            "reconstruction": str(path),
            "correlation": correlation(raw, recon),
            "rmse": float(np.sqrt(np.nanmean(np.square(raw - recon, dtype=np.float64)))),
            "spatial_rms_corr": float(np.corrcoef(spatial_raw[spatial_valid], spatial_recon[spatial_valid])[0, 1]) if spatial_valid.any() and np.std(spatial_raw[spatial_valid]) and np.std(spatial_recon[spatial_valid]) else 0.0,
            "spectral_sample_rate_hz": sample_rate,
            "band_0_1_power_error": band_error(raw, recon, sample_rate, 0, 1) if sample_rate else float("nan"),
            "band_1_5_power_error": band_error(raw, recon, sample_rate, 1, 5) if sample_rate else float("nan"),
            "band_5_25_power_error": band_error(raw, recon, sample_rate, 5, 25) if sample_rate else float("nan"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"[metrics] success: wrote {len(rows)} fidelity row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
