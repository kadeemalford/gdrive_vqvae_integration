#!/usr/bin/env python3
"""Reconstruct one full-spatial, model-sized FEBUS window per file."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def load_model(repo_root: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(repo_root))
    from vqvae.evaluate_vqvae import build_model, load_checkpoint_config

    checkpoint, config = load_checkpoint_config(checkpoint_path, device)
    return build_model(config, checkpoint, device), config


def normalize(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = np.isfinite(array)
    mean = float(np.mean(array[valid]))
    std = float(np.std(array[valid]))
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[valid] = (array[valid] - mean) / (std + 1e-6)
    return normalized, mean, std


def encode_and_decode(model, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = tensor.shape[-2:]
    multiple = 2 ** model.downsample_levels
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    padded = torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))
    latent = model.encoder(padded)
    _, _, indices, _ = model.quantizer(latent)
    code_vectors = model.quantizer.embedding(indices).view(
        latent.shape[0], latent.shape[2], latent.shape[3], latent.shape[1]
    ).permute(0, 3, 1, 2).contiguous()
    return model.decoder(code_vectors)[..., :height, :width], indices


def corr(x: np.ndarray, y: np.ndarray) -> float:
    a, b = x.reshape(-1).astype(np.float64), y.reshape(-1).astype(np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if not a.size:
        return float("nan")
    a -= a.mean()
    b -= b.mean()
    denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def write_simple_summary(rows: list[dict[str, object]], output: Path) -> None:
    fields = ["file_name", "variable", "frequency_hz", "channels", "correlation", "rmse_relative", "mae", "theoretical_ratio"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reconstructions"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/reconstruction_summary.csv"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    device = torch.device(args.device or config.get("device", "cpu"))
    checkpoint_path = args.checkpoint or Path(config["checkpoint"])
    model, model_config = load_model(Path(config["repo_root"]), checkpoint_path, device)
    chunk_size = int(config.get("model_time_samples", config.get("time_samples", 1024)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    source_paths = sorted(args.samples_dir.glob("*.npz"))
    print(f"[reconstruct] starting: {len(source_paths)} sample(s), device={device}, chunk={chunk_size}")

    with torch.no_grad():
        for source_path in source_paths:
            print(f"[reconstruct] processing: {source_path}")
            with np.load(source_path, allow_pickle=False) as payload:
                raw = np.asarray(payload["data"], dtype=np.float32)
                metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
            if raw.ndim == 2:
                raw = raw[np.newaxis, ...]
            if raw.ndim != 3:
                raise ValueError(f"Expected [block, channel, time], got {raw.shape}")

            reconstruction = np.empty_like(raw)
            index_chunks: list[np.ndarray] = []
            for block_index in range(raw.shape[0]):
                for start in range(0, raw.shape[2], chunk_size):
                    end = min(start + chunk_size, raw.shape[2])
                    prepared, mean, std = normalize(raw[block_index, :, start:end])
                    tensor = torch.from_numpy(prepared).unsqueeze(0).unsqueeze(0).to(device)
                    recon_norm, indices = encode_and_decode(model, tensor)
                    reconstruction[block_index, :, start:end] = (
                        recon_norm[0, 0].cpu().numpy().astype(np.float32) * (std + 1e-6) + mean
                    )
                    index_chunks.append(indices.cpu().numpy().astype(np.uint16).reshape(-1))

            reconstruction[~np.isfinite(raw)] = np.nan
            error = raw - reconstruction
            codes = np.concatenate(index_chunks)
            mse = float(np.nanmean(np.square(error, dtype=np.float64)))
            rmse = math.sqrt(mse)
            rms = float(np.sqrt(np.nanmean(np.square(raw, dtype=np.float64))))
            output_path = args.output_dir / source_path.name
            output_metadata = {
                **metadata,
                "model_config": model_config,
                "normalization": "per-time-chunk",
                "model_time_samples": chunk_size,
            }
            np.savez_compressed(
                output_path,
                raw=raw,
                recon=reconstruction,
                error=error,
                indices=codes,
                metadata_json=json.dumps(output_metadata, sort_keys=True),
            )
            rows.append({
                **output_metadata,
                "file_name": metadata.get("local_filename", source_path.name),
                "source_sample": str(source_path),
                "reconstruction": str(output_path),
                "mse": mse,
                "rmse": rmse,
                "rmse_relative": rmse / (rms + 1e-6),
                "mae": float(np.nanmean(np.abs(error), dtype=np.float64)),
                "correlation": corr(raw, reconstruction),
                "original_bytes": int(raw.nbytes),
                "index_bytes": int(codes.nbytes),
                "theoretical_ratio": float(raw.nbytes / max(codes.nbytes, 1)),
                "num_codes": int(np.unique(codes).size),
            })
            print(f"[reconstruct] complete: {output_path} (correlation={rows[-1]['correlation']:.4f}, rmse={rmse:.4g})")

    metrics_path = args.metrics_dir / "per_sample_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with metrics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_simple_summary(rows, args.summary_output)
    (args.metrics_dir / "summary.json").write_text(json.dumps({
        "num_files": len(rows),
        "checkpoint": str(checkpoint_path),
        "normalization": "per-time-chunk",
        "model_time_samples": chunk_size,
        "model_config": model_config,
    }, indent=2, sort_keys=True))
    print(f"[reconstruct] success: wrote {len(rows)} reconstruction(s) and {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
