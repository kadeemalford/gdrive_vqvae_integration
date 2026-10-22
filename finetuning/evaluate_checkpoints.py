#!/usr/bin/env python3
"""Evaluate base and fine-tuned checkpoints on the same FEBUS test windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from .latent_geometry import require_standard_geometry
except ImportError:
    from latent_geometry import require_standard_geometry


def load(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        data = np.asarray(payload["data"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    mean = float(np.mean(data, dtype=np.float64))
    std = float(np.std(data, dtype=np.float64))
    return ((data.astype(np.float64) - mean) / (std + 1e-6)).astype(np.float32), metadata


def metrics(values: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    error = values - recon
    rmse = float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    corr = float(np.corrcoef(values.ravel(), recon.ravel())[0, 1]) if np.std(recon) > 0 else 0.0
    return {"correlation": corr, "relative_rmse": rmse / (rms + 1e-12), "snr_db": float(20 * math.log10((rms + 1e-12) / (rmse + 1e-12)))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path; repeat for each checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-samples-per-group", type=int, default=0, help="Stratified smoke-test cap; 0 evaluates all windows.")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root))
    from vqvae.evaluate_vqvae import build_model, load_checkpoint_config

    specs = [item.split("=", 1) for item in args.checkpoint]
    if any(len(item) != 2 for item in specs):
        raise ValueError("--checkpoint must be label=path")
    paths = sorted(args.samples_dir.glob("*.npz"))
    if args.max_samples_per_group:
        if not args.manifest or args.max_samples_per_group < 1:
            raise ValueError("A positive group cap requires --manifest")
        with args.manifest.open(newline="") as stream:
            metadata = {Path(row["sample_path"]).resolve(): row for row in csv.DictReader(stream)}
        groups = defaultdict(list)
        for path in paths:
            row = metadata[path.resolve()]
            groups[(row["family"], row["configuration"])].append(path)
        rng = random.Random(2026)
        paths = sorted(path for key in sorted(groups) for path in rng.sample(groups[key], min(len(groups[key]), args.max_samples_per_group)))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError("No evaluation windows found")
    device = torch.device(args.device)
    models = {}
    for label, path_text in specs:
        checkpoint, config = load_checkpoint_config(Path(path_text), device)
        require_standard_geometry(checkpoint)
        models[label] = build_model(config, checkpoint, device)
    rows = []
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, path in enumerate(paths, start=1):
        values, metadata = load(path)
        tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device)
        for label, model in models.items():
            with torch.no_grad():
                recon, _, _, _ = model(tensor)
            result = metrics(values, recon[0, 0].cpu().numpy())
            key = f"{label}:{metadata['family']}"
            for metric, value in result.items():
                totals[label][metric].append(value)
                totals[key][metric].append(value)
            rows.append({"checkpoint": label, "sample_path": str(path), "family": metadata["family"], "configuration": metadata["configuration"], **result})
        if index % 500 == 0:
            print(f"[finetune-evaluate] processed {index}/{len(paths)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {key: {metric: float(np.mean(values)) for metric, values in grouped.items()} | {"samples": len(next(iter(grouped.values())))} for key, grouped in totals.items()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[finetune-evaluate] success: {len(paths)} sample(s) -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
