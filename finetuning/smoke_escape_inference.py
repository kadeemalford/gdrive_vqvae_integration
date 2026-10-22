#!/usr/bin/env python3
"""Four validation windows, frozen real model; integration smoke, not selection."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from .latent_geometry import require_standard_geometry
    from .reversible_normalization import encode, decode
    from .finetune_vqvae import load_sample
    from .analyze_sampling_results import write_csv
    from .run_capacity_probe import ROOT
except ImportError:
    from latent_geometry import require_standard_geometry
    from reversible_normalization import encode, decode
    from finetune_vqvae import load_sample
    from analyze_sampling_results import write_csv
    from run_capacity_probe import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/home/murilo-barroso/source/smartgrids"))
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/experiments/sampling_full_20261005/fold_0/window/best.pt")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    with (ROOT / "data/finetune/fold_0/samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    ids = {"1xlM62wnebrkHFOUcWspz0lzWDDzGCXJj", "1k74OEu0i80YMqpO9U8_JTJ7lJH8V9jZA"}
    selected = [r for r in rows if r["drive_id"] in ids and int(r["block_index"]) in (0, 1) and int(r["time_start"]) == 0]
    if len(selected) != 4 or any(r["split"] != "val" for r in selected):
        raise ValueError("Smoke must use exactly four expected validation windows")
    sys.path.insert(0, str(args.repo_root))
    from vqvae.evaluate_vqvae import load_checkpoint_config, build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, config = load_checkpoint_config(args.checkpoint, device)
    require_standard_geometry(checkpoint)
    model = build_model(config, checkpoint, device).eval()
    results = []
    with torch.no_grad():
        for row in selected:
            path = ROOT / row["sample_path"]
            with np.load(path, allow_pickle=False) as payload:
                raw = payload["data"]
            packet = encode(raw)
            for policy in ("legacy", "escape-v1"):
                values = load_sample(path) if policy == "legacy" else packet.values
                recon = model(torch.from_numpy(values)[None, None].to(device))[0][0, 0].cpu().numpy()
                if not np.isfinite(recon).all():
                    raise AssertionError("Model returned non-finite output")
                if policy == "escape-v1":
                    restored = decode(recon, packet.side_stream).astype(np.float64)
                    exact = restored[~packet.valid].astype(np.float32).tobytes() == raw[~packet.valid].tobytes()
                    if not exact:
                        raise AssertionError("Escape restoration failed")
                else:
                    restored = recon.astype(np.float64) * (raw.std(dtype=np.float64) + 1e-6) + raw.mean(dtype=np.float64)
                    exact = None
                ordinary = raw[packet.valid].astype(np.float64)
                error = restored[packet.valid] - ordinary
                results.append({"sample_path": row["sample_path"], "policy": policy,
                                "escaped_cells": int((~packet.valid).sum()), "escape_bit_exact": exact,
                                "ordinary_raw_relative_rmse": float(np.sqrt(np.sum(error ** 2) / max(float(np.sum(ordinary ** 2)), 1e-60))),
                                "side_stream_bytes": len(packet.side_stream) if policy == "escape-v1" else 0})
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "windows.csv", results)
    (args.output_dir / "protocol.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "split": "val", "windows": 4,
        "training_updates": 0, "purpose": "integration smoke only; hand-selected affected/unaffected windows, not generalization evidence"}, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
