#!/usr/bin/env python3
"""Disposable forward/backward memory test on the largest train geometry.

One optimizer step per policy is discarded. No model/checkpoint/source-data writes.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .latent_geometry import POLICIES, apply_geometry, latent_positions
    from .reversible_normalization import encode
    from .run_sampling_experiment import ROOT
except ImportError:
    from latent_geometry import POLICIES, apply_geometry, latent_positions
    from reversible_normalization import encode
    from run_sampling_experiment import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    with (ROOT / "data/finetune/fold_1/samples_manifest.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    maximum = max(int(row["channels"]) for row in rows)
    case = min((row for row in rows if int(row["channels"]) == maximum), key=lambda row: row["sample_path"])
    with np.load(ROOT / case["sample_path"], allow_pickle=False) as payload:
        raw = payload["data"]
    packet = encode(raw)
    source = ROOT / "results/experiments/channel_representation_pilot_20261008/global/best_macro.pt"
    sys.path.insert(0, args.repo_root)
    from vqvae.evaluate_vqvae import build_model, load_checkpoint_config
    checkpoint, config = load_checkpoint_config(source, torch.device("cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(packet.values)[None, None].to(device)
    valid = torch.from_numpy(packet.valid)[None, None].to(device)
    results = []
    for policy in POLICIES:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = apply_geometry(build_model(config, checkpoint, device), policy).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
        recon, vq_loss, indices, _ = model(tensor)
        expected = latent_positions(model, *raw.shape)
        if recon.shape != tensor.shape or indices.numel() != expected:
            raise AssertionError("Model crop/index count mismatch")
        loss = .5 * F.mse_loss(recon[valid], tensor[valid]) + .5 * F.l1_loss(recon[valid], tensor[valid]) + vq_loss
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite smoke objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        results.append({"policy": policy, "input_shape": list(raw.shape), "latent_positions": expected,
                        "objective_before_discarded_update": float(loss.item()), "discarded_optimizer_steps": 1,
                        "elapsed_sec": time.perf_counter() - started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None})
        del loss, recon, vq_loss, indices, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"source_checkpoint": str(source), "case": case,
        "device": str(device), "model_weights_saved": False, "policies": results}, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("Disposable native-geometry gradient/memory smoke complete. No models saved.\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
