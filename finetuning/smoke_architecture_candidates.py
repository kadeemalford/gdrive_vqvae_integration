#!/usr/bin/env python3
"""Disposable maximum-size gradient/memory check; saves metrics, never model weights."""
import argparse
import csv
import json
import sys
import time

import torch
import torch.nn.functional as F

try:
    from .finetune_vqvae import load_sample, load_valid_mask
    from .latent_geometry import latent_positions
    from .model_factory import build_model, new_model
    from .run_sampling_experiment import ROOT
    from .run_temporal_density_long import digest, require_cuda
except ImportError:
    from finetune_vqvae import load_sample, load_valid_mask
    from latent_geometry import latent_positions
    from model_factory import build_model, new_model
    from run_sampling_experiment import ROOT
    from run_temporal_density_long import digest, require_cuda


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve existing smoke evidence")
    require_cuda()
    sys.path.insert(0, args.repo_root)
    source = ROOT / "results/experiments/temporal_density_long_20261008/temporal-4x/best_macro.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    with (ROOT / "data/finetune/fold_1/samples_manifest.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    meta = max(rows, key=lambda r: (int(r["channels"]) * int(r["time_samples"]), r["sample_path"]))
    path = ROOT / meta["sample_path"]
    values = load_sample(path, normalization="escape-v1")
    tensor = torch.from_numpy(values).cuda()[None, None]
    mask = torch.from_numpy(load_valid_mask(path)).cuda()[None, None]
    results = []
    for label in ("transferred_4x", "scratch_4x", "scratch_temporal"):
        torch.manual_seed(2026)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = (build_model(config, checkpoint, "cuda") if label == "transferred_4x" else
                 new_model({**config, "architecture": "temporal-dilated-v1" if label == "scratch_temporal" else "native-v1"}, "cuda"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        model.train()
        recon, vq, indices, _ = model(tensor)
        if recon.shape != tensor.shape or indices.numel() != latent_positions(model, *values.shape):
            raise AssertionError("Shape/rate accounting differs from native forward")
        loss = .5 * F.mse_loss(recon[mask], tensor[mask]) + .5 * F.l1_loss(recon[mask], tensor[mask]) + vq
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite smoke loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError("Nonfinite or absent gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        results.append({"label": label, "parameters": sum(p.numel() for p in model.parameters()),
                        "shape": list(values.shape), "indices": indices.numel(), "discarded_updates": 1,
                        "elapsed_sec": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        del model, optimizer, recon, vq, indices, loss
    if len({r["parameters"] for r in results}) != 1:
        raise AssertionError("Parameter counts differ")
    args.output_dir.mkdir(parents=True)
    summary = {"source": str(source), "source_sha256": digest(source), "training_case": meta,
               "weights_saved": False, "updates_discarded": True, "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("Maximum-size gradient smoke passed, all updates discarded.\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
