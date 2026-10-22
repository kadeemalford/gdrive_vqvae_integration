#!/usr/bin/env python3
"""Frozen checkpoint/global versus channel normalization, train/val only.

Raw-space comparisons use the SAME ordinary cells and denominator for each policy.
Channel normalization is inference-only domain shift, not a trained-model comparison.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

try:
    from .latent_geometry import require_standard_geometry
    from .physical_validation import ordinary_fidelity
    from .reversible_normalization import encode, decode
    from .channel_normalization import encode_channel, decode_channel
    from .analyze_sampling_results import write_csv
    from .run_capacity_probe import ROOT, nominal_rates
except ImportError:
    from latent_geometry import require_standard_geometry
    from physical_validation import ordinary_fidelity
    from reversible_normalization import encode, decode
    from channel_normalization import encode_channel, decode_channel
    from analyze_sampling_results import write_csv
    from run_capacity_probe import ROOT, nominal_rates


def aggregate(rows):
    configs = defaultdict(list)
    keys = [key for key in rows[0] if key not in ("sample_path", "drive_id", "title", "split", "family", "configuration", "policy")]
    for row in rows:
        configs[(row["split"], row["policy"], row["family"], row["configuration"])].append(row)
    groups, families = [], defaultdict(list)
    for (split, policy, family, configuration), values in sorted(configs.items()):
        metrics = {key: float(np.mean([v[key] for v in values])) for key in keys}
        groups.append({"split": split, "policy": policy, "family": family, "configuration": configuration, "windows": len(values), **metrics})
        families[(split, policy, family)].append(metrics)
    family_rows = [{"split": split, "policy": policy, "family": family,
                    **{key: float(np.mean([v[key] for v in values])) for key in keys}}
                   for (split, policy, family), values in sorted(families.items())]
    return groups, family_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/experiments/preservation_long_20261007/full/best_macro.pt")
    parser.add_argument("--repo-root", type=Path, default=Path("/home/murilo-barroso/source/smartgrids"))
    parser.add_argument("--samples-per-file", type=int, default=2)
    args = parser.parse_args()
    if args.samples_per_file < 1 or args.output_dir.exists():
        raise ValueError("Use positive samples-per-file and a new output directory")
    with (ROOT / "data/finetune/fold_1/samples_manifest.csv").open() as stream:
        manifest = list(csv.DictReader(stream))
    selected = []
    for split in ("train", "val"):
        by_file = defaultdict(list)
        for row in manifest:
            if row["split"] == split:
                by_file[row["drive_id"]].append(row)
        rng = random.Random(2026)
        for key in sorted(by_file):
            selected.extend(rng.sample(sorted(by_file[key], key=lambda r: r["sample_path"]), min(args.samples_per_file, len(by_file[key]))))
    if {r["family"] for r in selected if r["split"] == "val"} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four validation families required")
    sys.path.insert(0, str(args.repo_root))
    from vqvae.evaluate_vqvae import build_model, load_checkpoint_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, config = load_checkpoint_config(args.checkpoint, device)
    require_standard_geometry(checkpoint)
    model = build_model(config, checkpoint, device).eval()
    del checkpoint
    rows = []
    with torch.no_grad():
        for meta in tqdm(selected, desc="[representation] frozen train/validation", unit="window"):
            with np.load(ROOT / meta["sample_path"], allow_pickle=False) as payload:
                raw = payload["data"]
            baseline_valid = encode(raw).valid
            for policy, encoder, decoder in (("global", encode, decode), ("channel", encode_channel, decode_channel)):
                packet = encoder(raw)
                if not np.array_equal(packet.valid, baseline_valid):
                    raise AssertionError("Different ordinary regions")
                tensor = torch.from_numpy(packet.values)[None, None].to(device)
                predicted = model(tensor)[0][0, 0].cpu().numpy()
                restored = decoder(predicted, packet.side_stream)
                roundtrip = decoder(packet.values, packet.side_stream)
                zero = decoder(np.zeros_like(packet.values), packet.side_stream)
                if restored[~packet.valid].tobytes() != raw[~packet.valid].tobytes():
                    raise AssertionError("Escapes not bit exact")
                rates = nominal_rates(*raw.shape, model.downsample_levels, model.num_embeddings, model.embedding_dim)
                index_bits = rates["latent_positions"] * rates["index_bits"]
                row = {key: meta[key] for key in ("sample_path", "drive_id", "title", "split", "family", "configuration")}
                row.update(policy=policy, **ordinary_fidelity(raw, restored, packet.valid),
                           side_stream_bytes=len(packet.side_stream), side_bits_per_cell=8 * len(packet.side_stream) / raw.size,
                           index_only_bits_per_cell=index_bits / raw.size,
                           index_plus_side_nominal_ratio=32 * raw.size / (index_bits + 8 * len(packet.side_stream)),
                           ordinary_roundtrip_relative_rmse=ordinary_fidelity(raw, roundtrip, packet.valid)["ordinary_global_centered_relative_rmse"],
                           zero_latent_baseline_relative_rmse=ordinary_fidelity(raw, zero, packet.valid)["ordinary_global_centered_relative_rmse"],
                           escaped_cells=int((~packet.valid).sum()))
                rows.append(row)
    groups, families = aggregate(rows)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "windows.csv", rows)
    write_csv(args.output_dir / "groups.csv", groups)
    write_csv(args.output_dir / "families.csv", families)
    summary = {"checkpoint": str(args.checkpoint), "seed": 2026, "fold": 1, "training_updates": 0,
               "selected_train_windows": sum(r["split"] == "train" for r in selected),
               "selected_validation_windows": sum(r["split"] == "val" for r in selected), "families": families,
               "warning": "Frozen globally-trained model: channel policy is an inference-domain-shift diagnostic, not a fair trained normalization ablation. No test inference.",
               "rate_note": "Fixed-width index + explicit normalization/escape bytes only; entropy coder, container and geometry maps not included."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("Frozen representation audit complete; no training or source-data changes.\n")
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
