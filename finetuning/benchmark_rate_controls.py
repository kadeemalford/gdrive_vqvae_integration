#!/usr/bin/env python3
"""CPU-only codec control on train/validation windows, never test data.

Sparse DCT byte streams include coefficient indices, quantization, frame header
and reversible normalization/escapes. VQ reference rate remains nominal.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
from tqdm import tqdm

try:
    from .sparse_dct_codec import compress, decompress
    from .reversible_normalization import encode
    from .physical_validation import ordinary_fidelity
except ImportError:
    from sparse_dct_codec import compress, decompress
    from reversible_normalization import encode
    from physical_validation import ordinary_fidelity

ROOT = Path(__file__).resolve().parents[1]
TIERS = (1, 4, 16, 64)
METRICS = ("ordinary_global_centered_relative_rmse", "ordinary_correlation", "ordinary_channel_macro_relative_rmse",
           "ordinary_channel_median_relative_rmse", "ordinary_channel_p90_relative_rmse",
           "ordinary_nonweak_channel_macro_relative_rmse", "bits_per_cell", "compression_ratio",
           "prequantization_energy_fraction_retained")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    configs = defaultdict(list)
    for row in rows:
        configs[(row["split"], row["policy"], row["family"], row["configuration"])].append(row)
    group_rows, families = [], defaultdict(list)
    for (split, policy, family, configuration), values in sorted(configs.items()):
        averages = {}
        for key in METRICS:
            finite = [float(row[key]) for row in values if np.isfinite(float(row[key]))]
            averages[key] = float(np.mean(finite)) if finite else float("nan")
        group_rows.append({"split": split, "policy": policy, "family": family, "configuration": configuration,
                           "windows": len(values), **averages})
        families[(split, policy, family)].append(averages)
    family_rows = []
    for (split, policy, family), values in sorted(families.items()):
        averages = {}
        for key in METRICS:
            finite = [value[key] for value in values if np.isfinite(value[key])]
            averages[key] = float(np.mean(finite)) if finite else float("nan")
        family_rows.append({"split": split, "policy": policy, "family": family, **averages})
    return group_rows, family_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; previous results are preserved")
    reference_root = ROOT / "results/experiments/channel_full_validation_20261008"
    if not (reference_root / "COMPLETE").is_file():
        raise ValueError("Completed full candidate validation required")
    attempts = sorted(reference_root.glob("pilot_global.windows.attempt_*.csv"))
    reference_path = attempts[-1]
    with reference_path.open() as stream:
        reference = {str(Path(row["sample_path"]).resolve()): row for row in csv.DictReader(stream)}
    with (ROOT / "data/finetune/fold_1/samples_manifest.csv").open() as stream:
        manifest = list(csv.DictReader(stream))
    selected = []
    for split in ("train", "val"):
        by_file = defaultdict(list)
        for row in manifest:
            if row["split"] == split:
                by_file[row["drive_id"]].append(row)
        rng = random.Random(2026)
        for file_id in sorted(by_file):
            selected.extend(rng.sample(sorted(by_file[file_id], key=lambda row: row["sample_path"]), min(2, len(by_file[file_id]))))
    if {row["family"] for row in selected if row["split"] == "val"} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    rows = []
    for meta in tqdm(selected, desc="[rate control] sparse DCT, no GPU/training", unit="window"):
        path = (ROOT / meta["sample_path"]).resolve()
        with np.load(path, allow_pickle=False) as payload:
            raw = payload["data"]
        packet = encode(raw)
        cells = raw.size
        index_bits = math.ceil(raw.shape[0] / 16) * math.ceil(raw.shape[1] / 16) * 11
        nominal_bytes = math.ceil(index_bits / 8) + len(packet.side_stream)
        base = {key: meta[key] for key in ("drive_id", "family", "configuration", "split")}
        base.update(sample_path=str(path), input_cells=cells, reference_nominal_bytes=nominal_bytes)
        for tier in TIERS:
            stream, stats = compress(raw, nominal_bytes * tier, quant_bits=8)
            restored = decompress(stream)
            if restored[~packet.valid].tobytes() != raw[~packet.valid].tobytes():
                raise AssertionError("Escape restoration changed source bits")
            metrics = ordinary_fidelity(raw, restored, packet.valid)
            rows.append({**base, "policy": f"sparse_dct_{tier}x", "encoded_bytes": len(stream),
                         "bits_per_cell": 8 * len(stream) / cells, "compression_ratio": 4 * cells / len(stream),
                         **{key: metrics[key] for key in METRICS if key in metrics},
                         "prequantization_energy_fraction_retained": stats["prequantization_energy_fraction_retained"]})
        if meta["split"] == "val":
            original = reference[str(path)]
            if int(original["input_cells"]) != cells or int(original["fixed_index_bits"]) != index_bits:
                raise ValueError("Reference VQ geometry differs from codec budget")
            rows.append({**base, "policy": "vq_global_nominal", "encoded_bytes": nominal_bytes,
                         "bits_per_cell": 8 * nominal_bytes / cells, "compression_ratio": 4 * cells / nominal_bytes,
                         "ordinary_global_centered_relative_rmse": float(original["relative_rmse"]),
                         "ordinary_correlation": float(original["correlation"]),
                         **{key: float(original[key]) for key in METRICS if key.startswith("ordinary_channel_") or key == "ordinary_nonweak_channel_macro_relative_rmse"},
                         "prequantization_energy_fraction_retained": float("nan")})
    groups, families = summarize(rows)
    pooled = []
    for split, policy in sorted({(row["split"], row["policy"]) for row in rows}):
        values = [row for row in rows if row["split"] == split and row["policy"] == policy]
        pooled.append({"split": split, "policy": policy, "input_cells": sum(row["input_cells"] for row in values),
                       "encoded_bytes": sum(row["encoded_bytes"] for row in values),
                       "pooled_compression_ratio": 4 * sum(row["input_cells"] for row in values) / sum(row["encoded_bytes"] for row in values)})
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "windows.csv", rows)
    write_csv(args.output_dir / "groups.csv", groups)
    write_csv(args.output_dir / "families.csv", families)
    write_csv(args.output_dir / "pooled_rates.csv", pooled)
    summary = {"kind": "transparent_sparse_dct_rate_control_v1", "fold": 1, "seed": 2026, "quant_bits": 8,
               "rate_budget_multipliers": list(TIERS), "training_updates": 0, "gpu_used": False,
               "train_windows": sum(row["split"] == "train" for row in selected),
               "validation_windows": sum(row["split"] == "val" for row in selected),
               "families": families, "pooled_rates": pooled, "reference_windows": str(reference_path),
               "warning": "DCT actual payload <= nominal VQ budget; not exactly equal rates, optimized codec or information-theoretic lower bound. Larger tiers are different rates. No test access."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("CPU classical codec control complete; no training, original changes or test access.\n")
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
