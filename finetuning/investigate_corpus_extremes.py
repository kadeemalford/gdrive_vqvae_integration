#!/usr/bin/env python3
"""Read-only follow-up: locate extreme finite values and test normalization precision."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .analyze_sampling_results import write_csv
    from .run_capacity_probe import ROOT
except ImportError:
    from analyze_sampling_results import write_csv
    from run_capacity_probe import ROOT


def normalization_impact(data, threshold=1e20):
    raw = np.asarray(data, dtype=np.float64)
    mask = np.abs(raw) > threshold
    count = int(mask.sum())
    untouched = ~mask.any(axis=1)
    normalized = ((raw - raw.mean()) / (raw.std() + 1e-6)).astype(np.float32)
    varying = np.ptp(raw[untouched], axis=1) > 0
    erased = np.ptp(normalized[untouched], axis=1) == 0
    total_energy = float(np.sum(normalized.astype(np.float64) ** 2))
    return {"extreme_cells": count, "raw_abs_max": float(np.abs(raw).max()),
            "raw_global_std": float(raw.std()), "channels_with_extremes": int(mask.any(axis=1).sum()),
            "channel_indices_with_extremes": json.dumps(np.flatnonzero(mask.any(axis=1)).tolist()),
            "unaffected_channels": int(untouched.sum()), "unaffected_varying_channels": int(varying.sum()),
            "varying_channels_erased_by_global_normalization": int(np.count_nonzero(varying & erased)),
            "extreme_normalized_energy_fraction": float(np.sum(normalized[mask].astype(np.float64) ** 2) / total_energy) if total_energy else None,
            "unaffected_raw_temporal_std_mean": float(np.mean(np.std(raw[untouched], axis=1))) if untouched.any() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    with (args.audit_dir / "files.csv").open() as stream:
        suspects = {r["drive_id"]: r for r in csv.DictReader(stream) if int(r["finite_values_abs_gt_1e20"]) > 0}
    with (ROOT / "data/finetune/fold_0/samples_manifest.csv").open() as stream:
        selected = [r for r in csv.DictReader(stream) if r["drive_id"] in suspects]
    results = []
    grouped = defaultdict(list)
    for row in selected:
        with np.load(ROOT / row["sample_path"], allow_pickle=False) as payload:
            measured = normalization_impact(payload["data"])
        result = {"drive_id": row["drive_id"], "configuration": row["configuration"], "title": row["title"],
                  "sample_path": row["sample_path"], "block_index": row["block_index"], "time_start": row["time_start"], **measured}
        results.append(result)
        grouped[row["drive_id"]].append(result)
    summaries = []
    for drive_id, rows in grouped.items():
        affected = [r for r in rows if r["extreme_cells"]]
        summaries.append({"drive_id": drive_id, "title": suspects[drive_id]["title"],
                          "configuration": suspects[drive_id]["configuration"], "prepared_windows_checked": len(rows),
                          "affected_windows": len(affected), "extreme_cells_in_prepared": sum(r["extreme_cells"] for r in rows),
                          "raw_extreme_cells": int(suspects[drive_id]["finite_values_abs_gt_1e20"]),
                          "affected_blocks": sorted({int(r["block_index"]) for r in affected}),
                          "max_abs_in_affected": max((r["raw_abs_max"] for r in affected), default=None),
                          "erased_channel_instances": sum(r["varying_channels_erased_by_global_normalization"] for r in affected),
                          "unaffected_varying_channel_instances": sum(r["unaffected_varying_channels"] for r in affected)})
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "prepared_windows.csv", results)
    (args.output_dir / "summary.json").write_text(json.dumps({"threshold": 1e20, "files": summaries,
        "warning": "Diagnostic threshold, not proof of invalid acquisition values. No masking, clipping or data changes were performed."}, indent=2) + "\n")
    print(f"[extremes] checked {len(results)} prepared windows -> {args.output_dir}")


if __name__ == "__main__":
    main()
