#!/usr/bin/env python3
"""Read-only corpus/coverage audit; signal descriptors use a training split only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:
    from .analyze_sampling_results import nominal_frequency, write_csv
    from .prepare_finetune_samples import choose_dataset
    from .run_capacity_probe import ROOT
except ImportError:
    from analyze_sampling_results import nominal_frequency, write_csv
    from prepare_finetune_samples import choose_dataset
    from run_capacity_probe import ROOT


def interval_union_length(intervals, tolerance=0):
    intervals = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not intervals:
        return 0.0
    start, end = intervals[0]
    total = 0.0
    for a, b in intervals[1:]:
        if a <= end + tolerance:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + end - start


def window_key(row):
    return (row["drive_id"], int(row["block_index"]), int(row["time_start"]),
            int(row["time_samples"]), int(row["channels"]))


def collect_prepared(manifests):
    canonical, copies = {}, defaultdict(list)
    files = set()
    metadata_keys = ("family", "configuration", "dataset", "dt_s", "dx_m", "fold")
    for manifest in manifests:
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        seen = set()
        for row in rows:
            key = window_key(row)
            if key in seen:
                raise ValueError(f"Repeated window coordinates in {manifest}")
            seen.add(key)
            if key in canonical and any(canonical[key][k] != row[k] for k in metadata_keys):
                raise ValueError("Fold copies disagree on source/geometry metadata")
            canonical.setdefault(key, row)
            path = ROOT / row["sample_path"]
            if not path.exists():
                raise FileNotFoundError(path)
            copies[key].append(path)
            files.add(path)
    unique_bytes = sum(paths[0].stat().st_size for paths in copies.values())
    return list(canonical.values()), {"unique_window_coordinates": len(canonical), "prepared_files": len(files),
        "copy_counts": dict(sorted((str(n), sum(len(p) == n for p in copies.values())) for n in {len(p) for p in copies.values()})),
        "prepared_file_bytes": sum(p.stat().st_size for p in files), "one_copy_estimated_bytes": unique_bytes,
        "fold_copy_excess_estimated_bytes": sum(p.stat().st_size for p in files) - unique_bytes}


def source_geometry(shape):
    if len(shape) == 3:
        return shape[0], shape[1], shape[2], True
    if len(shape) == 2:
        return (1, shape[0], shape[1], True) if shape[0] >= shape[1] else (1, shape[1], shape[0], False)
    raise ValueError(f"Unsupported FEBUS shape: {shape}")


def scan_dataset(dataset, chunk_samples=1024):
    blocks, time_samples, channels, time_first = source_geometry(dataset.shape)
    finite_channels = np.ones((blocks, channels), dtype=bool)
    extremes = 0
    near_float32_limit = 0
    fingerprint = hashlib.sha256((str(dataset.shape) + dataset.dtype.str).encode())
    for block in range(blocks):
        for start in range(0, time_samples, chunk_samples):
            stop = min(time_samples, start + chunk_samples)
            if dataset.ndim == 3:
                chunk = dataset[block, start:stop, :]
            elif time_first:
                chunk = dataset[start:stop, :]
            else:
                chunk = dataset[:, start:stop].T
            fingerprint.update(np.ascontiguousarray(chunk).tobytes())
            finite_channels[block] &= np.isfinite(chunk).all(axis=0)
            absolute = np.abs(chunk.astype(np.float64))
            finite = np.isfinite(absolute)
            extremes += int(np.count_nonzero(finite & (absolute > 1e20)))
            near_float32_limit += int(np.count_nonzero(finite & (absolute >= 0.99 * np.finfo(np.float32).max)))
    return finite_channels, fingerprint.hexdigest(), extremes, near_float32_limit


def scalar_attr(parent, key):
    values = np.asarray(parent.attrs.get(key, []), dtype=float).reshape(-1)
    return float(values[0]) if values.size else None


def audit_raw(row, windows, raw_dir):
    path = raw_dir / f"{row['drive_id']}.h5"
    with h5py.File(path, "r") as handle:
        name, dataset = choose_dataset(handle)
        blocks, time_samples, channels, _ = source_geometry(dataset.shape)
        spacing = np.asarray(dataset.parent.attrs.get("Spacing", []), dtype=float).reshape(-1)
        if spacing.size < 2 or spacing[1] <= 0:
            raise ValueError(f"Invalid/missing Spacing in {path}")
        dt = float(spacing[1]) * 1e-3
        dx = float(spacing[0])
        finite, checksum, extremes, near_limit = scan_dataset(dataset)
        source = dataset.parent.parent
        timestamps = np.asarray(source["time"], dtype=float) if "time" in source else np.array([])
        clock_valid = timestamps.shape == (blocks,) and np.isfinite(timestamps).all() and np.all((timestamps > 1e9) & (timestamps < 1e11))
        origin = np.asarray(dataset.parent.attrs.get("Origin", []), dtype=float).reshape(-1)
        offset = float(origin[1]) * 1e-3 if origin.size >= 2 else 0.0
        gauge = scalar_attr(dataset.parent, "GaugeLength")
        attrs = {key: scalar_attr(dataset.parent, key) for key in ("BlockRate", "BlockOverlap", "DerivationTime", "PulseWidth")}
    by_block = defaultdict(list)
    for window in windows:
        block = int(window["block_index"])
        start = int(window["time_start"])
        count = int(window["time_samples"])
        if window["dataset"] != name or abs(float(window["dt_s"]) - dt) > 1e-12:
            raise ValueError("Prepared/raw metadata mismatch")
        if block >= blocks or start < 0 or start + count > time_samples:
            raise ValueError("Window outside raw block")
        by_block[block].append(window)
    covered_samples, retained_channel_samples, finite_channel_samples = 0., 0., 0
    internal_missing = 0
    raw_intervals, prepared_intervals = [], []
    for block in range(blocks):
        covered = interval_union_length((int(w["time_start"]), int(w["time_start"]) + int(w["time_samples"])) for w in by_block[block])
        covered_samples += covered
        finite_count = int(finite[block].sum())
        finite_channel_samples += finite_count * time_samples
        valid_indices = np.flatnonzero(finite[block])
        if valid_indices.size:
            internal_missing += int(np.count_nonzero(~finite[block, valid_indices[0]:valid_indices[-1] + 1]))
        expected_channels = min(512, finite_count)
        if any(int(w["channels"]) != expected_channels for w in by_block[block]):
            raise ValueError("Prepared channel count differs from current 512-channel preparation policy")
        retained_channel_samples += covered * expected_channels
        if clock_valid:
            base = float(timestamps[block]) + offset
            raw_intervals.append((base, base + time_samples * dt))
            prepared_intervals.extend((base + int(w["time_start"]) * dt,
                                       base + (int(w["time_start"]) + int(w["time_samples"])) * dt) for w in by_block[block])
    file_row = {**row, "local_bytes": path.stat().st_size, "dataset": name, "blocks": blocks,
                "raw_time_samples_per_block": time_samples, "raw_channels": channels, "dt_s": dt, "dx_m": dx,
                "gauge_length_m": gauge, **attrs, "nominal_frequency_hz": nominal_frequency(row["title"]),
                "prepared_windows": len(windows), "raw_stored_seconds": blocks * time_samples * dt,
                "prepared_stored_window_seconds": sum(int(w["time_samples"]) for w in windows) * dt,
                "prepared_block_union_seconds": covered_samples * dt,
                "block_time_coverage": covered_samples / (blocks * time_samples),
                "finite_channel_count_min": int(finite.sum(axis=1).min()), "finite_channel_count_max": int(finite.sum(axis=1).max()),
                "nonfinite_removed_channel_instances": int((~finite).sum()), "internal_removed_channel_instances": internal_missing,
                "retained_finite_channel_time_fraction": retained_channel_samples / finite_channel_samples if finite_channel_samples else 0,
                "clock_timestamps_valid": bool(clock_valid),
                "raw_file_clock_union_seconds": interval_union_length(raw_intervals, 1e-6) if clock_valid else None,
                "prepared_file_clock_union_seconds": interval_union_length(prepared_intervals, 1e-6) if clock_valid else None,
                "finite_values_abs_gt_1e20": extremes, "values_near_float32_limit": near_limit,
                "signal_payload_sha256": checksum}
    return file_row, raw_intervals, prepared_intervals


def content_descriptors(data):
    raw = np.asarray(data, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Prepared sample contains non-finite values")
    std = float(raw.std())
    values = (raw - raw.mean()) / (std + 1e-6)
    energy = values ** 2
    total = float(energy.sum())
    if total <= 1e-12:
        raise ValueError("Constant sample needs a separate analysis")
    channels, time_samples = values.shape
    channel_energy = energy.sum(axis=1)
    probabilities = channel_energy / total
    edge_indices = sorted(set(range(min(3, channels))) | set(range(max(0, channels - 3), channels)))
    top_channels = max(1, math.ceil(channels * .1))
    top_cells = max(1, math.ceil(values.size * .001))
    top_energy = np.partition(energy.ravel(), values.size - top_cells)[-top_cells:].sum()
    centered = values - values.mean(axis=1, keepdims=True)
    temporal_energy = float(np.square(centered).sum())
    spectrum = np.fft.rfft(centered * np.hanning(time_samples), axis=1)
    power = np.mean(np.abs(spectrum) ** 2, axis=0)
    power_positive = power[1:]
    p = power_positive / max(float(power_positive.sum()), 1e-30)
    spectral_entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0]))) / math.log(len(p))
    scale = max(float(power_positive.mean()), 1e-30)
    flatness = float(np.exp(np.mean(np.log(power_positive / scale + 1e-12))))
    frequencies = np.fft.rfftfreq(time_samples) / .5
    high_fraction = float(power[frequencies >= .5].sum() / max(power.sum(), 1e-30))
    norms = np.sqrt(np.square(centered).sum(axis=1))
    denom = norms[:-1] * norms[1:]
    valid = denom > 1e-12
    adjacent = float(np.mean(np.sum(centered[:-1] * centered[1:], axis=1)[valid] / denom[valid])) if valid.any() else None
    return {"channels": channels, "time_samples": time_samples, "raw_abs_max": float(np.abs(raw).max()),
            "finite_values_abs_gt_1e20": int(np.count_nonzero(np.abs(raw) > 1e20)),
            "raw_global_std": std, "channel_mean_energy_fraction": float(time_samples * np.square(values.mean(axis=1)).sum() / total),
            "top_10pct_channels_energy_fraction": float(np.sort(channel_energy)[-top_channels:].sum() / total),
            "edge_3_channels_energy_fraction": float(channel_energy[edge_indices].sum() / total),
            "effective_channels_fraction": float(1 / np.sum(probabilities ** 2) / channels),
            "top_0_1pct_cells_energy_fraction": float(top_energy / total),
            "temporal_spectral_entropy": spectral_entropy, "temporal_spectral_flatness": flatness,
            "high_nyquist_temporal_energy_fraction": high_fraction,
            "adjacent_prepared_row_correlation": adjacent,
            "in_phase_common_mode_fraction": float(channels * np.square(centered.mean(axis=0)).sum() / max(temporal_energy, 1e-30))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/finetune/raw")
    parser.add_argument("--content-fold", type=int, default=1, choices=range(4))
    parser.add_argument("--samples-per-file", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.samples_per_file < 1:
        raise ValueError("samples-per-file must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new audit output directory; existing artifacts are preserved")
    with (ROOT / "data/manifests/febus_folds.csv").open() as stream:
        manifest = list(csv.DictReader(stream))
    canonical, prepared_summary = collect_prepared([ROOT / f"data/finetune/fold_{f}/samples_manifest.csv" for f in range(4)])
    windows_by_file = defaultdict(list)
    for window in canonical:
        windows_by_file[window["drive_id"]].append(window)
    if set(windows_by_file) != {r["drive_id"] for r in manifest}:
        raise ValueError("Prepared file set differs from raw manifest")
    files, intervals = [], defaultdict(lambda: {"raw": [], "prepared": []})
    for row in tqdm(manifest, desc="[corpus] raw coverage/hash/finite channels", unit="file"):
        audited, raw_times, prepared_times = audit_raw(row, windows_by_file[row["drive_id"]], args.raw_dir)
        files.append(audited)
        key = (row["family"], row["configuration"])
        intervals[key]["raw"].extend(raw_times)
        intervals[key]["prepared"].extend(prepared_times)
    grouped = defaultdict(list)
    duplicates = defaultdict(list)
    for row in files:
        grouped[row["family"]].append(row)
        duplicates[row["signal_payload_sha256"]].append(row["drive_id"])
    family_rows = []
    for family, rows in sorted(grouped.items()):
        raw_seconds = sum(r["raw_stored_seconds"] for r in rows)
        retained_seconds = sum(r["prepared_block_union_seconds"] for r in rows)
        clock_raw = sum(interval_union_length(v["raw"], 1e-6) for (f, _), v in intervals.items() if f == family)
        clock_prepared = sum(interval_union_length(v["prepared"], 1e-6) for (f, _), v in intervals.items() if f == family)
        family_rows.append({"family": family, "files": len(rows), "configuration_groups": len({r["configuration"] for r in rows}),
            "local_raw_bytes": sum(r["local_bytes"] for r in rows), "prepared_windows": sum(r["prepared_windows"] for r in rows),
            "raw_stored_seconds": raw_seconds, "prepared_block_union_seconds": retained_seconds,
            "block_time_coverage": retained_seconds / raw_seconds,
            "group_clock_union_seconds": clock_raw, "prepared_group_clock_union_seconds": clock_prepared,
            "files_cropped_below_finite_channels": sum(r["finite_channel_count_max"] > 512 for r in rows),
            "finite_values_abs_gt_1e20": sum(r["finite_values_abs_gt_1e20"] for r in rows),
            "values_near_float32_limit": sum(r["values_near_float32_limit"] for r in rows)})
    # Content statistics ONLY on train windows of one fold, not tests/validation.
    from random import Random
    with (ROOT / f"data/finetune/fold_{args.content_fold}/samples_manifest.csv").open() as stream:
        training = [r for r in csv.DictReader(stream) if r["split"] == "train"]
    by_file = defaultdict(list)
    for row in training:
        by_file[row["drive_id"]].append(row)
    rng = Random(args.seed)
    selected = [r for key in sorted(by_file) for r in rng.sample(sorted(by_file[key], key=lambda x: x["sample_path"]), min(args.samples_per_file, len(by_file[key])))]
    content = []
    for meta in tqdm(selected, desc="[corpus] train-only signal descriptors", unit="window"):
        with np.load(ROOT / meta["sample_path"], allow_pickle=False) as payload:
            data = payload["data"]
        content.append({"sample_path": meta["sample_path"], "drive_id": meta["drive_id"], "family": meta["family"],
                        "configuration": meta["configuration"], "title": meta["title"],
                        "nominal_frequency_hz": nominal_frequency(meta["title"]), **content_descriptors(data)})
    content_families = {}
    excluded = {"sample_path", "drive_id", "family", "configuration", "title", "nominal_frequency_hz"}
    for family in sorted({r["family"] for r in content}):
        values = [r for r in content if r["family"] == family]
        keys = [k for k in values[0] if k not in excluded]
        content_families[family] = {"sampled_windows": len(values), "sampled_files": len({r["drive_id"] for r in values}),
            "sampled_configurations": len({r["configuration"] for r in values}),
            "median_descriptors": {key: float(np.median([r[key] for r in values if r[key] is not None])) for key in keys}}
    summary = {"raw_files": len(files), "configuration_groups": sum(r["configuration_groups"] for r in family_rows),
               "local_raw_bytes": sum(r["local_bytes"] for r in files), "prepared": prepared_summary,
               "families": family_rows, "exact_signal_payload_duplicate_sets": [ids for ids in duplicates.values() if len(ids) > 1],
               "content_split": "train", "content_fold": args.content_fold, "content_windows": len(content),
               "content_families": content_families,
               "timestamp_note": "Source/time assumed epoch seconds, Origin temporal ms like Spacing; group clock unions are not independent-acquisition counts",
               "extreme_value_note": "abs>1e20 is a diagnostic alert, not an automatic invalid-data criterion"}
    output.mkdir(parents=True)
    write_csv(output / "files.csv", files)
    write_csv(output / "families.csv", family_rows)
    write_csv(output / "train_content.csv", content)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "COMPLETE").write_text("Read-only corpus audit complete; no data or models modified.\n")
    print(f"[corpus] complete: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
