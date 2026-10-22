#!/usr/bin/env python3
"""Extract finite native-geometry FEBUS windows for one cross-validation fold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

try:
    from .channel_geometry import select_spatial, map_identifier
except ImportError:
    from channel_geometry import select_spatial, map_identifier


def choose_dataset(handle: h5py.File) -> tuple[str, h5py.Dataset]:
    found: list[tuple[str, h5py.Dataset]] = []
    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
            found.append((name, obj))
    handle.visititems(visit)
    rate = [(name, value) for name, value in found if "strain rate" in name.lower() or "strain_rate" in name.lower()]
    return (rate or found)[0]


def blocks(data: h5py.Dataset) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim == 3:
        return values.transpose(0, 2, 1)
    return values.T[None, ...] if values.shape[0] >= values.shape[1] else values[None, ...]


def partition(fold: int, test_fold: int, val_fold: int, family: str, test_family: str | None) -> str:
    if test_family:
        if family == test_family:
            return "test"
        return "val" if fold == val_fold else "train"
    if fold == test_fold:
        return "test"
    if fold == val_fold:
        return "val"
    return "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/febus_folds.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/finetune/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/finetune/fold_0"))
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--val-fold", type=int, default=1)
    parser.add_argument("--test-family", help="Optional family-wide held-out test split, for example doped.")
    parser.add_argument("--time-samples", type=int, default=1000)
    parser.add_argument("--windows-per-block", type=int, default=20)
    parser.add_argument("--max-channels", type=int, default=512)
    args = parser.parse_args()
    if args.test_fold == args.val_fold:
        raise ValueError("test and validation folds must differ")
    if min(args.time_samples, args.windows_per_block, args.max_channels) < 1:
        raise ValueError("Window length/count and channel limit must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new/empty preparation directory; existing data is preserved")
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    manifest_rows: list[dict[str, object]] = []
    counts = {"train": 0, "val": 0, "test": 0}
    channel_maps = {}
    for row in rows:
        source = args.raw_dir / f"{row['drive_id']}.h5"
        if not source.exists():
            raise FileNotFoundError(f"Download missing: {source}")
        split = partition(int(row["fold"]), args.test_fold, args.val_fold, row["family"], args.test_family)
        with h5py.File(source, "r") as handle:
            dataset_name, dataset = choose_dataset(handle)
            source_blocks = blocks(dataset)
            spacing = np.asarray(dataset.parent.attrs.get("Spacing", []), dtype=float).reshape(-1)
            attributes = {key: dataset.parent.attrs[key] for key in ("Spacing", "Origin", "Extent") if key in dataset.parent.attrs}
        for block_index, block in enumerate(source_blocks):
            spatial, mapping = select_spatial(block, args.max_channels, attributes)
            if spatial.shape[0] == 0 or spatial.shape[1] < args.time_samples:
                continue
            mapping = {"drive_id": row["drive_id"], "dataset": dataset_name, "block_index": block_index, **mapping}
            map_id = map_identifier(mapping)
            channel_maps[map_id] = {"channel_map_id": map_id, **mapping}
            starts = np.linspace(0, spatial.shape[1] - args.time_samples, num=args.windows_per_block, dtype=int)
            for window_index, start in enumerate(np.unique(starts)):
                sample = spatial[:, start:start + args.time_samples]
                name = f"{row['drive_id']}_b{block_index:02d}_w{window_index:02d}.npz"
                destination = args.output_dir / split / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                metadata = {**row, "split": split, "dataset": dataset_name, "block_index": block_index, "time_start": int(start), "channels": int(sample.shape[0]), "time_samples": int(sample.shape[1]), "dt_s": float(spacing[1]) * 1e-3 if spacing.size >= 2 else None, "dx_m": float(spacing[0]) if spacing.size else None}
                metadata.update(channel_map_id=map_id, channel_map_version=1, spatial_rows_contiguous=mapping["selected_rows_contiguous"])
                np.savez_compressed(destination, data=sample, metadata_json=json.dumps(metadata, sort_keys=True))
                manifest_rows.append({"sample_path": str(destination), **metadata})
                counts[split] += 1
    manifest_path = args.output_dir / "samples_manifest.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]) if manifest_rows else ["sample_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (args.output_dir / "channel_maps.jsonl").open("w") as stream:
        for map_id in sorted(channel_maps):
            stream.write(json.dumps(channel_maps[map_id], sort_keys=True, allow_nan=False) + "\n")
    print(f"[finetune-prepare] success: train={counts['train']} val={counts['val']} test={counts['test']} -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
