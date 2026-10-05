#!/usr/bin/env python3
"""Extract one full-spatial FEBUS window as a [block, channel, time] array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def dataset_candidates(handle: h5py.File) -> list[tuple[str, h5py.Dataset]]:
    found: list[tuple[str, h5py.Dataset]] = []

    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
            found.append((name, obj))

    handle.visititems(visit)
    return found


def choose_dataset(handle: h5py.File, requested: str) -> tuple[str, h5py.Dataset, str]:
    candidates = dataset_candidates(handle)
    if requested != "auto":
        needles = ["strain rate", "strain_rate"] if requested == "strain-rate" else ["strain"]
        matching = [(name, ds) for name, ds in candidates if any(needle in name.lower() for needle in needles)]
        if not matching:
            raise ValueError(f"No dataset matching {requested!r}")
        name, dataset = matching[0]
        return name, dataset, requested

    rate = [(name, ds) for name, ds in candidates if "strain rate" in name.lower() or "strain_rate" in name.lower()]
    strain = [(name, ds) for name, ds in candidates if "strain" in name.lower() and "rate" not in name.lower()]
    name, dataset = (rate or strain or candidates)[0]
    return name, dataset, "strain-rate" if rate else "strain"


def to_blocks(dataset: h5py.Dataset) -> np.ndarray:
    values = np.asarray(dataset, dtype=np.float32)
    if dataset.ndim == 3:
        # FEBUS datasets are [block, time, distance].
        return values.transpose(0, 2, 1)
    if values.shape[0] >= values.shape[1]:
        return values.T[np.newaxis, ...]
    return values[np.newaxis, ...]


def spatial_indices(config: dict[str, Any], channels: int) -> tuple[int, int, float, float]:
    source_start = float(config.get("source_distance_start_m", 0.0))
    source_end = float(config.get("source_distance_end_m", channels - 1))
    requested_start = float(config.get("spatial_start_m", source_start))
    requested_end = float(config.get("spatial_end_m", source_end))
    if not source_start <= requested_start <= requested_end <= source_end:
        raise ValueError("Requested spatial range is outside the configured source range")
    scale = (channels - 1) / (source_end - source_start) if channels > 1 else 0.0
    start = int(round((requested_start - source_start) * scale))
    end = int(round((requested_end - source_start) * scale)) + 1
    return start, end, requested_start, requested_end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    time_samples = int(config.get("time_samples", 1024))
    time_offset = int(config.get("source_time_offset", 0))
    source_block_samples = int(config.get("source_block_samples", time_samples))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    print(f"[extract] starting: {len(config['files'])} file(s), {time_samples} time sample(s) -> {args.output_dir}")

    for item in config["files"]:
        filename = item.get("local_filename", f"{item['id']}.h5")
        path = args.raw_dir / filename
        if not path.exists():
            print(f"[extract] skipped missing: {path}")
            continue
        print(f"[extract] reading: {path}")
        with h5py.File(path, "r") as handle:
            dataset_name, dataset, variable = choose_dataset(handle, str(config.get("data_type", "auto")))
            full_data = to_blocks(dataset)
            if full_data.shape[2] < time_offset + source_block_samples:
                print(f"[extract] skipped incompatible: {path}, time dimension={full_data.shape[2]}")
                continue
            start_channel, end_channel, spatial_start, spatial_end = spatial_indices(config, full_data.shape[1])
            # Rebuild the processed DAS timeline by retaining the same interval from every raw block.
            selected = full_data[:, start_channel:end_channel, time_offset:time_offset + source_block_samples]
            data = np.concatenate(list(selected), axis=1)[np.newaxis, ...]
            if data.shape[2] < time_samples:
                print(f"[extract] skipped incompatible: {path}, reconstructed timeline={data.shape[2]}")
                continue
            data = data[:, :, :time_samples]
            metadata = {
                **item,
                "sample_id": path.stem,
                "source_path": str(path),
                "dataset": dataset_name,
                "variable": variable,
                "shape_source": list(dataset.shape),
                "shape_sample": list(data.shape),
                "num_blocks": int(data.shape[0]),
                "source_blocks": int(full_data.shape[0]),
                "block_indices": list(range(full_data.shape[0])),
                "source_time_offset": time_offset,
                "source_block_samples": source_block_samples,
                "channels": int(data.shape[1]),
                "time_samples": int(data.shape[2]),
                "spatial_start_m": spatial_start,
                "spatial_end_m": spatial_end,
                "start_channel": start_channel,
                "end_channel": end_channel,
                "normalization": "none",
            }
            for key, value in dataset.parent.attrs.items():
                if key in {"Spacing", "Origin", "Extent", "GaugeLength", "BlockRate", "SamplingRate"}:
                    metadata[key] = np.asarray(value).tolist()
            spacing = np.asarray(dataset.parent.attrs.get("Spacing", []), dtype=float).reshape(-1)
            origin = np.asarray(dataset.parent.attrs.get("Origin", []), dtype=float).reshape(-1)
            extent = np.asarray(dataset.parent.attrs.get("Extent", []), dtype=float).reshape(-1)
            if spacing.size >= 2:
                metadata["dx_m"] = float(spacing[0])
                metadata["dt_s"] = float(spacing[1]) * 1e-3
            if origin.size >= 1 and extent.size >= 2 and spacing.size >= 1:
                metadata["distance_start_m"] = float(origin[0] + extent[0] * spacing[0])
                metadata["distance_end_m"] = float(origin[0] + extent[1] * spacing[0])
        sample_path = args.output_dir / f"{path.stem}.npz"
        np.savez_compressed(sample_path, data=data, metadata_json=json.dumps(metadata, sort_keys=True))
        manifest.append({"sample_path": str(sample_path), **metadata})
        print(f"[extract] complete: {sample_path} ({variable}, {data.shape[1]} channels x {data.shape[2]} samples)")

    manifest_path = Path("results/manifests/extracted_samples.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[extract] success: wrote {len(manifest)} window(s) and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
