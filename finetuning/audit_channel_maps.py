#!/usr/bin/env python3
"""Recover source-index maps as external sidecars; verify old arrays without rewriting."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

try:
    from .prepare_finetune_samples import choose_dataset
    from .audit_febus_corpus import source_geometry
    from .channel_geometry import selected_indices, spatial_metadata, map_identifier
    from .analyze_sampling_results import write_csv
except ImportError:
    from prepare_finetune_samples import choose_dataset
    from audit_febus_corpus import source_geometry
    from channel_geometry import selected_indices, spatial_metadata, map_identifier
    from analyze_sampling_results import write_csv

ROOT = Path(__file__).resolve().parents[1]


def scan_finite(dataset, chunk_samples=1024):
    blocks, samples, channels, time_first = source_geometry(dataset.shape)
    finite = np.ones((blocks, channels), dtype=bool)
    for block in range(blocks):
        for start in range(0, samples, chunk_samples):
            stop = min(samples, start + chunk_samples)
            if dataset.ndim == 3:
                chunk = dataset[block, start:stop, :]
            elif time_first:
                chunk = dataset[start:stop, :]
            else:
                chunk = dataset[:, start:stop].T
            finite[block] &= np.isfinite(np.asarray(chunk, dtype=np.float32)).all(axis=0)
    return finite


def read_window(dataset, block, start, count):
    _, _, _, time_first = source_geometry(dataset.shape)
    if dataset.ndim == 3:
        return np.asarray(dataset[block, start:start + count, :], dtype=np.float32).T
    if time_first:
        return np.asarray(dataset[start:start + count, :], dtype=np.float32).T
    return np.asarray(dataset[:, start:start + count], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    with (ROOT / "data/finetune/fold_0/samples_manifest.csv").open() as stream:
        prepared = list(csv.DictReader(stream))
    by_file = defaultdict(lambda: defaultdict(list))
    for row in prepared:
        by_file[row["drive_id"]][int(row["block_index"])].append(row)
    args.output_dir.mkdir(parents=True)
    summaries, links = [], []
    total_blocks = gap_blocks = missing_rows = verified_windows = 0
    with (args.output_dir / "channel_maps.jsonl").open("w") as stream:
        for file_id, by_block in tqdm(sorted(by_file.items()), desc="[geometry] read-only channel maps", unit="file"):
            with h5py.File(ROOT / f"data/finetune/raw/{file_id}.h5", "r") as handle:
                name, dataset = choose_dataset(handle)
                _, _, channels, _ = source_geometry(dataset.shape)
                finite = scan_finite(dataset)
                attributes = {key: dataset.parent.attrs[key] for key in ("Spacing", "Origin", "Extent") if key in dataset.parent.attrs}
                for block, rows in sorted(by_block.items()):
                    indices = selected_indices(finite[block], 512)
                    if any(int(row["channels"]) != len(indices) or row["dataset"] != name for row in rows):
                        raise ValueError("Current preparation policy differs from existing manifest")
                    mapping = {"drive_id": file_id, "dataset": name, "block_index": block,
                               **spatial_metadata(indices, channels, attributes)}
                    identifier = map_identifier(mapping)
                    mapping["channel_map_id"] = identifier
                    row = min(rows, key=lambda row: int(row["time_start"]))
                    native = read_window(dataset, block, int(row["time_start"]), int(row["time_samples"]))[indices]
                    with np.load(ROOT / row["sample_path"], allow_pickle=False) as sample:
                        old = sample["data"]
                    if native.dtype != old.dtype or native.shape != old.shape or native.tobytes() != old.tobytes():
                        raise AssertionError("Recovered map changes stored sample signal")
                    stream.write(json.dumps(mapping, sort_keys=True, allow_nan=False) + "\n")
                    summaries.append({"drive_id": file_id, "family": row["family"], "configuration": row["configuration"],
                                      "block_index": block, "channel_map_id": identifier, "channels": len(indices),
                                      "rows_contiguous": mapping["selected_rows_contiguous"],
                                      "missing_source_rows_inside_selection": mapping["missing_source_rows_inside_selection"],
                                      "largest_source_index_step": mapping["largest_source_index_step"],
                                      "absolute_metadata_coordinates_available": mapping["metadata_positions_m"] is not None,
                                      "verified_sample_path": row["sample_path"]})
                    links.extend({"drive_id": file_id, "block_index": block, "time_start": r["time_start"],
                                  "time_samples": r["time_samples"], "channels": r["channels"],
                                  "canonical_sample_path": r["sample_path"], "channel_map_id": identifier} for r in rows)
                    total_blocks += 1
                    verified_windows += 1
                    gap_blocks += not mapping["selected_rows_contiguous"]
                    missing_rows += mapping["missing_source_rows_inside_selection"]
    write_csv(args.output_dir / "blocks.csv", summaries)
    write_csv(args.output_dir / "window_maps.csv", links)
    summary = {"files": len(by_file), "prepared_blocks": total_blocks, "coordinate_window_references": len(links),
               "blocks_with_internal_spatial_gaps": gap_blocks, "missing_source_row_instances_inside_selected_ranges": missing_rows,
               "bit_exact_verified_windows": verified_windows, "verification_sampling": "one earliest-time window per block, canonical fold_0 copy",
               "coordinate_assumption": "Metadata-derived positions, not independently verified calibration; source array indices are exact",
               "data_changed": False, "gpu_used": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("External channel maps recovered; no data/checkpoint changes.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
