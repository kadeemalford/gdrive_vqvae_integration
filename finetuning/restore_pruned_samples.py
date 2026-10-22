#!/usr/bin/env python3
"""Restore archived prepared arrays to original paths from raw HDF5 and manifest.

Existing files are never overwritten. Data/metadata ZIP record CRC and sizes are
checked against the pruning inventory. CRC is an integrity check, not a secure
hash; raw files/manifest remain the authoritative reproducible sources.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import tempfile
import zipfile

import h5py
import numpy as np
from tqdm import tqdm

try:
    from .prepare_finetune_samples import choose_dataset, blocks
    from .channel_geometry import selected_indices
except ImportError:
    from prepare_finetune_samples import choose_dataset, blocks
    from channel_geometry import selected_indices

ROOT = Path(__file__).resolve().parents[1]
INTEGER_FIELDS = {"block_index", "time_start", "channels", "time_samples", "channel_map_version"}
FLOAT_FIELDS = {"dt_s", "dx_m"}
BOOLEAN_FIELDS = {"spatial_rows_contiguous"}


def metadata_from_row(row):
    metadata = {}
    for key, value in row.items():
        if key == "sample_path":
            continue
        if key in INTEGER_FIELDS:
            value = int(value)
        elif key in FLOAT_FIELDS:
            value = float(value) if value else None
        elif key in BOOLEAN_FIELDS:
            if value.lower() not in ("true", "false"):
                raise ValueError("Invalid archived boolean")
            value = value.lower() == "true"
        metadata[key] = value
    return metadata


def verify_archive(path, expected):
    with zipfile.ZipFile(path) as archive:
        actual = {entry.filename: {"crc32": entry.CRC, "uncompressed_bytes": entry.file_size}
                  for entry in archive.infolist() if not entry.is_dir()}
    if actual != expected:
        raise ValueError(f"Regenerated array/metadata does not match archived records: {path}")


def restore(root, folder, limit=0):
    root, folder = root.resolve(), folder.resolve()
    if not folder.is_relative_to((root / "data/finetune").resolve()):
        raise ValueError("Prepared folder outside repository data/finetune")
    marker = folder / "PRUNED_ARRAYS.json"
    info = json.loads(marker.read_text())
    logs = info.get("audit_logs", [info["audit_log"]])
    group = str(folder.relative_to(root))
    expected = {}
    for log in logs:
        with Path(log).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["kind"] == "prepared_npz" and row["group"] == group:
                    expected[row["path"]] = row["zip_records"]
    with (folder / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    missing = defaultdict(list)
    for row in rows:
        path = (root / row["sample_path"]).resolve()
        if not path.is_relative_to(folder) or path.suffix != ".npz":
            raise ValueError("Archived sample path outside prepared folder")
        if not path.exists():
            relative = str(path.relative_to(root))
            if relative not in expected:
                raise ValueError("Missing sample has no pruning inventory; refuse guessed regeneration")
            missing[row["drive_id"]].append(row)
    restored = 0
    for file_id, file_rows in tqdm(sorted(missing.items()), desc="[restore] raw to archived arrays", unit="file"):
        if limit and restored >= limit:
            break
        raw = root / f"data/finetune/raw/{file_id}.h5"
        if not raw.resolve().is_relative_to((root / "data/finetune/raw").resolve()):
            raise ValueError("Invalid raw source path")
        with h5py.File(raw, "r") as handle:
            name, dataset = choose_dataset(handle)
            source_blocks = blocks(dataset)
        spatial_cache = {}
        for row in sorted(file_rows, key=lambda row: row["sample_path"]):
            if limit and restored >= limit:
                break
            if row["dataset"] != name:
                raise ValueError("Raw dataset selection changed")
            block_index, count = int(row["block_index"]), int(row["channels"])
            key = block_index, count
            if key not in spatial_cache:
                block = source_blocks[block_index]
                indices = selected_indices(np.isfinite(block).all(axis=1), count)
                if len(indices) != count:
                    raise ValueError("Raw finite-channel mask changed")
                spatial_cache[key] = block[indices]
            start, length = int(row["time_start"]), int(row["time_samples"])
            sample = spatial_cache[key][:, start:start + length]
            if sample.shape != (count, length):
                raise ValueError("Archived sample bounds invalid")
            path = root / row["sample_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".restore-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, data=sample, metadata_json=json.dumps(metadata_from_row(row), sort_keys=True))
                    stream.flush()
                    os.fsync(stream.fileno())
                verify_archive(temporary, expected[str(path.relative_to(root))])
                if path.exists():
                    raise FileExistsError("Sample appeared during restoration; refusing overwrite")
                os.replace(temporary, path)
                restored += 1
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        del source_blocks, spatial_cache
    remaining = sum(not (root / row["sample_path"]).is_file() for row in rows)
    info.update(status="restored" if not remaining else "pruned", missing_manifest_samples=remaining,
                expected_manifest_samples=len(rows), restored_in_last_invocation=restored)
    marker.write_text(json.dumps(info, indent=2) + "\n")
    return {"samples_dir": str(folder), "restored": restored, "missing": remaining}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional small verification run; 0 restores all missing arrays")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("Limit must be nonnegative")
    folder = args.samples_dir.resolve()
    lock = (folder / ".restore.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(restore(ROOT, folder, args.limit), indent=2))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
