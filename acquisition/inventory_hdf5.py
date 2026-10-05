#!/usr/bin/env python3
"""Inventory HDF5 structure and metadata before extracting samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/manifests/hdf5_inventory.csv"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows: list[dict[str, Any]] = []
    print(f"[inventory] starting: {len(config['files'])} file(s) from {args.raw_dir}")
    for item in config["files"]:
        path = args.raw_dir / item.get("local_filename", f"{item['id']}.h5")
        if not path.exists():
            rows.append({"file_id": item["id"], "path": str(path), "error": "missing"})
            print(f"[inventory] missing: {path}")
            continue
        print(f"[inventory] inspecting: {path}")
        with h5py.File(path, "r") as handle:
            def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                rows.append({
                    "file_id": item["id"],
                    "path": str(path),
                    "dataset": name,
                    "shape": json.dumps(obj.shape),
                    "dtype": str(obj.dtype),
                    "attributes": json.dumps({key: scalar(value) for key, value in obj.attrs.items()}, default=str),
                    "error": "",
                })
            handle.visititems(visit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file_id", "path", "dataset", "shape", "dtype", "attributes", "error"]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[inventory] success: wrote {len(rows)} dataset row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
