"""Metadata-only routing and manifest-safe domain selection, never error-based."""
import csv
from pathlib import Path

FAMILIES = ("doped", "multicore", "multimode", "single_mode")


def route_label(family):
    return f"expert_{family}" if family in FAMILIES else "shared"


def family_paths(paths, manifest, family, split):
    if family not in FAMILIES or split not in ("train", "val"):
        raise ValueError("Known family and train/val split required")
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    keyed = {Path(row["sample_path"]).resolve(): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Repeated manifest sample path")
    result = []
    for path in paths:
        row = keyed[path.resolve()]
        if row["split"] != split:
            raise ValueError("Domain sample crosses split boundary")
        if row["family"] == family:
            result.append(path)
    if not result:
        raise ValueError("Requested family has no samples in this split")
    return result
