#!/usr/bin/env python3
"""Assign conservative cross-validation folds by FEBUS family and configuration."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=Path("data/manifests/febus_drive_inventory.csv"))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/febus_folds.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/manifests/febus_folds_summary.json"))
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")
    with args.inventory.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row["group_id"])
    fold_by_group: dict[str, int] = {}
    for family, group_ids in groups.items():
        for index, group_id in enumerate(sorted(set(group_ids), key=natural_key)):
            fold_by_group[group_id] = index % args.folds
    for row in rows:
        row["fold"] = str(fold_by_group[row["group_id"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0].keys()])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"folds": args.folds, "groups_by_family": {family: sorted(set(values), key=natural_key) for family, values in groups.items()}, "fold_by_group": fold_by_group}
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[finetune-folds] success: assigned {len(fold_by_group)} configuration group(s) across {args.folds} fold(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
