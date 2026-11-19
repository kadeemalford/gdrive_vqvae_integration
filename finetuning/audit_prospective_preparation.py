"""Train-only preparation audit and calibration for report 90 P0."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from finetuning.prospective_metrics import VERSION, PhysicalAccumulator
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/finetune/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.corpus / "samples_manifest.csv"
    rows = [r for r in csv.DictReader(manifest.open()) if r["split"] == "train"]
    maps_path = args.corpus / "channel_maps.jsonl"
    maps = {m["channel_map_id"]: m for m in map(json.loads, maps_path.open())}
    protocol = {
        "version": VERSION, "manifest_sha256": sha256(manifest),
        "script_sha256": sha256(__file__), "signal_split": "train",
        "metrics_sha256": sha256(Path(__file__).with_name("prospective_metrics.py")),
        "channel_maps_sha256": sha256(maps_path),
        "calibration": "per group median of positive per-window ordinary physical std, times 0.01",
        "calibration_status": "numerical development scale, not an application tolerance",
        "group_weighting": "pool per physical channel/acquisition, then macro channels/files/groups/families",
        "coverage": "existing retained corpus; no expansion, denoising or resampling",
        "masks": {"ordinary": "packet.valid", "escape": "~packet.valid",
                  "outside_coverage": "original channel maps", "padding": "none in stored NPZ"},
        "transport": "independent stored-window frames; gaps explicitly retained, not continuous 1 s",
        "historical_metrics_changed": False,
        "unseen_configuration_scale": "median of calibrated training group scales in the same family; no evaluation fitting",
    }
    write_json(args.output / "protocol.json", protocol)
    group_scales = defaultdict(list)
    inventory = defaultdict(lambda: {"windows": 0, "cells": 0, "escape_cells": 0,
                                    "constant_windows": 0, "constant_channels": 0,
                                    "side_bytes": 0, "max_roundtrip_error": 0., "sse": 0.,
                                    "ordinary_cells": 0})
    files = {}
    with (args.output / "windows.jsonl").open("w") as output:
        for row in rows:
            path = args.corpus / "train" / Path(row["sample_path"]).name
            with np.load(path, allow_pickle=False) as payload:
                raw = payload["data"]
            mapping = maps[row["channel_map_id"]]
            indices = np.asarray(mapping["source_channel_indices"])
            if (raw.shape != (int(row["channels"]), int(row["time_samples"]))
                    or len(indices) != len(raw) or np.any(np.diff(indices) != 1)):
                raise ValueError("Corpus geometry violates the retained contiguous-tile contract")
            packet = encode_robust(raw)
            restored = decode(packet.values, packet.side_stream)
            if raw[~packet.valid].tobytes() != restored[~packet.valid].tobytes():
                raise AssertionError("Escaped bits changed")
            stats = PhysicalAccumulator()
            stats.update(raw[packet.valid], restored[packet.valid])
            scale = float(raw[packet.valid].astype(np.float64).std())
            if scale > 0:
                group_scales[row["group_id"]].append(scale)
            result = inventory[row["drive_id"]]
            result["windows"] += 1
            result["cells"] += raw.size
            result["ordinary_cells"] += stats.count
            result["escape_cells"] += int((~packet.valid).sum())
            bits = raw.view(np.uint32)
            constant = bool(np.all(bits == bits.flat[0]))
            constant_channels = int(np.all(bits == bits[:, :1], axis=1).sum())
            result["constant_windows"] += int(constant)
            result["constant_channels"] += constant_channels
            result["side_bytes"] += len(packet.side_stream)
            result["sse"] += stats.sse
            result["max_roundtrip_error"] = max(result["max_roundtrip_error"], stats.max_error)
            output.write(json.dumps({"sample": path.name, "sha256": sha256(path),
                                     "group_id": row["group_id"], "shape": list(raw.shape),
                                     "std": scale, "constant_bits": constant,
                                     "constant_channels": constant_channels,
                                     "side_bytes": len(packet.side_stream),
                                     "roundtrip_max_error": stats.max_error}) + "\n")
            files.setdefault(row["drive_id"], row)
    scales = {group: float(np.median(values) * .01) for group, values in group_scales.items()}
    missing = sorted({r["group_id"] for r in rows} - scales.keys())
    if missing:
        raise ValueError(f"No positive training calibration for groups: {missing}")
    family_groups = defaultdict(list)
    for group, scale in scales.items():
        family_groups[group.split("/")[0]].append(scale)
    write_json(args.output / "reference_scales.json", {
        "version": VERSION, "unit": "native stored product unit; see physical_manifest",
        "scales": scales,
        "family_scales_for_unseen_configurations": {
            family: float(np.median(values)) for family, values in family_groups.items()},
        "protocol_sha256": sha256(args.output / "protocol.json")})
    with (args.output / "physical_manifest.jsonl").open("w") as output:
        for file_id, row in sorted(files.items()):
            source = args.raw_dir / f"{file_id}.h5"
            with h5py.File(source, "r") as handle:
                dataset = handle[row["dataset"]]
                attributes = {}
                node = dataset
                while True:
                    attributes[node.name] = {
                        k: np.asarray(v).tolist() if np.asarray(v).dtype.kind not in "SO"
                        else str(v) for k, v in node.attrs.items()}
                    if node.name == "/":
                        break
                    node = node.parent
                output.write(json.dumps({
                    "drive_id": file_id, "group_id": row["group_id"], "family": row["family"],
                    "dataset": row["dataset"], "source_dtype": str(dataset.dtype),
                    "source_shape": list(dataset.shape), "dt_s": float(row["dt_s"]),
                    "dx_m": float(row["dx_m"]), "attributes_by_path": attributes,
                    "unit_policy": "verbatim attributes; no guessed physical conversion",
                }, default=str) + "\n")
    totals = {key: sum(v[key] for v in inventory.values())
              for key in ("windows", "cells", "ordinary_cells", "escape_cells",
                          "constant_windows", "constant_channels", "side_bytes", "sse")}
    totals["max_roundtrip_error"] = max(v["max_roundtrip_error"] for v in inventory.values())
    totals["roundtrip_rmse"] = float(np.sqrt(totals["sse"] / totals["ordinary_cells"]))
    write_json(args.output / "audit.json", {"totals": totals, "files": dict(inventory),
                                         "heldout_signals_read": False})
    (args.output / "COMPLETE").touch()
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
