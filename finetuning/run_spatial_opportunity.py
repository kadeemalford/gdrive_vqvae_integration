"""Fit/check spatial opportunity on disjoint files, all inside training."""
import argparse
import csv
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.prospective_metrics import PhysicalAccumulator
from finetuning.robust_escape import encode_robust
from finetuning.spatial_opportunity import decode_frame, encode_frame, neighbors


def select(corpus):
    groups = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader((corpus / "samples_manifest.csv").open()):
        if row["split"] == "train":
            groups[row["group_id"]][row["drive_id"]].append(row)
    selected = []
    for group, files in sorted(groups.items()):
        ids = sorted(files)
        if len(ids) < 2:
            raise ValueError("Disjoint training files required")
        for index, file_id in enumerate(ids):
            # Three deterministic quantile positions in the largest contiguous tile.
            candidates = files[file_id]
            largest = max(int(r["channels"]) for r in candidates)
            candidates = sorted([r for r in candidates if int(r["channels"]) == largest],
                                key=lambda r: r["sample_path"])
            for position in sorted({0, len(candidates) // 2, len(candidates) - 1}):
                selected.append({**candidates[position],
                                 "role": "fit" if index < (len(ids) + 1) // 2 else "check"})
    return selected


def load(corpus, row):
    with np.load(corpus / "train" / Path(row["sample_path"]).name, allow_pickle=False) as payload:
        return payload["data"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scales = json.loads(args.calibration.read_text())["scales"]
    rows = select(args.corpus)
    write_json(args.output / "protocol.json", {
        "version": "decoded-anchor-opportunity-v1", "selection": rows,
        "sources": {str(p): sha256(p) for p in
                    [Path(__file__), Path("finetuning/prospective_metrics.py"),
                     Path("finetuning/spatial_opportunity.py"),
                     Path("compression/research_coefficient_stream.py")]},
        "manifest_sha256": sha256(args.corpus / "samples_manifest.csv"),
        "calibration_sha256": sha256(args.calibration),
        "steps": [.5, 1., 2., 4., 8.], "heldout_signals": False,
        "limits": "Largest tiles only, train checks not independent validation; zlib int32 planes baseline",
        "transport": "independent stored windows, no invented temporal continuity",
        "cost": "every frame includes anchors, innovations, coefficient, side, shape and CRC",
        "fit": "single scalar per group by ordinary-cell least squares on original neighbors"})
    moments = defaultdict(lambda: np.zeros(2))
    for row in rows:
        if row["role"] != "fit":
            continue
        packet = encode_robust(load(args.corpus, row))
        x = packet.values.astype(np.float64)
        if len(x) < 2:
            continue
        p = neighbors(x[::2], len(x) // 2)
        mask = packet.valid[1::2] & packet.valid[::2][:len(x) // 2]
        right_valid = packet.valid[::2][np.minimum(np.arange(len(x) // 2) + 1, len(x[::2]) - 1)]
        mask &= right_valid
        moments[row["group_id"]] += [float((p[mask] * x[1::2][mask]).sum()),
                                     float((p[mask] ** 2).sum())]
    coefficients = {g: float(np.clip(a / b, -2, 2)) if b else 0.
                    for g, (a, b) in moments.items()}
    write_json(args.output / "coefficients.json", coefficients)
    metrics = defaultdict(PhysicalAccumulator)
    costs = defaultdict(lambda: {"bytes": 0, "cells": 0, "frames": 0})
    maps = {m["channel_map_id"]: m for m in
            map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    setup = defaultdict(list)
    for row in rows:
        if row["role"] == "check":
            mapping = maps[row["channel_map_id"]]
            setup[row["drive_id"]].append({
                "dataset": row["dataset"], "block": int(row["block_index"]),
                "start": int(row["time_start"]), "dt_s": float(row["dt_s"]),
                "dx_m": float(row["dx_m"]),
                "source_channel_indices": mapping["source_channel_indices"]})
    setup_sizes = {file_id: len(zlib.compress(json.dumps(value, sort_keys=True).encode(), 6))
                   for file_id, value in setup.items()}
    write_json(args.output / "transport_setup.json", dict(setup))
    optimistic = defaultdict(lambda: np.zeros(2))
    frame_dir = args.output / "frames"
    frame_dir.mkdir()
    for row in rows:
        if row["role"] != "check":
            continue
        raw = load(args.corpus, row)
        coefficient = coefficients.get(row["group_id"], 0.)
        packet = encode_robust(raw)
        x = packet.values.astype(np.float64)
        if len(x) > 1:
            prediction = coefficient * neighbors(x[::2], len(x) // 2)
            mask = packet.valid[1::2]
            optimistic[row["group_id"]] += [
                float(((x[1::2] - prediction)[mask] ** 2).sum()),
                float((x[1::2][mask] ** 2).sum())]
        for step in (.5, 1., 2., 4., 8.):
            for predictive in (False, True):
                frame = encode_frame(raw, step, coefficient, predictive)
                reconstruction = decode_frame(frame)
                arm = "decoded_anchors" if predictive else "independent"
                key = (arm, step, row["group_id"], row["drive_id"])
                cost = costs[key]
                cost["bytes"] += len(frame)
                cost["cells"] += raw.size
                cost["frames"] += 1
                for channel in range(len(raw)):
                    source_channel = maps[row["channel_map_id"]]["source_channel_indices"][channel]
                    metrics[key + (source_channel,)].update(raw[channel], reconstruction[channel])
                if step == 2.:
                    name = f"{Path(row['sample_path']).stem}_{arm}.frc"
                    (frame_dir / name).write_bytes(frame)
    records = []
    for key, cost in sorted(costs.items()):
        channels = [v.result(scales[key[2]]) for k, v in metrics.items() if k[:4] == key]
        records.append({"arm": key[0], "step": key[1], "group_id": key[2],
                        "drive_id": key[3], **cost, "setup_bytes": setup_sizes[key[3]],
                        "bpc": 8 * (cost["bytes"] + setup_sizes[key[3]]) / cost["cells"],
                        "nrmse_star_macro_channels": float(np.mean([c["nrmse_star"] for c in channels])),
                        "channels": channels})
    write_json(args.output / "results.json", {
        "records": records,
        "optimistic_original_context_innovation_energy_ratio":
            {g: float(a / b) if b else None for g, (a, b) in optimistic.items()},
        "interpretation": "Optimistic ratio is not a byte saving; only decoded frames are codec measurements"})
    (args.output / "COMPLETE").touch()
    print(f"Completed {len(records)} file/rate/arm records", flush=True)


if __name__ == "__main__":
    main()
