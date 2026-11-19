"""Train-file-check RD curves with independently loaded receivers, no fitting.

Checkpoint selection precedes this expanded check. Step multipliers probe rate
calibration post hoc; they do not create a new independent validation set.
"""
import argparse
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from compression.lifting_stream import decode_frame, encode_frame, load_receiver, receiver_bytes
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.prospective_metrics import PhysicalAccumulator
from finetuning.prospective_spectral import merge_spectral, spectral_result, spectral_statistics
from finetuning.run_lifting_study import macro_scores
from finetuning.run_spatial_opportunity import load, select


def select_receiver(study):
    protocol = json.loads((study / "protocol.json").read_text())
    target = protocol["target_proxy_bpc"]
    results = [json.loads(p.read_text()) for p in study.glob("check_*.json")]
    selected = min(results, key=lambda r: r["macro_families"] ** 2 + .1 * r["stream_bpc"] / target)
    return study / f"receiver_{selected['update']:06d}.pt", selected["update"]


def summarize(rows, reconstructions, frames, scales, maps, setup_bytes):
    metrics = defaultdict(PhysicalAccumulator)
    costs = defaultdict(lambda: {"frame_bytes": 0, "cells": 0})
    spectra = defaultdict(dict)
    for row, (raw, predicted), frame in zip(rows, reconstructions, frames):
        key = row["group_id"], row["drive_id"]
        costs[key]["frame_bytes"] += len(frame)
        costs[key]["cells"] += raw.size
        if "dt_s" in row:
            merge_spectral(spectra[key], spectral_statistics(
                raw, predicted, float(row["dt_s"]), scales[key[0]]))
        for channel, source_channel in enumerate(maps[row["channel_map_id"]]["source_channel_indices"]):
            metrics[key + (source_channel,)].update(raw[channel], predicted[channel])
    records = []
    for key, cost in sorted(costs.items()):
        channels = [v.result(scales[key[0]]) for k, v in metrics.items() if k[:2] == key]
        records.append({"group_id": key[0], "drive_id": key[1], **cost,
                        "nrmse_star_macro_channels": float(np.mean([c["nrmse_star"] for c in channels])),
                        "p90_channel_nrmse_star": float(np.quantile([c["nrmse_star"] for c in channels], .9)),
                        "correlation_defined_channels": sum(c["correlation_defined"] for c in channels),
                        "spectral": spectral_result(spectra[key]),
                        "channels": channels})
    cells = sum(v["cells"] for v in costs.values())
    frame_bytes = sum(v["frame_bytes"] for v in costs.values())
    return {"records": records, **macro_scores(records), "cells": cells,
            "frame_bytes": frame_bytes, "setup_bytes": setup_bytes,
            "stream_bytes": frame_bytes + setup_bytes, "stream_bpc": 8 * (frame_bytes + setup_bytes) / cells}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--study", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    scales = json.loads(args.calibration.read_text())["scales"]
    rows = [r for r in select(args.corpus) if r["role"] == "check"]
    maps = {m["channel_map_id"]: m for m in
            map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    setup = [{"drive_id": r["drive_id"], "dataset": r["dataset"],
              "block": int(r["block_index"]), "start": int(r["time_start"]),
              "dt_s": float(r["dt_s"]), "dx_m": float(r["dx_m"]),
              "source_channel_indices": maps[r["channel_map_id"]]["source_channel_indices"]}
             for r in rows]
    setup_wire = zlib.compress(json.dumps(setup, sort_keys=True, separators=(",", ":")).encode(), 6)
    (args.output / "transport_setup.zlib").write_bytes(setup_wire)
    candidates = [(study, *select_receiver(study)) for study in args.study]
    multipliers = [.5, .75, 1., 1.5, 2., 3., 4.]
    write_json(args.output / "protocol.json", {
        "kind": "expanded_train_file_checks_posthoc_rd_v1", "signal_split": "train",
        "samples": [r["sample_path"] for r in rows], "multipliers": multipliers,
        "candidates": [{"study": str(study), "receiver": str(path), "update": update,
                        "sha256": sha256(path)} for study, path, update in candidates],
        "manifest_sha256": sha256(args.corpus / "samples_manifest.csv"),
        "calibration_sha256": sha256(args.calibration), "script_sha256": sha256(__file__),
        "sources": {str(p): sha256(p) for p in [
            Path("finetuning/learned_lifting.py"), Path("finetuning/prospective_metrics.py"),
            Path("finetuning/prospective_spectral.py"), Path("compression/lifting_stream.py"),
            Path("compression/research_coefficient_stream.py")]},
        "selection": "frozen original 10-window criterion, not reselected on 129-window results",
        "limits": "largest tiles, reused training configurations; not independent validation; multipliers post hoc",
        "spectra": "Hann, channel centered, one-sided unnormalized FFT sums; phase coverage requires both bin powers above training-reference-scale squared times Hann energy",
        "acceptance": "no interpolation claim; compare measured discrete Pareto points only"})
    results = []
    for index, (study, path, update) in enumerate(candidates):
        for multiplier in multipliers:
            model = load_receiver(path.read_bytes(), args.device)
            model.log_step.add_(math.log(multiplier))
            data = receiver_bytes(model)
            receiver = load_receiver(data, args.device)
            directory = args.output / f"candidate{index}_step{multiplier:g}"
            directory.mkdir()
            (directory / "receiver.pt").write_bytes(data)
            frames, reconstructions = [], []
            for row in rows:
                raw = load(args.corpus, row)
                frame = encode_frame(raw, model)
                predicted = decode_frame(frame, receiver)
                frames.append(frame)
                reconstructions.append((raw, predicted))
                (directory / (Path(row["sample_path"]).stem + ".frc")).write_bytes(frame)
            result = summarize(rows, reconstructions, frames, scales, maps, len(setup_wire))
            result.update(study=str(study), update=update, multiplier=multiplier,
                          receiver_bytes=len(data), candidate=index,
                          directory=str(directory))
            write_json(directory / "results.json", result)
            results.append({k: v for k, v in result.items() if k != "records"})
            print(json.dumps({k: result[k] for k in
                              ("candidate", "multiplier", "macro_families", "stream_bpc")}), flush=True)
    write_json(args.output / "results.json", results)
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
