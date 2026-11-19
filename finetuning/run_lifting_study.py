"""Pre-registered fixed versus learned lifting study, one arm/GPU per process."""
import argparse
import csv
import json
import math
import os
import random
import time
import zlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from compression.lifting_stream import decode_frame, encode_frame, load_receiver, receiver_bytes
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.learned_lifting import LiftingCodec
from finetuning.prospective_metrics import PhysicalAccumulator
from finetuning.robust_escape import encode_robust
from finetuning.run_spatial_opportunity import select


@lru_cache(maxsize=32)
def load_raw(path):
    with np.load(path, allow_pickle=False) as payload:
        return payload["data"]


def macro_scores(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record["nrmse_star_macro_channels"])
    group_scores = {key: float(np.mean(value)) for key, value in groups.items()}
    families = defaultdict(list)
    for group, value in group_scores.items():
        families[group.split("/")[0]].append(value)
    return {"groups": group_scores,
            "families": {key: float(np.mean(value)) for key, value in families.items()},
            "macro_families": float(np.mean([np.mean(v) for v in families.values()]))}


@torch.no_grad()
def evaluate(model, checks, corpus, scales, maps, output, update):
    model.eval()
    receiver_data = receiver_bytes(model)
    receiver = load_receiver(receiver_data, next(model.parameters()).device)
    (output / f"receiver_{update:06d}.pt").write_bytes(receiver_data)
    frame_directory = output / f"frames_{update:06d}"
    frame_directory.mkdir(exist_ok=True)
    metrics = defaultdict(PhysicalAccumulator)
    costs = defaultdict(lambda: {"frame_bytes": 0, "cells": 0, "frames": 0})
    replay_max = 0.
    start = time.monotonic()
    for row in checks:
        raw = load_raw(str(corpus / "train" / Path(row["sample_path"]).name))
        frame = encode_frame(raw, model)
        (frame_directory / (Path(row["sample_path"]).stem + ".frc")).write_bytes(frame)
        reconstructed = decode_frame(frame, receiver)
        packet = encode_robust(raw)
        tensor = torch.from_numpy(packet.values)[None, None].to(next(model.parameters()).device)
        hard, _ = model(tensor, hard=True)
        if not np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
            # Replay in normalized coordinates, ordinary cells only.
            from finetuning.reversible_normalization import decode
            expected = decode(hard[0, 0].cpu().numpy(), packet.side_stream)
            error = np.abs(expected.astype(np.float64) - reconstructed.astype(np.float64))
            replay_max = max(replay_max, float(error.max()))
            np.testing.assert_allclose(expected, reconstructed, rtol=1e-5, atol=1e-5)
        key = (row["group_id"], row["drive_id"])
        costs[key]["frame_bytes"] += len(frame)
        costs[key]["cells"] += raw.size
        costs[key]["frames"] += 1
        for channel, source_channel in enumerate(maps[row["channel_map_id"]]["source_channel_indices"]):
            metrics[key + (source_channel,)].update(raw[channel], reconstructed[channel])
    records = []
    for key, cost in sorted(costs.items()):
        channels = [value.result(scales[key[0]]) for k, value in metrics.items() if k[:2] == key]
        records.append({"group_id": key[0], "drive_id": key[1], **cost,
                        "nrmse_star_macro_channels": float(np.mean([c["nrmse_star"] for c in channels])),
                        "channels": channels})
    total_cells = sum(v["cells"] for v in costs.values())
    frame_bytes = sum(v["frame_bytes"] for v in costs.values())
    setup_bytes = len(zlib.compress((output / "CHECK_TRANSPORT.json").read_bytes(), 6))
    result = {"update": update, "records": records, **macro_scores(records),
              "frame_bpc": 8 * frame_bytes / total_cells,
              "setup_bytes": setup_bytes, "stream_bytes": setup_bytes + frame_bytes,
              "stream_bpc": 8 * (setup_bytes + frame_bytes) / total_cells,
              "receiver_bytes": len(receiver_data), "evaluation_seconds": time.monotonic() - start,
              "independent_receiver_max_absolute_difference": replay_max,
              "transport_setup": "common immutable CHECK_TRANSPORT.json, charged separately"}
    write_json(output / f"check_{update:06d}.json", result)
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("fixed", "learned"), required=True)
    parser.add_argument("--target-bpc", type=float, required=True)
    parser.add_argument("--updates", type=int, default=32768)
    parser.add_argument("--check-every", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2111)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Optional integration/pilot boundary within the registered budget")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA required; local training is not authorized by this runner")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    scales = json.loads(args.calibration.read_text())["scales"]
    rows = list(csv.DictReader((args.corpus / "samples_manifest.csv").open()))
    selected = select(args.corpus)
    check_files = {r["drive_id"] for r in selected if r["role"] == "check"}
    train = sorted([r for r in rows if r["split"] == "train" and r["drive_id"] not in check_files],
                   key=lambda r: r["sample_path"])
    # Integration/learning checks: one deterministic file/window per group.
    checks_by_group = {}
    for row in selected:
        if row["role"] == "check":
            checks_by_group.setdefault(row["group_id"], row)
    checks = list(checks_by_group.values())
    maps = {m["channel_map_id"]: m for m in
            map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    protocol = {
        "architecture": LiftingCodec.VERSION, "arm": args.arm, "seed": args.seed,
        "target_proxy_bpc": args.target_bpc, "updates": args.updates,
        "check_every": args.check_every, "batch_windows": 1,
        "train_windows": len(train), "train_files": len({r["drive_id"] for r in train}),
        "check_files": sorted({r["drive_id"] for r in checks}),
        "check_samples": [r["sample_path"] for r in checks],
        "training_file_exclusion": sorted(check_files),
        "manifest_sha256": sha256(args.corpus / "samples_manifest.csv"),
        "calibration_sha256": sha256(args.calibration),
        "sources": {str(p): sha256(p) for p in
                    [Path(__file__), Path("finetuning/learned_lifting.py"),
                     Path("compression/lifting_stream.py"), Path("finetuning/prospective_metrics.py"),
                     Path("compression/research_coefficient_stream.py")]},
        "optimizer": "AdamW lr=0.0003 wd=0.00001 clip=1 FP32; constant LR",
        "determinism": "torch deterministic algorithms, cuDNN deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "objective": "ordinary normalized MSE + adaptive lambda * logistic proxy bits/cell",
        "lambda": "initial .1; every 32 updates *= exp(.02*(mean_proxy/target-1)), clip [.00001,100]",
        "transform": "5 spatial levels with optional learned local residual P/U; 3 fixed temporal levels on final low; complete details",
        "quantization": "hard round forward, straight-through backward; trainable step per band",
        "selection": "lowest check macro_families**2 + .1*stream_bpc/target, including initialization",
        "extension": "none automatic beyond budget; inspect rate, learning, stability, and real bytes first",
        "heldout_signals": False, "pretraining": "none; both arms from same Haar function",
        "reference": "mature VQ compact/dense still required before frontier claim",
        "stream": "FRC1/zlib int32 planes plus side/CRC; same for both arms",
    }
    if args.resume:
        original = json.loads((args.output / "protocol.json").read_text())
        if original != protocol:
            raise ValueError("Resume protocol/source mismatch")
    else:
        write_json(args.output / "protocol.json", protocol)
        # Minimal geometry setup for checks; no unrelated full-corpus manifest charge.
        write_json(args.output / "CHECK_TRANSPORT.json", [
            {"drive_id": r["drive_id"], "dataset": r["dataset"], "block": int(r["block_index"]),
             "start": int(r["time_start"]), "dt_s": float(r["dt_s"]), "dx_m": float(r["dx_m"]),
             "source_channel_indices": maps[r["channel_map_id"]]["source_channel_indices"]}
            for r in checks])
    model = LiftingCodec(learned=args.arm == "learned").cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-5)
    update, multiplier, best, rate_sum = 0, .1, float("inf"), 0.
    if args.resume:
        state = torch.load(args.output / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        update, multiplier, best = state["update"], state["multiplier"], state["best"]
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    else:
        result = evaluate(model, checks, args.corpus, scales, maps, args.output, 0)
        best = result["macro_families"] ** 2 + .1 * result["stream_bpc"] / args.target_bpc
        (args.output / "best_receiver.pt").write_bytes(receiver_bytes(model))
    started = time.monotonic()
    cells_seen = 0
    with (args.output / "history.jsonl").open("a") as history:
        endpoint = min(args.updates, args.stop_after or args.updates)
        for update in range(update + 1, endpoint + 1):
            row = train[rng.randrange(len(train))]
            raw = load_raw(str(args.corpus / "train" / Path(row["sample_path"]).name))
            packet = encode_robust(raw)
            x = torch.from_numpy(packet.values)[None, None].cuda()
            mask = torch.from_numpy(packet.valid)[None, None].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, rate = model(x)
            distortion = (prediction - x).square()[mask].mean()
            loss = distortion + multiplier * rate
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite lifting objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            rate_sum += float(rate.detach())
            cells_seen += raw.size
            if update % 32 == 0:
                mean_rate = rate_sum / 32
                multiplier = float(np.clip(multiplier * math.exp(
                    float(np.clip(.02 * (mean_rate / args.target_bpc - 1), -1, 1))), 1e-5, 100))
                rate_sum = 0.
                record = {"update": update, "mse": float(distortion.detach()), "proxy_bpc": mean_rate,
                          "lambda": multiplier, "gradient_norm": float(norm),
                          "cells_seen_this_run": cells_seen, "elapsed_seconds": time.monotonic() - started,
                          "steps": model.steps.detach().cpu().tolist()}
                history.write(json.dumps(record, allow_nan=False) + "\n")
                history.flush()
            if update % args.check_every == 0 or update == endpoint:
                result = evaluate(model, checks, args.corpus, scales, maps, args.output, update)
                score = result["macro_families"] ** 2 + .1 * result["stream_bpc"] / args.target_bpc
                if score < best:
                    best = score
                    (args.output / "best_receiver.pt").write_bytes(receiver_bytes(model))
                temporary = args.output / "last.pt.partial"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "update": update, "multiplier": multiplier, "best": best,
                            "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state()}, temporary)
                temporary.replace(args.output / "last.pt")
                print(json.dumps({"update": update, "macro": result["macro_families"],
                                  "frame_bpc": result["frame_bpc"], "elapsed": time.monotonic() - started}),
                      flush=True)
    name = "endpoint_receiver.pt" if update == args.updates else "pilot_receiver.pt"
    (args.output / name).write_bytes(receiver_bytes(model))
    (args.output / ("BUDGET_COMPLETE" if update == args.updates else "PILOT_COMPLETE")).touch()


if __name__ == "__main__":
    main()
