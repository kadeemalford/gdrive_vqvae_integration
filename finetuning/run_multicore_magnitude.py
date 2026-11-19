"""Fit-only neural magnitude buckets; reproduce both paid multicore waveforms."""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from compression.bounded_residual_stream import audit, decode as decode_bound, encode as encode_bound
from compression.multicore_magnitude_stream import MagnitudeModel, features, pack_pair, unpack_pair
from compression.predictive_coefficient_stream import unpack_predictive
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.clock_aligned_multicore import OUTPUT as PRIOR
from finetuning.run_dense_product_study import SCALES


def load(row):
    path = PRIOR / "probes" / row["file"]
    if sha256(path) != row["sha256"]:
        raise ValueError("Frozen paired input hash changed")
    with np.load(path, allow_pickle=False) as data:
        a, b = data["left"], data["right"]
        shape = (len(row["channels"]), row["length"])
        if (a.dtype != np.float32 or b.dtype != np.float32 or a.shape != shape or b.shape != shape
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError("Invalid paired training crop")
        return a.copy(), b.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("none", "aligned", "shifted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(106)
    torch.cuda.manual_seed_all(106)
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = json.loads((PRIOR / "FROZEN_FIT.json").read_text())
    rows = frozen["probes"]
    fitting, checks = [r for r in rows if r["role"] == "fit"], [r for r in rows if r["role"] == "check"]
    groups = sorted({r["group_id"] for r in rows})
    fit_pairs, check_pairs = {r["pair_index"] for r in fitting}, {r["pair_index"] for r in checks}
    if (len(fitting) != 30 or len(checks) != 24 or len(groups) != 2 or len(fit_pairs) != 10
            or len(check_pairs) != 8 or fit_pairs & check_pairs
            or {r[k] for r in fitting for k in ("left_id", "right_id")}
            & {r[k] for r in checks for k in ("left_id", "right_id")}):
        raise ValueError("Paired fit/check coverage or connected-file leakage")
    scales = json.loads(SCALES.read_text())["scales"]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve().relative_to(Path.cwd().resolve()),
               Path("compression/multicore_magnitude_stream.py"), Path("compression/bounded_residual_stream.py"),
               Path("compression/predictive_coefficient_stream.py"), Path("compression/adaptive_coefficient_stream.py"),
               Path("compression/research_coefficient_stream.py"), Path("tools/run_multicore_magnitude.py")]
    hashes = {str(p): sha256(p) for p in sources}
    write_json(args.output / "protocol.json", {
        "kind": "multicore-conditional-magnitude-v1", "mode": args.mode, "seed": 106,
        "updates": 4096, "batch_cells": 4096, "alpha": .03, "sources": hashes,
        "scales_sha256": sha256(SCALES), "frozen_pairs_sha256": sha256(PRIOR / "FROZEN_FIT.json"),
        "fitting": fitting, "checks": checks, "reserved_waveform_access": False,
        "limits": "Nominal clock alignment; common retained rows only; consumed train development."})
    for path in sources:
        destination = args.output / "SOURCE_SNAPSHOT" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    data = []
    for row in fitting:
        a, b = load(row)
        zero = np.zeros_like(a)
        epsilon = .03 * scales[row["group_id"]]
        fa, fb = encode_bound(a, zero, epsilon), encode_bound(b, zero, epsilon)
        context = decode_bound(fa, zero)
        arrays, _, _ = unpack_predictive(fb)
        data.append((row, context, arrays[0]))
    statistics = []
    for group in groups:
        contexts = np.concatenate([context.ravel().astype(np.float64) for row, context, q in data if row["group_id"] == group])
        coefficients = np.concatenate([q.ravel().astype(np.float64) for row, context, q in data if row["group_id"] == group])
        statistics.append([float(contexts.mean()), max(float(contexts.std()), 1e-20),
                           max(float(np.sqrt(np.square(coefficients).mean())), 1.)])
    statistics = np.array(statistics, np.float64)
    training = defaultdict(lambda: defaultdict(list))
    for row, context, q in data:
        index = groups.index(row["group_id"])
        center, scale, target_scale = statistics[index]
        x = features(context, index, center, scale, args.mode)
        y = np.log2(1 + np.abs(q.astype(np.float64).ravel()) / target_scale).astype(np.float32)
        training[row["group_id"]][row["pair_index"]].append((x, y))
    del data, contexts, coefficients
    network = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)).cuda()
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    rng = np.random.default_rng(106)
    history = []
    start = time.monotonic()
    schedule = hashlib.sha256()
    for update in range(1, 4097):
        group = groups[int(rng.integers(2))]
        pairs = sorted(training[group])
        pair = pairs[int(rng.integers(len(pairs)))]
        windows = training[group][pair]
        window = int(rng.integers(len(windows)))
        x, y = windows[window]
        positions = rng.integers(len(y), size=4096)
        schedule.update(f"{update}:{group}:{pair}:{window}:".encode() + positions.astype("<i8").tobytes())
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = torch.from_numpy(x[positions]).cuda(), torch.from_numpy(y[positions]).cuda()
        loss = (network(inputs)[:, 0] - targets).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite conditional-magnitude loss")
        loss.backward()
        optimizer.step()
        if update == 1 or update % 512 == 0:
            record = {"update": update, "loss": float(loss.detach())}
            history.append(record)
            print(json.dumps({"mode": args.mode, **record}), flush=True)
    weights = {}
    for index, layer in enumerate((network[0], network[2], network[4])):
        weights[f"w{index}"] = layer.weight.detach().cpu().numpy().T.copy()
        weights[f"b{index}"] = layer.bias.detach().cpu().numpy().copy()
    model = MagnitudeModel(weights, args.mode, groups, statistics, np.zeros((2, 7), np.float32))
    thresholds = []
    for group in groups:
        scores = np.concatenate([model.score(x) for windows in training[group].values() for x, y in windows])
        thresholds.append(np.quantile(scores, np.arange(1, 8) / 8).astype(np.float32))
    model.thresholds = np.array(thresholds, np.float32)
    setup = model.setup_bytes()
    sender, receiver = MagnitudeModel.from_setup(setup), MagnitudeModel.from_setup(setup)
    (args.output / "model.npz").write_bytes(setup)
    write_json(args.output / "history.json", history)
    (args.output / "baseline").mkdir()
    (args.output / "candidate").mkdir()
    records = []
    joint_cells = 0
    for row in checks:
        a, b = load(row)
        zero = np.zeros_like(a)
        epsilon = .03 * scales[row["group_id"]]
        fa, fb = encode_bound(a, zero, epsilon), encode_bound(b, zero, epsilon)
        metadata = {key: row[key] for key in (
            "group_id", "left_id", "right_id", "clock_native", "left_block", "right_block",
            "start", "length", "channels", "left_unmatched_rows", "right_unmatched_rows")}
        metadata["shape"] = list(a.shape)
        baseline = pack_pair(fa, fb, metadata)
        frame = pack_pair(fa, fb, metadata, sender)
        left, right, paid_metadata, restored_b = unpack_pair(frame, receiver)
        qa, da, sa = unpack_predictive(fb)
        qb, db, sb = unpack_predictive(restored_b)
        if not np.array_equal(qa[0], qb[0]) or da != db or sa != sb or metadata != paid_metadata:
            raise AssertionError("Magnitude decoder changed coefficients or physical side")
        if left.tobytes() != decode_bound(fa, zero).tobytes() or right.tobytes() != decode_bound(fb, zero).tobytes():
            raise AssertionError("Joint magnitude physical bits changed")
        audit(a, left, epsilon)
        audit(b, right, epsilon)
        if len(frame) > len(baseline):
            raise AssertionError("Per-pair charged frame cap violated")
        name = Path(row["file"]).stem + ".frame"
        (args.output / "baseline" / name).write_bytes(baseline)
        (args.output / "candidate" / name).write_bytes(frame)
        records.append({"file": row["file"], "group_id": row["group_id"], "pair_index": row["pair_index"],
                        "baseline_bytes": len(baseline), "candidate_bytes": len(frame), "fallback": frame == baseline,
                        "joint_cells": a.size + b.size, "epsilon": epsilon})
        joint_cells += a.size + b.size
    baseline_bytes = sum(r["baseline_bytes"] for r in records)
    candidate_bytes = sum(r["candidate_bytes"] for r in records) + len(setup)
    if any(sha256(Path(p)) != digest for p, digest in hashes.items()) or sha256(SCALES) != json.loads(
            (args.output / "protocol.json").read_text())["scales_sha256"]:
        raise AssertionError("Magnitude sources changed")
    write_json(args.output / "results.json", {
        "mode": args.mode, "updates": 4096, "cell_presentations": 4096 * 4096,
        "schedule_sha256": schedule.hexdigest(), "records": records, "check_pairs": 8, "check_windows": 24,
        "joint_cells": joint_cells, "baseline_complete_bytes": baseline_bytes,
        "candidate_complete_bytes": candidate_bytes, "model_setup_bytes": len(setup),
        "baseline_bpc": 8 * baseline_bytes / joint_cells, "candidate_bpc": 8 * candidate_bytes / joint_cells,
        "relative_joint_byte_saving": 1 - candidate_bytes / baseline_bytes,
        "coefficient_and_physical_bits_equal": True, "bound_violations": 0,
        "elapsed_seconds": time.monotonic() - start})
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
