"""Paired fixed-budget causal/base-only neural lossless correction experiment."""
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

from compression.bounded_residual_stream import audit, decode as physical_decode, encode as physical_encode
from compression.full_tile_vq_stream import FullTileVQWire
from compression.neural_residual_stream import IntegerPredictor, decode, encode, features
from compression.predictive_coefficient_stream import pack_predictive, unpack_predictive
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.error_bounded_residual import PRIOR
from finetuning.fit_lifting_allocation import fitting_rows
from finetuning.model_factory import build_model
from finetuning.run_dense_product_study import CORPUS, SCALES, SOURCE, load
from finetuning.run_spatial_opportunity import select

FIRST = Path("results/experiments/error_bounded_residual_20261115")
UPDATES = 4096
BATCH = 4096


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("causal", "base_only"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path("vendor/smartgrids").resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(103)
    torch.cuda.manual_seed_all(103)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    first = json.loads((FIRST / "protocol.json").read_text())
    fitting = fitting_rows(select(CORPUS))
    checks = first["checks"]
    if (len(fitting) != 44 or any(row["split"] != "train" for row in fitting + checks)
            or {row["drive_id"] for row in fitting} & {row["drive_id"] for row in checks}
            or sha256(SOURCE) != first["source_sha256"]
            or sha256(SCALES) != first["scale_sha256"]
            or not (FIRST / "COMPLETE").exists()):
        raise ValueError("Train-only causal preflight failed")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve().relative_to(Path.cwd().resolve()), Path("compression/neural_residual_stream.py"),
               Path("compression/bounded_residual_stream.py"),
               Path("compression/predictive_coefficient_stream.py"),
               Path("compression/adaptive_coefficient_stream.py")]
    hashes = {str(p): sha256(p) for p in sources}
    write_json(args.output / "protocol.json", {
        "method": args.method, "updates": UPDATES, "batch_cells": BATCH, "seed": 103,
        "alpha": .03, "sources": hashes, "fitting": fitting, "checks": checks,
        "source_sha256": sha256(SOURCE), "scale_sha256": sha256(SCALES),
        "input_sha256": {row["sample_path"]: sha256(CORPUS / "train" / Path(row["sample_path"]).name)
                         for row in fitting + checks},
        "reserved_waveform_access": False, "selection": "fixed endpoint; no checks during training"})
    for p in sources:
        destination = args.output / "SOURCE_SNAPSHOT" / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    scales = json.loads(SCALES.read_text())["scales"]
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    config = {**saved["config"], "exact_eval_quantization": True}
    base_model = build_model(config, saved, "cuda").eval().requires_grad_(False)
    wire = FullTileVQWire.from_setup(base_model, (PRIOR / "registry.zlib").read_bytes())
    fit_data = []
    sums = defaultdict(lambda: [0., 0])
    with torch.no_grad():
        for row in fitting:
            raw = load(row)
            base = wire.decode(wire.encode(raw))
            correction = physical_encode(raw, base, .03 * scales[row["group_id"]])
            arrays, _, _ = unpack_predictive(correction)
            q, = arrays
            fit_data.append((row["group_id"], base, q))
            sums[row["group_id"]][0] += float(np.square(q.astype(np.float64)).sum())
            sums[row["group_id"]][1] += q.size
    coefficient_scales = {group: max(1., float(np.sqrt(total / count))) for group, (total, count) in sums.items()}
    write_json(args.output / "coefficient_scales.json", coefficient_scales)
    contextual = args.method == "causal"
    training = defaultdict(list)
    for group, base, q in fit_data:
        scale = coefficient_scales[group]
        training[group].append((features(base, q, scale, contextual).reshape(-1, 8),
                                (q.astype(np.float64).ravel() / scale).astype(np.float32)))
    del fit_data, base_model, wire, saved
    torch.cuda.empty_cache()
    network = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)).cuda()
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    rng = np.random.default_rng(103)
    groups = sorted(training)
    history = []
    start = time.monotonic()
    for update in range(1, UPDATES + 1):
        group = groups[int(rng.integers(len(groups)))]
        rows = training[group]
        x, y = rows[int(rng.integers(len(rows)))]
        indices = rng.integers(len(y), size=BATCH)
        inputs = torch.from_numpy(x[indices]).cuda()
        target = torch.from_numpy(y[indices]).cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = (network(inputs)[:, 0] - target).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite causal training loss")
        loss.backward()
        optimizer.step()
        if update == 1 or update % 512 == 0:
            record = {"update": update, "loss": float(loss.detach()), "group": group}
            history.append(record)
            print(json.dumps({"method": args.method, **record}), flush=True)
    weights = {}
    for index, layer in enumerate((network[0], network[2], network[4])):
        weights[f"w{index}"] = layer.weight.detach().cpu().numpy().T.copy()
        weights[f"b{index}"] = layer.bias.detach().cpu().numpy().copy()
    predictor = IntegerPredictor(weights, contextual)
    setup = predictor.setup_bytes()
    identifier = hashlib.sha256(setup).hexdigest()
    receiver = IntegerPredictor.from_setup(setup)
    (args.output / "predictor.npz").write_bytes(setup)
    write_json(args.output / "history.json", history)
    # Reload the frozen VQ receiver for check-only decoding, never updating it.
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    base_model = build_model(config, saved, "cuda").eval().requires_grad_(False)
    wire = FullTileVQWire.from_setup(base_model, (PRIOR / "registry.zlib").read_bytes())
    (args.output / "frames").mkdir()
    baseline_summary = json.loads((FIRST / "alpha_0.03/neural/results.json").read_text())
    records = []
    family_bytes = defaultdict(lambda: [0, 0])
    replay_start = time.monotonic()
    for index, row in enumerate(checks):
        name = Path(row["sample_path"]).stem
        base_frame = (PRIOR / "baseline" / f"{name}.frame").read_bytes()
        base = wire.decode(base_frame)
        original = (FIRST / "alpha_0.03/neural" / f"{name}.frame").read_bytes()
        frame = encode(original, base, predictor, identifier, coefficient_scales[row["group_id"]])
        arrays, descriptor, side = decode(frame, base.copy(), receiver, identifier)
        original_arrays, original_descriptor, original_side = unpack_predictive(original)
        if (not np.array_equal(arrays[0], original_arrays[0]) or descriptor != original_descriptor
                or side != original_side):
            raise AssertionError("Causal coefficient replay differs from original correction")
        restored = physical_decode(pack_predictive(arrays, descriptor, side), base.copy())
        original_restored = physical_decode(original, base)
        if restored.tobytes() != original_restored.tobytes():
            raise AssertionError("Causal physical reconstruction bits changed")
        audit(load(row), restored, .03 * scales[row["group_id"]])
        (args.output / "frames" / f"{name}.frame").write_bytes(frame)
        family = row["group_id"].split("/")[0]
        family_bytes[family][0] += len(base_frame) + len(original)
        family_bytes[family][1] += len(base_frame) + len(frame)
        records.append({"sample_path": row["sample_path"], "base_bytes": len(base_frame),
                        "original_correction_bytes": len(original), "correction_bytes": len(frame),
                        "fallback": frame == original})
        if (index + 1) % 16 == 0:
            print(json.dumps({"method": args.method, "replayed": index + 1}), flush=True)
    original_bytes = baseline_summary["stream_bytes"]
    predicted_frames = sum(r["base_bytes"] + r["correction_bytes"] for r in records)
    scale_setup = (args.output / "coefficient_scales.json").stat().st_size
    stream_bytes = predicted_frames + baseline_summary["setup_bytes"] + len(setup) + scale_setup
    saving = 1 - stream_bytes / original_bytes
    no_family_increase = all(new <= old for old, new in family_bytes.values())
    if any(sha256(Path(p)) != expected for p, expected in hashes.items()):
        raise AssertionError("Causal sources changed during execution")
    write_json(args.output / "results.json", {
        "method": args.method, "updates": UPDATES, "training_cell_presentations": UPDATES * BATCH,
        "cells": baseline_summary["cells"], "windows": len(checks), "records": records,
        "coefficient_and_physical_replay_exact": True, "guarantee_violations": 0,
        "original_stream_bytes": original_bytes, "stream_bytes": stream_bytes,
        "stream_bpc": 8 * stream_bytes / baseline_summary["cells"],
        "relative_complete_byte_saving": saving, "family_frame_bytes_original_new": dict(family_bytes),
        "predictor_bytes": len(setup), "coefficient_scale_setup_bytes": scale_setup,
        "full_shared_vq_model_bytes": SOURCE.stat().st_size,
        "opportunity_gate_without_control": saving >= .05 and no_family_increase,
        "elapsed_seconds": time.monotonic() - start,
        "check_codec_seconds": time.monotonic() - replay_start,
        "limits": "Same correction q and physical bits; consumed train checks, not independent confirmation."})
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
