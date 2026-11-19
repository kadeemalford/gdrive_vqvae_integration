"""Controlled FiLM metadata study with combination-excluded training and paid context."""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from compression.conditioned_vq_stream import ConditionedVQWire
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.dense_product_vq import export
from finetuning.fit_lifting_allocation import transport
from finetuning.full_tile_bottleneck import select_probes
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.model_factory import new_model
from finetuning.physical_conditioning import (
    METADATA, RECORDS, ConditionalGroupNorm, build, condition, initialize, load_plan, set_condition,
)
from finetuning.regime_diversity import crop
from finetuning.reversible_normalization import decode as physical_decode
from finetuning.robust_escape import encode_robust
from finetuning.run_dense_product_study import CORPUS, SCALES, load
from finetuning.run_joint_anchor_study import atomic_save
from finetuning.run_regime_diversity import EXPECTED, ORIGINAL
from finetuning.streaming_physical_summary import StreamingPhysicalSummary


@torch.no_grad()
def evaluate(model, config, fitting, rows, scales, maps, output, update):
    model.eval()
    folder = output / f"check_{update:06d}"
    folder.mkdir()
    counts = np.zeros(2048, np.int64)
    for row in fitting:
        raw = load(row)
        if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
            continue
        set_condition(model, condition(config["plan"], config["method"], row["group_id"]))
        packet = encode_robust(raw)
        x = torch.from_numpy(packet.values)[None, None].cuda()
        _, _, ids, _, _ = exact_branch_forward(model, x)
        counts += np.bincount(ids.cpu().numpy(), minlength=2048)
    data = export(model, config)
    identifier = hashlib.sha256(data).hexdigest()
    sender = ConditionedVQWire.create(model, counts, identifier, config)
    registry = sender.setup_bytes()
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    receiver_model = build(saved["config"], saved["model_state_dict"], "cuda")
    receiver = ConditionedVQWire.from_setup(receiver_model, registry, config)
    geometry = transport(rows, maps)
    (folder / "registry.zlib").write_bytes(registry)
    (folder / "transport.zlib").write_bytes(geometry)
    results = {}
    for missing in (False, True):
        label = "missing" if missing else "normal"
        target = folder / label
        target.mkdir()
        summary = StreamingPhysicalSummary(scales, maps)
        for row in rows:
            raw = load(row)
            frame = sender.encode(raw, row["group_id"], missing)
            restored = receiver.decode(frame)
            if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
                expected = raw
            else:
                packet = encode_robust(raw)
                x = torch.from_numpy(packet.values)[None, None].cuda()
                direct, _, _, _, _ = exact_branch_forward(model, x)
                expected = physical_decode(direct[0, 0].cpu().numpy(), packet.side_stream)
            if expected.tobytes() != restored.tobytes():
                raise AssertionError("Paid conditioned replay differs from direct physical path")
            summary.update(row, raw, restored, len(frame))
            (target / (Path(row["sample_path"]).stem + ".frame")).write_bytes(frame)
        result = summary.result(len(registry) + len(geometry))
        result.update(update=update, condition_missing=missing, model_sha256=identifier,
                      full_shared_model_bytes=len(data), independent_receiver_bits_equal=True)
        write_json(target / "results.json", result)
        results[label] = result
    return results, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("constant", "hardware", "category"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(Path("vendor/smartgrids").resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(2117)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rng, crop_rng = random.Random(2117), np.random.default_rng(2117)
    if sha256(ORIGINAL) != EXPECTED:
        raise ValueError("Conditioning requires original pre-FEBUS OOI source")
    plan = load_plan()
    fitting, probes = select_probes(CORPUS)
    fitting = [r for r in fitting if r["group_id"] in plan["seen"]]
    checks = [r for r in probes if r["role"] == "check"]
    excluded = {r["drive_id"] for r in checks}
    training = [r for r in csv.DictReader((CORPUS / "samples_manifest.csv").open())
                if r["split"] == "train" and r["drive_id"] not in excluded
                and r["group_id"] in plan["seen"] and int(r["channels"]) >= 8 and int(r["time_samples"]) >= 1000]
    if (len(checks) != 136 or len(excluded) != 43 or not fitting or not training
            or any(r["split"] != "train" for r in fitting + checks + training)
            or any(r["group_id"] in plan["heldout"] for r in fitting + training)
            or {r["drive_id"] for r in fitting + training} & excluded):
        raise ValueError("Conditioning cohort leakage or coverage mismatch")
    pool = defaultdict(lambda: defaultdict(list))
    for row in sorted(training, key=lambda r: r["sample_path"]):
        pool[row["group_id"]][row["drive_id"]].append(row)
    if set(pool) != set(plan["seen"]):
        raise ValueError("Missing training configurations")
    families = sorted({g.split("/")[0] for g in plan["seen"]})
    if len(families) != 4:
        raise ValueError("Heldout selection removed a training family")
    scales = json.loads(SCALES.read_text())["scales"]
    maps = {r["channel_map_id"]: r for r in map(json.loads, (CORPUS / "channel_maps.jsonl").open())}
    sources = [Path(__file__).resolve().relative_to(Path.cwd().resolve()),
               Path("finetuning/physical_conditioning.py"), Path("compression/conditioned_vq_stream.py"),
               Path("compression/full_tile_vq_stream.py"), Path("compression/alphabet_stream.py"),
               Path("finetuning/model_factory.py"), Path("finetuning/latent_geometry.py"),
               Path("finetuning/joint_anchor_vq.py"), Path("finetuning/dense_product_vq.py"),
               Path("finetuning/regime_diversity.py"), Path("finetuning/streaming_physical_summary.py"),
               Path("finetuning/prospective_metrics.py"), Path("finetuning/prospective_spectral.py"),
               Path("finetuning/robust_escape.py"), Path("finetuning/reversible_normalization.py")]
    hashes = {str(p): sha256(p) for p in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    endpoint = 2 if args.smoke else 16384
    write_json(args.output / "protocol.json", {
        "kind": "hardware-versus-category-conditioning-v1", "method": args.method,
        "smoke": args.smoke, "updates": endpoint, "source_sha256": EXPECTED,
        "metadata_sha256": sha256(METADATA), "records_sha256": sha256(RECORDS),
        "scales_sha256": sha256(SCALES), "sources": hashes, "plan": plan,
        "training_files": sorted({r["drive_id"] for r in training}),
        "fitting": fitting, "checks": checks,
        "input_sha256": {r["sample_path"]: sha256(CORPUS / "train" / Path(r["sample_path"]).name)
                         for r in {r["sample_path"]: r for r in fitting + checks + training}.values()},
        "missing_rule": "every tenth update of each family; constant ignores flag",
        "reserved_waveform_access": False, "selection": "fixed endpoint only"})
    for p in sources:
        target = args.output / "SOURCE_SNAPSHOT" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    source = torch.load(ORIGINAL, map_location="cpu", weights_only=False)
    model, config = initialize(source, args.method, plan, "cuda")
    if args.smoke:
        original = new_model(config["base_config"], "cuda").eval()
        original.load_state_dict(source["model_state_dict"], strict=True)
        row = max(fitting, key=lambda r: int(r["channels"]))
        raw, _, _ = crop(load(row), np.random.default_rng(105))
        packet = encode_robust(raw)
        x = torch.from_numpy(packet.values)[None, None].cuda()
        with torch.no_grad():
            model.eval()
            set_condition(model, condition(plan, args.method, row["group_id"]))
            a, _, ia, _, _ = exact_branch_forward(original, x)
            b, _, ib, _, _ = exact_branch_forward(model, x)
            if not torch.equal(a.view(torch.int32), b.view(torch.int32)) or not torch.equal(ia, ib):
                raise AssertionError("Zero FiLM changed initial source function or indices")
        write_json(args.output / "INITIAL_PARITY.json", {"normalized_bits_equal": True, "indices_equal": True})
        del original, a, b, ia, ib, x
    del source
    eval_rows = checks if not args.smoke else [checks[0], checks[-1]]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        initial, _ = evaluate(model, config, fitting, eval_rows, scales, maps, args.output, 0)
    del _
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    family_updates, group_updates = defaultdict(int), defaultdict(int)
    schedule = hashlib.sha256()
    missing_updates = 0
    with (args.output / "history.jsonl").open("w") as history:
        for update in range(1, endpoint + 1):
            family = families[(update - 1) % len(families)]
            group = rng.choice([g for g in plan["seen"] if g.split("/")[0] == family])
            row = rng.choice(pool[group][rng.choice(sorted(pool[group]))])
            raw, offset, start = crop(load(row), crop_rng)
            family_updates[family] += 1
            group_updates[group] += 1
            missing = family_updates[family] % 10 == 0
            missing_updates += int(missing)
            schedule.update(f"{update}:{row['sample_path']}:{offset}:{start}:{int(missing)}\n".encode())
            set_condition(model, condition(plan, args.method, group, missing))
            packet = encode_robust(raw)
            x = torch.from_numpy(packet.values)[None, None].cuda()
            mask = torch.from_numpy(packet.valid)[None, None].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted, qloss, _, _, perplexity = exact_branch_forward(model, x)
            error = (predicted - x)[mask]
            distortion = .5 * error.square().mean() + .5 * error.abs().mean()
            loss = distortion + qloss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite physical conditioning loss")
            loss.backward()
            if args.smoke and update == 1:
                gradient = sum(float(m.modulator[-1].weight.grad.abs().sum())
                               for m in model.modules() if isinstance(m, ConditionalGroupNorm)
                               and m.modulator[-1].weight.grad is not None)
                if gradient <= 0 or not np.isfinite(gradient):
                    raise AssertionError("Inactive or nonfinite FiLM gradients")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if update % 32 == 0 or update == endpoint:
                history.write(json.dumps({"update": update, "cells": update * 8000,
                    "distortion": float(distortion.detach()), "perplexity": float(perplexity),
                    "gradient_norm": float(norm), "missing_updates": missing_updates}) + "\n")
                history.flush()
            if update % 2048 == 0 or update == endpoint:
                atomic_save({"config": config, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "update": update},
                    args.output / "last.pt")
                print(json.dumps({"method": args.method, "update": update, "cells": update * 8000}), flush=True)
    final, data = evaluate(model, config, fitting, eval_rows, scales, maps, args.output, endpoint)
    (args.output / "receiver.pt").write_bytes(data)
    if sha256(ORIGINAL) != EXPECTED or any(sha256(Path(p)) != digest for p, digest in hashes.items()):
        raise AssertionError("Conditioning sources changed")
    write_json(args.output / "results.json", {
        "method": args.method, "updates": endpoint, "cells": endpoint * 8000,
        "family_updates": dict(family_updates), "group_updates": dict(group_updates),
        "missing_updates": missing_updates, "schedule_sha256": schedule.hexdigest(),
        "initial_nrmse_star": initial["normal"]["macro_families"],
        "endpoint_nrmse_star": final["normal"]["macro_families"],
        "endpoint_bpc": final["normal"]["stream_bpc"], "receiver_sha256": sha256(args.output / "receiver.pt"),
        "seen": plan["seen"], "heldout": plan["heldout"], "check_windows": len(eval_rows),
        "norm_layers": config["norm_layers"], "condition_receiver_replay_exact": True})
    (args.output / ("SMOKE_COMPLETE" if args.smoke else "COMPLETE")).touch()


if __name__ == "__main__":
    main()
