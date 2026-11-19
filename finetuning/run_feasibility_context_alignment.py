#!/usr/bin/env python3
"""F4 paired context alignment; fixed endpoints, never checkpoint selection.

Protocol JSON (frozen by the caller) must contain ``inputs`` mapping metadata,
manifest, checks, clock_exclusions, reference_scales, channel_maps and
mature_registry to {path, sha256}. Optional ``source`` and ``mature`` entries use
the same schema. The checks input is the historical constant-FiLM protocol,
not a newly selected panel. All paths are checkout-relative or absolute.
Smoke requires only a protocol with ``smoke_config`` (or the full-size defaults),
and never opens FEBUS inputs or pretrained weights.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/smartgrids"))

from finetuning.run_feasibility_context import (
    MATURE_SHA, canonical, digest, model_state_hash, resolve, write_json,
)

SEED = 2117
UPDATES = 8192
SOURCE = "models/febus_finetune_base/best.pt"
SOURCE_SHA = "7d2da0d1cbd1d3dae579d7fd900850afc9ef15f645ea844168c0f78f18b0a27c"
CHECKS = "results/experiments/physical_conditioning_20261115/scientific/constant/protocol.json"
CLOCK_EXCLUSIONS = "results/experiments/feasibility_clock_envelopes_20261118/results.json"
CHECKS_SHA = "b9788c790855335e36ab66f46ecc4be65b42a07572a94f7c5a0e12639bcd0766"
CLOCK_EXCLUSIONS_SHA = "1fe50b17fe13acc92f903cd04fbab8d90a014eb03f0906c19ba79a7605cf33f0"


def safe_train_row(row):
    path = Path(row["sample_path"])
    if (row.get("split") != "train" or path.suffix != ".npz"
            or ".." in path.parts or path.is_absolute()
            or not path.parts or path.parts[0] != "data"
            or int(row["channels"]) < 1):
        raise ValueError("Only prepared train NPZ rows are authorized")


def select_training_rows(rows, freeze, exclusions):
    """Conservative pool32 minus discovery110's two IDs; no D2c roles reused."""
    pool = freeze["f2_candidate_pool"]
    files = {r["drive_id"]: r["group_id"] for r in pool["files"]}
    removed = set(exclusions["removed_file_ids"])
    historical = set(pool["historical_check_ids"])
    if (len(files) != 32 or len(removed) != 2 or not removed <= files.keys()
            or exclusions.get("kind") != "feasibility-clock-envelope-sensitivity-v1"
            or exclusions.get("remaining_file_count") != 30 or len(historical) != 43):
        raise ValueError("Expected frozen pool32 minus two conservative exclusions")
    allowed = files.keys() - removed
    if allowed & historical:
        raise ValueError("Historical checks cannot enter F4 fit")
    selected = []
    for row in rows:
        if row["drive_id"] not in allowed:
            continue
        safe_train_row(row)
        if row["group_id"] != files[row["drive_id"]]:
            raise ValueError("Frozen file/group identity differs")
        selected.append(dict(row))
    if {r["drive_id"] for r in selected} != allowed:
        raise ValueError("Missing prepared rows for the 30 authorized files")
    if len({r["sample_path"] for r in selected}) != len(selected):
        raise ValueError("Duplicate prepared row")
    return sorted(selected, key=lambda r: r["sample_path"])


def make_schedule(rows, updates=UPDATES, seed=SEED):
    """Sorted family round robin, uniform group/file/row, one immutable full tile."""
    tree = defaultdict(lambda: defaultdict(list))
    for row in rows:
        safe_train_row(row)
        tree[row["group_id"]][row["drive_id"]].append(row)
    families = sorted({g.split("/")[0] for g in tree})
    if not families or updates < 1:
        raise ValueError("Empty schedule")
    rng = random.Random(seed)
    result = []
    for index in range(updates):
        family = families[index % len(families)]
        group = rng.choice(sorted(g for g in tree if g.split("/")[0] == family))
        file_id = rng.choice(sorted(tree[group]))
        row = rng.choice(sorted(tree[group][file_id], key=lambda r: r["sample_path"]))
        result.append({"update": index + 1, **row})
    return result


def parts(channels, arm):
    if type(channels) is not int or channels < 1 or arm not in ("A", "B"):
        raise ValueError("Invalid arm/geometry")
    width = 8 if arm == "A" else channels
    return [(a, min(a + width, channels)) for a in range(0, channels, width)]


def microbatches(channels, arm):
    fragments = parts(channels, arm)
    if arm == "B":
        return [fragments]
    full = [p for p in fragments if p[1] - p[0] == 8]
    tail = [p for p in fragments if p[1] - p[0] != 8]
    return [full[i:i + 4] for i in range(0, len(full), 4)] + [[p] for p in tail]


def normalized_loss(prediction, target, mask, qloss, cells, total_cells):
    """Native normalized distortion, weighted by acquired (not valid) cells.

    qloss is the native mean over padded latent vectors, including its native
    commitment term. Equal-size microbatch fragments make that mean compatible
    with cell weighting; padded tokens are logged, never treated as source cells.
    """
    import torch
    terms = []
    for index, count in enumerate(cells):
        error = (prediction[index] - target[index])[mask[index]]
        distortion = (.5 * error.square().mean() + .5 * error.abs().mean()
                      if error.numel() else prediction[index].sum() * 0)
        terms.append(distortion * (count / total_cells))
    return torch.stack(terms).sum() + qloss * (sum(cells) / total_cells)


def checked_input(entry):
    path = resolve(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Frozen input hash differs: {path}")
    return path


def preflight(protocol):
    """Metadata/hash validation only; all waveform reads occur later in the CLI."""
    entries = protocol["inputs"]
    required = ("metadata", "manifest", "checks", "clock_exclusions",
                "reference_scales", "channel_maps", "mature_registry")
    paths = {key: checked_input(entries[key]) for key in required}
    if (paths["checks"].resolve() != resolve(CHECKS).resolve()
            or paths["clock_exclusions"].resolve() != resolve(CLOCK_EXCLUSIONS).resolve()
            or entries["checks"]["sha256"] != CHECKS_SHA
            or entries["clock_exclusions"]["sha256"] != CLOCK_EXCLUSIONS_SHA):
        raise ValueError("Use the historical constant protocol and discovery110 evidence")
    freeze = json.loads(paths["metadata"].read_text())
    if freeze.get("kind") != "feasibility-metadata-freeze-v1":
        raise ValueError("Wrong metadata freeze")
    for key, dependency in freeze["input_sha256"].items():
        checked_input(dependency)
        if key in entries and dependency["sha256"] != entries[key]["sha256"]:
            raise ValueError("Metadata dependency identity differs")
    exclusions = json.loads(paths["clock_exclusions"].read_text())
    lineage = exclusions["input_and_source_sha256"]
    if not any(resolve(p).resolve() == paths["metadata"].resolve()
               and sha == entries["metadata"]["sha256"] for p, sha in lineage.items()):
        raise ValueError("Clock evidence is not bound to this metadata freeze")
    with paths["manifest"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = select_training_rows(rows, freeze, exclusions)
    history = json.loads(paths["checks"].read_text())
    checks = history["checks"]
    if len(checks) != 136 or len({r["drive_id"] for r in checks}) != 43:
        raise ValueError("Expected all 136 historical checks / 43 files")
    by_path = {r["sample_path"]: r for r in rows}
    checks = [{**by_path[r["sample_path"]], **r} for r in checks]
    historical_ids = set(freeze["f2_candidate_pool"]["historical_check_ids"])
    if {r["drive_id"] for r in checks} != historical_ids:
        raise ValueError("Historical check identities differ")
    for row in checks:
        safe_train_row(row)
    source = protocol.get("source", {"path": SOURCE, "sha256": SOURCE_SHA})
    if source["sha256"] != SOURCE_SHA or resolve(source["path"]).resolve() != resolve(SOURCE).resolve():
        raise ValueError("Wrong prescribed offshore source")
    paths["source"] = checked_input(source)
    # Per-path hashes, never a global 'verified' label for only a subset.
    inputs, verification = {}, {}
    for row in train + checks:
        name = row["sample_path"]
        sha = digest(resolve(name))
        known = history.get("input_sha256", {}).get(name)
        if known is not None and known != sha:
            raise ValueError(f"Historical NPZ hash differs: {name}")
        if row in checks and known is None:
            raise ValueError("Historical check requires its historical NPZ hash")
        inputs[name] = sha
        verification[name] = "matched" if known else "not_in_historical_protocol"
    maps = {r["channel_map_id"]: r for r in
            map(json.loads, paths["channel_maps"].read_text().splitlines())}
    scale_data = json.loads(paths["reference_scales"].read_text())
    scales = scale_data["scales"]
    reference = freeze["input_sha256"]["manifest_reference"]
    ref_path = checked_input(reference)
    if (scale_data["protocol_sha256"] != reference["sha256"]
            or json.loads(ref_path.read_text())["channel_maps_sha256"]
            != entries["channel_maps"]["sha256"]):
        raise ValueError("Scale/channel map lineage differs")
    for row in train + checks:
        mapping = maps[row["channel_map_id"]]
        if (mapping["drive_id"] != row["drive_id"]
                or len(mapping["source_channel_indices"]) != int(row["channels"])
                or not np.isfinite(scales[row["group_id"]]) or scales[row["group_id"]] <= 0):
            raise ValueError("Invalid geometry or physical reference scale")
    registry = paths["mature_registry"].read_bytes()
    metadata, counts = zlib.decompress(registry).split(b"\n", 1)
    metadata = json.loads(metadata)
    if (metadata != {"kind": "full-tile-vq-fak1-v1", "identifier": MATURE_SHA, "K": 2048}
            or len(counts) != 2048 * 8):
        raise ValueError("Original mature FAK1 registry required")
    counts = np.frombuffer(counts, "<i8").copy()
    if np.any(counts < 0) or np.any(counts > 2**40):
        raise ValueError("Invalid original mature counts")
    return paths, train, checks, maps, scales, inputs, verification, counts


def load_raw(row, inputs):
    from finetuning.run_feasibility_full_coverage import validate_raw
    path = resolve(row["sample_path"])
    if digest(path) != inputs[row["sample_path"]]:
        raise ValueError("Prepared NPZ changed after freeze")
    with np.load(path, allow_pickle=False) as data:
        raw = data["data"].copy()
    validate_raw(raw, int(row["channels"]))
    return raw


def train_update(model, optimizer, raw, arm, device, guard=lambda: None):
    import torch
    from finetuning.joint_anchor_vq import exact_branch_forward
    from finetuning.robust_escape import encode_robust
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tokens, batches, loss_value = 0, 0, 0.
    audits = []
    for batch in microbatches(len(raw), arm):
        guard()
        packets = [encode_robust(np.ascontiguousarray(raw[a:b])) for a, b in batch]
        target = torch.from_numpy(np.stack([p.values for p in packets]))[:, None].to(device)
        mask = torch.from_numpy(np.stack([p.valid for p in packets]))[:, None].to(device)
        # Native affine scale is stored in the reversible side stream.
        from finetuning.reversible_normalization import parse_side_stream
        scales = [parse_side_stream(p.side_stream)[2] for p in packets]
        prediction, qloss, ids, latent, _ = exact_branch_forward(model, target)
        cells = [(b - a) * raw.shape[1] for a, b in batch]
        loss = normalized_loss(prediction, target, mask, qloss, cells, raw.size)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite F4 training loss")
        loss.backward()
        loss_value += float(loss.detach())
        tokens += ids.numel()
        batches += 1
        audits.append({"parts": batch, "cells": sum(cells), "latent": list(latent),
                       "tokens": ids.numel(), "qloss": float(qloss.detach()),
                       "valid_cells": int(mask.sum()), "affine_scales": scales,
                       "codebook_used_ids": int(torch.unique(ids).numel()),
                       "index_sha256": hashlib.sha256(
                           ids.detach().cpu().numpy().astype("<u4").tobytes()).hexdigest(),
                       "padding_bottom_per_fragment": [(-(b-a)) % 16 for a, b in batch],
                       "padding_right": (-raw.shape[1]) % 16,
                       "weight": sum(cells) / raw.size})
    guard()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {"loss": loss_value, "gradient_norm": float(norm), "cells": raw.size,
            "tokens": tokens, "microbatches": batches, "fragments": audits}


def evaluate(model, config, checks, maps, scales, inputs, counts, output, code_bytes, guard):
    import torch
    from compression.full_tile_vq_stream import FullTileVQWire
    from finetuning.fit_lifting_allocation import transport
    from finetuning.streaming_physical_summary import StreamingPhysicalSummary
    from finetuning.feasibility_metrics import target_supported_spectra
    from finetuning.run_feasibility_full_coverage import (
        coverage_guards, decode_coverage, encode_coverage, rate_ledger,
    )
    output.mkdir()
    model.eval().requires_grad_(False)
    before = model_state_hash(model)
    torch.save({"config": config, "model_state_dict": {
        k: v.detach().cpu() for k, v in model.state_dict().items()}}, output / "receiver.pt")
    wire = FullTileVQWire(model, counts, digest(output / "receiver.pt"))
    registry = wire.setup_bytes()
    (output / "registry.zlib").write_bytes(registry)
    geometry = transport(checks, maps)
    (output / "transport.zlib").write_bytes(geometry)
    contract = canonical({"wire": "FPB1/FZC1/FAK1", "model_sha256": wire.identifier,
                          "registry_sha256": hashlib.sha256(registry).hexdigest(),
                          "sources": "../code_snapshot", "full_tile_only": True,
                          "platform": "Python/NumPy/Torch/OS shared"})
    (output / "decoder_contract.json").write_bytes(contract)
    summary, tiles = StreamingPhysicalSummary(scales, maps), []
    with torch.no_grad():
        for index, row in enumerate(checks):
            guard()
            raw = load_raw(row, inputs)
            bundle = encode_coverage(raw, wire, "full")
            path = output / f"{index:03d}.fpb"
            path.write_bytes(bundle)
            restored = decode_coverage(bundle, wire)
            repeated = decode_coverage(path.read_bytes(), wire)
            if restored.tobytes() != repeated.tobytes():
                raise AssertionError("Independent decode-only replay differs")
            summary.update(row, raw, restored, len(bundle))
            error = restored.astype(np.float64) - raw.astype(np.float64)
            tiles.append({"sample_path": row["sample_path"], "group_id": row["group_id"],
                          "sse": float(np.sum(error**2)), "bundle_bytes": len(bundle),
                          "sha256": digest(path), "decode_replay_bit_exact": True,
                          "spectral_statistics": target_supported_spectra(
                              raw, restored, float(row["dt_s"]), scales[row["group_id"]])})
    result = summary.result(len(geometry))
    result["coverage_guards"] = coverage_guards(result, tiles)
    result["tiles"] = tiles
    result["rate"] = rate_ledger(result["frame_bytes"], len(geometry), len(registry),
                                  (output / "receiver.pt").stat().st_size, code_bytes,
                                  len(contract), result["cells"])
    result.update(parameter_updates_during_checks=0, heldout_claim=False, model_promoted=False)
    if model_state_hash(model) != before:
        raise AssertionError("Endpoint evaluation changed model state")
    write_json(output / "results.json", result)
    return result


def gpu_identity(device):
    import torch
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (device != "cuda:0" or not visible or len(visible.split(",")) != 1
            or visible.strip() in ("", "-1") or torch.cuda.device_count() != 1):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU; use --device cuda:0")
    props = torch.cuda.get_device_properties(0)
    return {"logical_device": device, "CUDA_VISIBLE_DEVICES": visible,
            "name": props.name, "total_memory": props.total_memory,
            "uuid": str(getattr(props, "uuid", "unavailable"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "both", "mature"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=86400.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    if args.output.exists():
        raise ValueError("New output required; no resume or overwrite")
    if args.smoke and args.arm == "mature":
        parser.error("Mature evaluation requires its real historical export")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("selection", "fixed_endpoint") != "fixed_endpoint":
        raise ValueError("Protocol cannot request checkpoint selection")
    if not args.smoke:
        if protocol.get("kind") != "feasibility-context-alignment-definition-v1":
            raise ValueError("Expected the fixed report-111 protocol")
        checked_input(protocol["scientific_report"])
    frozen = None if args.smoke else preflight(protocol)
    import torch
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    gpu = gpu_identity(args.device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    sources = sorted(set(ROOT.glob("finetuning/*.py")) | set(ROOT.glob("compression/*.py"))
                     | set((ROOT / "vendor").rglob("*.py"))
                     | {ROOT / "tests/test_feasibility_context_alignment.py"})
    import vqvae
    vendor = Path(next(iter(vqvae.__path__))).resolve()
    sources = sorted(set(sources) | set(vendor.rglob("*.py")))
    if not (vendor / "train_vqvae.py").is_file():
        raise RuntimeError("Native vendor sources required")
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("vendor/vqvae") / path.relative_to(vendor)
        destination = args.output / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(path)] = digest(path)
    code_bytes = sum(p.stat().st_size for p in (args.output / "code_snapshot").rglob("*") if p.is_file())
    if args.smoke:
        rng = np.random.default_rng(SEED)
        synthetic = [rng.normal(size=(n, 1000)).astype(np.float32) for n in (512, 2)]
        schedule = [{"update": i + 1, "sample_path": f"synthetic:{len(raw)}x1000",
                     "channels": len(raw)} for i, raw in enumerate(synthetic)]
        config = protocol.get("smoke_config", {
            "base_channels": 64, "embedding_dim": 128, "num_embeddings": 2048,
            "downsample_levels": 4, "commitment_cost": .25,
            "latent_geometry": "spatial4-temporal4"})
        inputs = {r["sample_path"]: hashlib.sha256(raw.tobytes()).hexdigest()
                  for r, raw in zip(schedule, synthetic)}
        verification = {}
        tracked = {}
    else:
        paths, train, checks, maps, scales, inputs, verification, counts = frozen
        schedule = make_schedule(train)
        tracked = {str(p): digest(p) for p in paths.values()}
        tracked[str(checked_input(protocol["scientific_report"]))] = protocol["scientific_report"]["sha256"]
        metadata = json.loads(paths["metadata"].read_text())
        tracked.update({str(resolve(entry["path"])): entry["sha256"]
                        for entry in metadata["input_sha256"].values()})
        if args.arm == "mature":
            entry = protocol["mature"]
            if entry["sha256"] != MATURE_SHA:
                raise ValueError("Wrong mature receiver")
            tracked[str(checked_input(entry))] = MATURE_SHA
        tracked.update({str(resolve(p)): sha for p, sha in inputs.items()})
        source = torch.load(paths["source"], map_location="cpu", weights_only=False)
        # The offshore source predates the rate-allocation geometry metadata.
        # Geometry is an explicit F4 intervention, never inferred from shapes.
        config = {**source["config"], "latent_geometry": "spatial4-temporal4"}
    if (config.get("latent_geometry") != "spatial4-temporal4"
            or config.get("num_embeddings") != 2048 or config.get("embedding_dim") != 128):
        raise ValueError("Fixed flat spatial4-temporal4 K2048x128 geometry required")
    config = {**config, "exact_eval_quantization": True, "activation_checkpointing": True}
    schedule_bytes = b"".join(canonical(row) + b"\n" for row in schedule)
    (args.output / "schedule.jsonl").write_bytes(schedule_bytes)
    shutil.copy2(args.protocol, args.output / "scientific_protocol.json")
    manifest = {"kind": "feasibility-context-alignment-v1", "seed": SEED,
                "updates_per_training_arm": len(schedule), "smoke": args.smoke,
                "scientific_protocol_sha256": digest(args.protocol), "gpu": gpu,
                "source_sha256": None if args.smoke else SOURCE_SHA,
                "source_and_input_sha256": {**source_hashes, **tracked},
                "input_sha256": inputs, "historical_input_verification": verification,
                "schedule_sha256": hashlib.sha256(schedule_bytes).hexdigest(),
                "config": config, "variant": "flat", "max_seconds": args.max_seconds,
                "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-5, "clip": 1.},
                 "objective": ".5 normalized MSE + .5 normalized MAE + native qloss; fragment cell weighted",
                "rounding": "FP32 native reductions; exact hard vectors + zero-valued ST gradient bridge",
                "qloss": "native padded latent/codebook/commitment mean; equal-size batches <=4; tail isolated",
                "normalization": "native encode_robust independently per fragment; B full-window",
                "paired_exposure": "same full source windows/cells/order/updates; unequal tokens and compute",
                "selection": "fixed endpoint only; wall guard aborts, never promotes checkpoint",
                "pool": "freeze32 minus discovery110 removed IDs only; no temporal roles/independence claim",
                "evaluation": "136 historical checks, 43 files; full tile both; no heldout claim",
                "check_hash": None if args.smoke else digest(paths["checks"]),
                "model_promoted": False}
    write_json(args.output / "manifest.json", manifest)
    start = perf_counter()

    def guard():
        if perf_counter() - start >= args.max_seconds:
            raise TimeoutError("Wall cap reached; run incomplete, no checkpoint selected")

    from finetuning.model_factory import new_model
    arms = ("A", "B") if args.arm == "both" else (args.arm,)
    results = {}
    try:
        for arm in arms:
            guard()
            torch.manual_seed(SEED)
            if arm == "mature":
                from finetuning.run_feasibility_context import build_checkpoint
                entry = protocol["mature"]
                if entry["sha256"] != MATURE_SHA:
                    raise ValueError("Wrong mature receiver")
                checked_input(entry)
                model = build_checkpoint(entry, args.device)
                arm_config = entry["config"]
            else:
                model = new_model(config, args.device).float()
                arm_config = config
                if not args.smoke:
                    model.load_state_dict(source["model_state_dict"], strict=True)
            output = args.output / arm
            output.mkdir()
            torch.cuda.reset_peak_memory_stats()
            initial = model_state_hash(model)
            optimizer = None if arm == "mature" else torch.optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=1e-5)
            cells = tokens = 0
            seen = Counter()
            with (output / "history.jsonl").open("x") as log:
                for index, row in enumerate([] if arm == "mature" else schedule):
                    guard()
                    raw = synthetic[index] if args.smoke else load_raw(row, inputs)
                    torch.cuda.synchronize()
                    tick = perf_counter()
                    stat = train_update(model, optimizer, raw, arm, args.device, guard)
                    torch.cuda.synchronize()
                    cells += stat["cells"]
                    tokens += stat["tokens"]
                    seen[row["sample_path"]] += 1
                    if (index + 1) % 32 == 0 or args.smoke or index + 1 == len(schedule):
                        stat.update(update=index + 1, cumulative_cells=cells, cumulative_tokens=tokens,
                                    elapsed_seconds=perf_counter() - start,
                                    update_seconds=perf_counter() - tick,
                                    allocated_bytes=torch.cuda.memory_allocated(),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                    unique_windows=len(seen),
                                    unique_coverage_sha256=hashlib.sha256(canonical(sorted(seen))).hexdigest())
                        log.write(canonical(stat).decode() + "\n")
                        log.flush()
                    if not args.smoke and (index + 1) % 1024 == 0:
                        temp = output / "last.pt.tmp"
                        torch.save({"config": arm_config, "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(), "update": index + 1,
                                    "cells": cells, "tokens": tokens,
                                    "schedule_sha256": manifest["schedule_sha256"]}, temp)
                        temp.replace(output / "last.pt")
            audit = {"initial_state_sha256": initial, "cells": cells, "tokens": tokens,
                     "updates": 0 if arm == "mature" else len(schedule), "coverage": dict(seen),
                     "elapsed_seconds": perf_counter() - start,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            if args.smoke:
                model.eval()
                from finetuning.joint_anchor_vq import exact_branch_forward
                from finetuning.robust_escape import encode_robust
                audit["check_losses"] = []
                before = model_state_hash(model)
                with torch.no_grad():
                    for raw in synthetic:
                        guard()
                        packet = encode_robust(raw)
                        x = torch.from_numpy(packet.values)[None, None].to(args.device)
                        prediction, qloss, _, _, _ = exact_branch_forward(model, x)
                        loss = .5 * (prediction - x).square().mean() + .5 * (prediction - x).abs().mean() + qloss
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite synthetic check loss")
                        audit["check_losses"].append(float(loss))
                if before != model_state_hash(model):
                    raise AssertionError("Smoke checks changed weights")
            else:
                audit["endpoint"] = evaluate(model, arm_config, checks, maps, scales, inputs,
                                             counts, output / "endpoint", code_bytes, guard)
            torch.cuda.synchronize()
            guard()
            audit.update(elapsed_seconds=perf_counter() - start,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(output / "results.json", audit)
            results[arm] = audit
            del optimizer, model
            torch.cuda.empty_cache()
        guard()
        for path, sha in {**source_hashes, **tracked}.items():
            if digest(path) != sha:
                raise ValueError(f"Frozen dependency changed: {path}")
        if set(results) == {"A", "B"} and (results["A"]["cells"] != results["B"]["cells"]
                                          or results["A"]["coverage"] != results["B"]["coverage"]):
            raise AssertionError("Paired exposure differs")
        write_json(args.output / "results.json", {"arms": results, "model_promoted": False,
                                                 "heldout_claim": False})
        (args.output / ("SMOKE_COMPLETE" if args.smoke else "COMPLETE")).touch()
    except Exception as error:
        write_json(args.output / "INCOMPLETE.json", {"error": str(error),
                   "elapsed_seconds": perf_counter() - start, "model_promoted": False})
        raise


if __name__ == "__main__":
    main()
