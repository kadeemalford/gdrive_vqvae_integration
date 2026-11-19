#!/usr/bin/env python3
"""Zero-update, local RTX 4060 full versus nonoverlapping part8 diagnostic.

The self-contained content scope is model export + decoder source + registry +
geometry + streams. Python, NumPy, Torch and the OS are shared platform software.
No training, remote execution, registry fitting, halos or selectors are provided.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from time import perf_counter
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/smartgrids"))

from finetuning.run_feasibility_context import (
    MATURE_SHA, build_checkpoint, canonical, digest, model_state_hash, resolve,
    score_core, validate_panel, write_json,
)
from finetuning.feasibility_metrics import target_supported_spectra, summarize_target_spectra

MODELS = ("mature_vq", "film_constant_endpoint")
FILM_SHA = "c33767748ff7efadfbec9c65249847feea5ea593b384c31843391953d4f11c62"
MATURE_REGISTRY = "results/experiments/direct_discrete_expanded_20261115/gradient/registry.zlib"
FILM_CHECK = "results/experiments/physical_conditioning_20261115/scientific/constant/check_016384"
OVERLAY = struct.Struct("<4sIII")


def partition(channels, arm):
    if type(channels) is not int or channels < 1 or arm not in ("full", "part8"):
        raise ValueError("Invalid coverage geometry")
    width = channels if arm == "full" else 8
    return [(start, min(start + width, channels)) for start in range(0, channels, width)]


def operation_budget(rows):
    fragments = len(MODELS) * sum(
        len(partition(int(row["channels"]), arm)) for row in rows for arm in ("full", "part8"))
    # FullTileVQWire.encode calls exact_branch_forward, which also calls decoder.
    result = {"encode_calls": fragments, "independent_decode_calls": 2 * fragments,
              "decoder_forwards_including_encode": 3 * fragments,
              "max_encode": 800, "max_decode": 1600}
    if fragments > 800 or 3 * fragments > 1600:
        raise ValueError("Coverage exceeds the frozen encode/decode budget")
    return result


def validate_raw(raw, channels=None):
    if (raw.dtype != np.float32 or raw.ndim != 2 or not raw.size
            or (channels is not None and raw.shape != (channels, 1000))
            or not np.isfinite(raw).all()):
        raise ValueError("Finite nonempty historical float32 tiles required; fail closed on nonfinites")


def pack_zero_overlay(frame, raw):
    """Keep native encode_robust input unchanged; transmit exact zero positions/bits."""
    validate_raw(raw)
    positions = np.flatnonzero(raw.ravel() == 0).astype("<u4")
    bits = raw.view(np.uint32).ravel()[positions].astype("<u4")
    payload = zlib.compress(positions.tobytes() + bits.tobytes(), 6)
    body = frame + payload
    return OVERLAY.pack(b"FZC1", len(frame), len(positions), zlib.crc32(body)) + body


def decode_zero_overlay(data, receiver):
    if len(data) < OVERLAY.size:
        raise ValueError("Truncated zero overlay")
    magic, size, count, checksum = OVERLAY.unpack_from(data)
    body = data[OVERLAY.size:]
    if magic != b"FZC1" or not 0 < size < len(body) or zlib.crc32(body) != checksum:
        raise ValueError("Corrupt zero overlay")
    decoded = receiver.decode(body[:size]).copy()
    validate_raw(decoded)
    if count > decoded.size:
        raise ValueError("Invalid zero overlay count")
    codec = zlib.decompressobj()
    payload = codec.decompress(body[size:], 8 * count + 1)
    if (len(payload) != 8 * count or not codec.eof
            or codec.unused_data or codec.unconsumed_tail):
        raise ValueError("Invalid zero overlay length")
    positions = np.frombuffer(payload[:4 * count], "<u4")
    bits = np.frombuffer(payload[4 * count:], "<u4")
    if (np.any(positions >= decoded.size) or np.any(positions[1:] <= positions[:-1])
            or np.any(bits & np.uint32(0x7fffffff))):
        raise ValueError("Invalid zero overlay positions/bits")
    decoded.view(np.uint32).ravel()[positions] = bits
    return decoded


def encode_coverage(raw, sender, arm, group=None):
    from compression.partition_bundle import pack_bundle
    validate_raw(raw)
    fragments = []
    for start, stop in partition(len(raw), arm):
        tile = np.ascontiguousarray(raw[start:stop])
        frame = sender.encode(tile) if group is None else sender.encode(tile, group)
        fragments.append((0, start, stop - start, pack_zero_overlay(frame, tile)))
    return pack_bundle(raw.shape, fragments)


def decode_coverage(bundle, receiver):
    """Receiver interface: transmitted bytes and an installed frozen decoder only."""
    from compression.partition_bundle import unpack_bundle
    shape, fragments = unpack_bundle(bundle)
    result = np.empty(shape, np.float32)
    for kind, start, rows, payload in fragments:
        if kind != 0:
            raise ValueError("This protocol forbids exact-only bypass fragments")
        restored = decode_zero_overlay(payload, receiver)
        if restored.shape != (rows, shape[1]):
            raise ValueError("Decoded fragment geometry differs")
        result[start:start + rows] = restored
    return result


def merge_spectra(destination, statistics):
    for band, values in statistics.items():
        dest = destination.setdefault(band, {})
        for key, value in values.items():
            dest[key] = dest.get(key, 0) + value


def coverage_guards(summary, tile_records):
    """Pool raw SSE and target-fixed spectra; do not average spectral ratios."""
    groups = {}
    for record in summary["records"]:
        group = groups.setdefault(record["group_id"], {
            "sse": 0., "max_error": 0., "channel_nrmse_star": [],
            "spectral_statistics": {}, "spectral_unavailable_windows": 0})
        for channel in record["channels"]:
            group["sse"] += channel["sse"]
            group["max_error"] = max(group["max_error"], channel["max_error"])
            group["channel_nrmse_star"].append(channel["nrmse_star"])
    for tile in tile_records:
        merge_spectra(groups[tile["group_id"]]["spectral_statistics"], tile["spectral_statistics"])
    for group in groups.values():
        channels = group.pop("channel_nrmse_star")
        group["p90_channel_nrmse_star"] = float(np.quantile(channels, .9))
        group["max_channel_nrmse_star"] = float(max(channels))
        group["spectral"] = summarize_target_spectra(group["spectral_statistics"])
    return groups


def no_regression(candidate, baseline, tolerance=.01):
    if candidate is None or baseline is None:
        return None
    if not np.isfinite(candidate) or not np.isfinite(baseline):
        return None
    return bool(candidate <= baseline * (1 + tolerance)) if baseline > 0 else bool(candidate <= baseline)


def paired_gate(full, part):
    """Only the SAME model's part8/full pair; unavailable guards never pass."""
    checks = {"macro_improvement_at_least_5pct":
              bool(full["macro_families"] > 0
                    and part["macro_families"] <= .95 * full["macro_families"])}
    checks["pooled_physical_sse_not_increased"] = (
        sum(g["sse"] for g in part["coverage_guards"].values())
        <= sum(g["sse"] for g in full["coverage_guards"].values()))
    for contract in ("shared_model", "self_contained"):
        checks[f"{contract}_bytes_not_increased"] = (
            part["rate"][contract]["bytes"] <= full["rate"][contract]["bytes"])
    for family, baseline in full["families"].items():
        checks[f"family:{family}"] = no_regression(part["families"].get(family), baseline)
    unavailable = []
    for group, baseline in full["coverage_guards"].items():
        candidate = part["coverage_guards"][group]
        checks[f"group_p90:{group}"] = no_regression(
            candidate["p90_channel_nrmse_star"], baseline["p90_channel_nrmse_star"])
        for band, spectral in baseline["spectral"].items():
            key = f"spectral:{group}:{band}"
            # If target energy is zero the ratio is undefined; compare raw error energy.
            metric = "relative_complex_rmse" if spectral["target_energy"] > 0 else "error_energy"
            checks[key] = no_regression(candidate["spectral"][band][metric], spectral[metric])
            if spectral["target_energy"] == 0:
                unavailable.append({"group": group, "band": band,
                                    "metric": "relative_complex_rmse", "fallback": "error_energy",
                                    "reason": "zero_target_energy"})
            if spectral["supported_target_energy"] == 0:
                unavailable.append({"group": group, "band": band,
                                    "metric": "supported_phase", "reason": "no_target_support"})
    return {"passed": all(value is True for value in checks.values()), "checks": checks,
            "unavailable": unavailable, "starts_f4": False, "model_promoted": False}


def validate_weights(weights, historical):
    if set(weights) != set(MODELS):
        raise ValueError("Manifest must contain exactly mature_vq and film_constant_endpoint")
    for name, expected in zip(MODELS, (MATURE_SHA, FILM_SHA)):
        entry = weights[name]
        if entry["sha256"] != expected or digest(resolve(entry["path"])) != expected:
            raise ValueError(f"Wrong prescribed receiver identity: {name}")
        if not isinstance(entry.get("config"), dict):
            raise ValueError("Complete config required in weights manifest")
        if entry["config"] != historical["weights"][name]["config"]:
            raise ValueError("Receiver config differs from replay F1")
    if weights["film_constant_endpoint"]["config"].get("method") != "constant":
        raise ValueError("Only the constant FiLM endpoint is allowed")


def load_export(entry, name, device):
    import torch
    saved = torch.load(resolve(entry["path"]), map_location="cpu", weights_only=True)
    if saved.get("config") != entry["config"] or "model_state_dict" not in saved:
        raise ValueError("Receiver export config/state differs")
    optimizer_keys = [key for key in saved if "optim" in key.lower()]
    if optimizer_keys:
        raise ValueError("Use the real weights-only receiver export, not a training checkpoint")
    if name == "mature_vq":
        model = build_checkpoint({**entry, "update": saved.get("update")}, device)
    else:
        from finetuning.physical_conditioning import build, condition, set_condition
        model = build(saved["config"], saved["model_state_dict"], device)
        set_condition(model, condition(saved["config"]["plan"], "constant", None))
    return model.eval().float().requires_grad_(False), {
        "keys": sorted(saved), "contains_optimizer": False,
        "binary_update": saved.get("update"), "historical_endpoint_update": 16384 if name != "mature_vq" else None,
        "content": "real historical receiver export, not claimed to be minimal decoder weights"}


def make_wire(model, name, registry, config):
    from compression.full_tile_vq_stream import FullTileVQWire
    from compression.conditioned_vq_stream import ConditionedVQWire
    if name == "mature_vq":
        return FullTileVQWire.from_setup(model, registry)
    return ConditionedVQWire.from_setup(model, registry, config)


def rate_ledger(frame_bytes, geometry_bytes, registry_bytes, model_bytes, code_bytes, contract_bytes, cells):
    shared = frame_bytes + geometry_bytes
    installation = registry_bytes + model_bytes + code_bytes + contract_bytes
    return {"components": {"bundles_including_zero_overlays": frame_bytes,
                           "geometry": geometry_bytes, "registry": registry_bytes,
                           "model_export": model_bytes, "decoder_sources": code_bytes,
                           "decoder_contract": contract_bytes},
            "installed_shared_bytes": installation,
            "shared_model": {"bytes": shared, "bpc": 8 * shared / cells},
            "self_contained": {"bytes": shared + installation,
                               "bpc": 8 * (shared + installation) / cells},
            "platform_shared": "Python/NumPy/Torch and OS; not bundled",
            "scope": "receiver/model export + decoder sources + registry + geometry + stream"}


def preflight(args):
    """Hash all sources, weights, registry and panel NPZ before any forward."""
    from tools.freeze_feasibility_metadata import select_f1_panel
    metadata = json.loads(args.metadata.read_text())
    rows, _ = validate_panel(metadata)
    replay = json.loads((args.f1_replay / "protocol.json").read_text())
    result = json.loads((args.f1_replay / "results.json").read_text())
    if (not (args.f1_replay / "COMPLETE").is_file() or result.get("forward_passes") != 414
            or not result.get("weights_unchanged") or result.get("parameter_updates") != 0):
        raise ValueError("Completed, unchanged-weight replay F1 required")
    if replay["metadata_sha256"] != digest(args.metadata) or replay["panel"] != metadata["f1_panel"]:
        raise ValueError("Panel/freeze differs from replay F1")
    for value in metadata["input_sha256"].values():
        if digest(resolve(value["path"])) != value["sha256"]:
            raise ValueError("Frozen metadata dependency changed")
    for module in ("feasibility_context", "feasibility_metrics"):
        if digest(ROOT / "finetuning" / (module + ".py")) != metadata[module + "_sha256"]:
            raise ValueError("Frozen planner/metrics changed")
    checks = json.loads(resolve(metadata["input_sha256"]["checks"]["path"]).read_text())
    manifest = resolve(metadata["input_sha256"]["manifest"]["path"])
    with manifest.open(newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    if select_f1_panel(manifest_rows, checks["checks"]) != rows:
        raise ValueError("Panel differs from historical selection")
    by_sample = {r["sample_path"]: r for r in manifest_rows}
    acquired = [{**by_sample[r["sample_path"]], **r} for r in rows]
    reference = resolve(metadata["input_sha256"]["manifest_reference"]["path"])
    maps_path = manifest.with_name("channel_maps.jsonl")
    scales_path = reference.with_name("reference_scales.json")
    scales_data = json.loads(scales_path.read_text())
    if (digest(maps_path) != replay["channel_maps_sha256"]
            or digest(maps_path) != json.loads(reference.read_text())["channel_maps_sha256"]
            or digest(scales_path) != replay["reference_scales_sha256"]
            or scales_data["protocol_sha256"] != digest(reference)):
        raise ValueError("Maps/reference scales lineage differs")
    maps = {r["channel_map_id"]: r for r in map(json.loads, maps_path.read_text().splitlines())}
    scales = scales_data["scales"]
    inputs = {}
    for row in acquired:
        path = row["sample_path"]
        sha = digest(resolve(path))
        if sha != replay["input_sha256"].get(path) or sha != checks["input_sha256"].get(path):
            raise ValueError("NPZ differs from historical freeze and replay F1")
        inputs[path] = sha
        mapping = maps[row["channel_map_id"]]
        if mapping["drive_id"] != row["drive_id"] or len(mapping["source_channel_indices"]) != int(row["channels"]):
            raise ValueError("Physical channel identity differs")
        if float(row["dt_s"]) != 1 / 20000 or not np.isfinite(scales[row["group_id"]]) or scales[row["group_id"]] <= 0:
            raise ValueError("Unexpected sample rate or frozen scale")
    weights = json.loads(args.weights_manifest.read_text())
    validate_weights(weights, replay)
    historical_film = json.loads(args.film_results.read_text())
    if (historical_film["model_sha256"] != FILM_SHA or historical_film["update"] != 16384
            or historical_film.get("condition_missing") is not False):
        raise ValueError("Historical FiLM results do not certify the constant receiver")
    registry_paths = {
        MODELS[0]: args.mature_registry or resolve(weights[MODELS[0]]["registry"]),
        MODELS[1]: args.film_registry or resolve(weights[MODELS[1]]["registry"])}
    registries = {name: path.read_bytes() for name, path in registry_paths.items()}
    # Registry parsing needs no model and no forward; never fit on the panel.
    for name, registry in registries.items():
        if hashlib.sha256(registry).hexdigest() != weights[name]["registry_sha256"]:
            raise ValueError("Registry hash differs from the audited weights manifest")
        raw = zlib.decompress(registry)
        metadata_registry, counts = raw.split(b"\n", 1)
        metadata_registry = json.loads(metadata_registry)
        if (metadata_registry.get("kind") != "full-tile-vq-fak1-v1"
                or metadata_registry.get("K") != 2048 or len(counts) != 2048 * 8):
            raise ValueError("Historical registry schema differs")
        # FiLM registry is explicitly bound to the real receiver export.
        if metadata_registry["identifier"] != weights[name]["sha256"]:
            raise ValueError("Registry is not bound to the historical receiver")
    return acquired, metadata, replay, weights, registries, maps, scales, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True, help="Scientific report/protocol 109, frozen before execution")
    parser.add_argument("--metadata", type=Path, default=Path(
        "results/experiments/feasibility_definition_20261117/metadata_freeze.json"))
    parser.add_argument("--f1-replay", type=Path, default=Path(
        "results/experiments/feasibility_context_replay_20261117"))
    parser.add_argument("--mature-registry", type=Path, help="Override audited manifest path, same hash required")
    parser.add_argument("--film-registry", type=Path, help="Recovered registry, audited manifest hash required")
    parser.add_argument("--film-results", type=Path, default=Path(FILM_CHECK) / "normal/results.json")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a new directory; no overwrite or resume")
    rows, metadata, replay, weights, registries, maps, scales, inputs = preflight(args)
    args.mature_registry = args.mature_registry or resolve(weights[MODELS[0]]["registry"])
    args.film_registry = args.film_registry or resolve(weights[MODELS[1]]["registry"])
    budget = operation_budget(rows)
    import torch
    if not torch.cuda.is_available() or "RTX 4060" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This protocol requires the local RTX 4060, cuda:0 only")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(109)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Validate every prepared tile (including nonfinite fail-closed) before forwarding.
    audits = []
    for row in rows:
        with np.load(resolve(row["sample_path"]), allow_pickle=False) as data:
            raw = data["data"]
            validate_raw(raw, int(row["channels"]))
            audits.append({"sample_path": row["sample_path"], "nonfinite_cells": 0,
                           "positive_zero_cells": int(np.sum(raw.view(np.uint32) == 0)),
                           "negative_zero_cells": int(np.sum(raw.view(np.uint32) == 0x80000000))})
    from finetuning.fit_lifting_allocation import transport
    from finetuning.streaming_physical_summary import StreamingPhysicalSummary
    geometry = transport(rows, maps)
    sources = sorted(set(ROOT.glob("finetuning/*.py")) | set(ROOT.glob("compression/*.py"))
                     | set((ROOT / "vendor/smartgrids").rglob("*.py")))
    import vqvae
    vendor = Path(next(iter(vqvae.__path__))).resolve()
    if not (vendor / "train_vqvae.py").is_file() or not (vendor / "evaluate_vqvae.py").is_file():
        raise RuntimeError("Historical native model sources are required for the decoder installation")
    sources = sorted(set(sources) | set(vendor.rglob("*.py")))
    tracked = {str(path): digest(path) for path in sources}
    for path in (args.protocol, args.metadata, args.weights_manifest, args.mature_registry,
                 args.film_registry, args.film_results, args.f1_replay / "protocol.json",
                 args.f1_replay / "results.json", args.f1_replay / "records.jsonl"):
        tracked[str(path)] = digest(path)
    for entry in weights.values():
        tracked[str(resolve(entry["path"]))] = entry["sha256"]
    for path, sha in inputs.items():
        tracked[str(resolve(path))] = sha
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "decoder_sources"
    for path in sources:
        relative = (path.relative_to(ROOT) if path.is_relative_to(ROOT)
                    else Path("vendor/vqvae") / path.relative_to(vendor))
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    code_bytes = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
    protocol = {"kind": "feasibility-full-coverage-v1", "parameter_updates": 0,
                "starts_f4": False, "device": torch.cuda.get_device_name(0),
                "operation_budget": budget, "source_and_input_sha256": tracked,
                "panel": rows, "input_audit": audits, "weights": weights,
                "registry_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in registries.items()},
                "normalization": "native encode_robust, unchanged input; paid exact signed-zero overlay",
                "nonfinite_policy": "fail closed before any forward", "arms": ["full", "part8"],
                "replay": "two independent decode-only passes, bit-exact; same frozen model permitted",
                 "f1_check": "report full/native core SSE versus historical replay; rtol 1e-5, atol 0; cross-device diagnostic only",
                "platform": "Python/NumPy/Torch/OS shared; model, sources, registry installation charged"}
    write_json(args.output / "protocol.json", protocol)
    shutil.copy2(args.protocol, args.output / "scientific_protocol")
    f1 = [json.loads(line) for line in (args.f1_replay / "records.jsonl").read_text().splitlines()]
    summaries, identities, model_audits, f1_comparisons = {}, [], {}, []
    encode_count, decode_count = 0, 0
    for name in MODELS:
        model, model_audit = load_export(weights[name], name, args.device)
        initial_hash = model_state_hash(model)
        sender = make_wire(model, name, registries[name], weights[name]["config"])
        receiver = make_wire(model, name, registries[name], weights[name]["config"])
        installation = args.output / name / "installation"
        installation.mkdir(parents=True)
        shutil.copy2(resolve(weights[name]["path"]), installation / "receiver.pt")
        (installation / "registry.zlib").write_bytes(registries[name])
        contract = canonical({"model_sha256": weights[name]["sha256"], "model": name,
                              "registry_sha256": hashlib.sha256(registries[name]).hexdigest(),
                              "sources": "../../decoder_sources", "platform": protocol["platform"],
                              "stream_format": "ordered FPB1/FZC1 bundles plus transport.zlib"})
        (installation / "decoder_contract.json").write_bytes(contract)
        summaries[name] = {}
        for arm in ("full", "part8"):
            output = args.output / name / arm
            output.mkdir()
            (output / "transport.zlib").write_bytes(geometry)
            summary = StreamingPhysicalSummary(scales, maps)
            tiles = []
            torch.cuda.reset_peak_memory_stats()
            for index, row in enumerate(rows):
                if digest(resolve(row["sample_path"])) != inputs[row["sample_path"]]:
                    raise ValueError("NPZ changed after preflight")
                with np.load(resolve(row["sample_path"]), allow_pickle=False) as data:
                    raw = data["data"].copy()
                torch.cuda.synchronize()
                start = perf_counter()
                bundle = encode_coverage(raw, sender, arm, row["group_id"] if name == MODELS[1] else None)
                torch.cuda.synchronize()
                encode_seconds = perf_counter() - start
                fragments = len(partition(len(raw), arm))
                encode_count += fragments
                path = output / f"{index:03d}.fpb"
                path.write_bytes(bundle)
                start = perf_counter()
                restored = decode_coverage(bundle, receiver)
                torch.cuda.synchronize()
                decode_seconds = perf_counter() - start
                repeated = decode_coverage(path.read_bytes(), receiver)
                decode_count += 2 * fragments
                if restored.tobytes() != repeated.tobytes():
                    raise AssertionError("Independent decode-only replay is not bit exact")
                if not np.isfinite(restored).all() or raw[raw == 0].tobytes() != restored[raw == 0].tobytes():
                    raise AssertionError("Nonfinite output or signed-zero preservation failure")
                if arm == "full":
                    plan = metadata["f1_panel"]["context_plans"][row["sample_path"]]
                    window = next(w for w in plan["windows"] if w["full_tile"])
                    channels = maps[row["channel_map_id"]]["source_channel_indices"][plan["core_start"]:plan["core_stop"]]
                    core = score_core(raw, restored, plan, window, scales[row["group_id"]], channels)
                    previous = next(r for r in f1 if r["model"] == name
                                    and r["sample_path"] == row["sample_path"]
                                    and r["context"]["full_tile"] and r["normalization"] == "native_per_context")
                    original_sse = previous["metrics"]["sse"]
                    f1_comparisons.append({"model": name, "sample_path": row["sample_path"],
                                           "f1_core_sse": original_sse, "local_core_sse": core["sse"],
                                           "within_rtol_1e_5": bool(np.isclose(
                                               core["sse"], original_sse, rtol=1e-5, atol=0))})
                summary.update(row, raw, restored, len(bundle))
                spectral = target_supported_spectra(raw, restored, 1 / 20000, scales[row["group_id"]])
                error = restored.astype(np.float64) - raw.astype(np.float64)
                tiles.append({"sample_path": row["sample_path"], "group_id": row["group_id"],
                              "bundle_bytes": len(bundle), "bpc": 8 * len(bundle) / raw.size,
                               "sha256": digest(path), "sse": float(np.sum(error ** 2)),
                               "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                              "max_error": float(np.max(np.abs(error))),
                              "spectral_statistics": spectral,
                              "spectral": summarize_target_spectra(spectral),
                              "spectral_unavailable_windows": 0, "fragments": fragments})
                identities.append({"model": name, "arm": arm, "sample_path": row["sample_path"],
                                   "decode_replay_bit_exact": True})
            result = summary.result(len(geometry))
            result["coverage_guards"] = coverage_guards(result, tiles)
            result["tiles"] = tiles
            result["timing"] = {"encode_seconds": sum(t["encode_seconds"] for t in tiles),
                                "decode_seconds": sum(t["decode_seconds"] for t in tiles),
                                "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                                "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved()}
            result["rate"] = rate_ledger(result["frame_bytes"], len(geometry), len(registries[name]),
                                          (installation / "receiver.pt").stat().st_size, code_bytes,
                                          len(contract), result["cells"])
            summaries[name][arm] = result
            write_json(output / "results.json", result)
        if model_state_hash(model) != initial_hash:
            raise AssertionError("Frozen model state changed")
        model_audits[name] = {**model_audit, "state_sha256": initial_hash, "unchanged": True}
        del sender, receiver, model
        torch.cuda.empty_cache()
    for path, sha in tracked.items():
        if digest(path) != sha:
            raise ValueError("Frozen dependency changed during execution")
    if encode_count != budget["encode_calls"] or decode_count != budget["independent_decode_calls"]:
        raise AssertionError("Executed operation counts differ from preflight")
    write_json(args.output / "results.json", {
        "models": summaries, "gates": {name: paired_gate(s["full"], s["part8"]) for name, s in summaries.items()},
        "parameter_updates": 0, "starts_f4": False, "model_promoted": False,
        "historical_seen_train_checks_not_independent": True, "model_audits": model_audits,
        "encode_calls": encode_count, "decode_calls": decode_count,
         "decoder_forwards_including_encode": encode_count + decode_count,
         "cross_device_f1_comparisons": f1_comparisons,
        "decode_replay": identities, "protocol_sha256": digest(args.output / "protocol.json")})
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
