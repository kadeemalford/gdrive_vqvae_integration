#!/usr/bin/env python3
"""Frozen, train-only report-107 context diagnostic (not a codec or training run)."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/smartgrids"))
from finetuning.feasibility_context import plan_context_windows
from finetuning.feasibility_metrics import target_supported_spectra, summarize_target_spectra
from finetuning.prospective_metrics import PhysicalAccumulator
from finetuning.reversible_normalization import HEADER, parse_side_stream
from finetuning.robust_escape import encode_robust

MODELS = ("mature_vq", "diversity_10_endpoint", "film_constant_endpoint")
NORMS = ("native_per_context", "fixed_full_tile_affine_diagnostic_only")
MATURE_SHA = "b943db9f5cf6418e61ffca1814d788ac23fd6111468a425a918ba7d7cb5bc593"
SCALES = "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def resolve(path):
    path = Path(path)
    if not path.is_absolute():
        return ROOT / path
    if path.exists():
        return path
    # Frozen metadata can have been prepared on a different checkout host.
    for prefix in ("results", "data", "config", "finetuning", "tools"):
        if prefix in path.parts:
            return ROOT.joinpath(*path.parts[path.parts.index(prefix):])
    return path


def validate_weights_manifest(manifest):
    """Accept the CPU inspection map; never infer identity from tensor shapes."""
    if set(manifest) != set(MODELS):
        raise ValueError("Weights manifest must contain exactly the three prescribed models")
    for name in MODELS:
        entry = manifest[name]
        sha = entry.get("sha256", "")
        if (not isinstance(entry.get("path"), str) or not entry["path"]
                or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
                or not isinstance(entry.get("config"), dict) or "update" not in entry):
            raise ValueError(f"Invalid weights manifest entry: {name}")
        config = entry["config"]
        base = config.get("base_config", config)
        if (base.get("latent_geometry") != "spatial4-temporal4"
                or base.get("num_embeddings") != 2048
                or base.get("embedding_dim") != 128):
            raise ValueError(f"Unexpected fixed-model geometry/config: {name}")
        if name == "mature_vq":
            if (sha != MATURE_SHA or entry["update"] is not None
                    or config.get("architecture", "native-v1") != "native-v1"):
                raise ValueError("Mature reference identity differs")
        elif name == "diversity_10_endpoint":
            if (entry["update"] != 8192 or config.get("architecture") != "dense-full-tile-product-v1"
                    or config.get("variant") != "flat"):
                raise ValueError("Diversity endpoint identity differs")
        elif (entry["update"] != 16384 or config.get("architecture") != "physical-film-v1"
              or config.get("method") != "constant"):
            raise ValueError("Constant FiLM endpoint identity differs")
    return manifest


def validate_panel(metadata):
    if metadata.get("kind") != "feasibility-metadata-freeze-v1":
        raise ValueError("Expected the frozen feasibility metadata")
    panel = metadata["f1_panel"]
    rows = panel["rows"]
    if (len(rows) != 20 or len({r["drive_id"] for r in rows}) != 20
            or len({r["sample_path"] for r in rows}) != 20
            or len(Counter(r["group_id"] for r in rows)) != 10
            or set(Counter(r["group_id"] for r in rows).values()) != {2}):
        raise ValueError("Expected exactly two distinct files in each of ten groups")
    for row in rows:
        path = Path(row["sample_path"])
        if (row.get("split") != "train" or path.is_absolute() or ".." in path.parts
                or path.suffix != ".npz"
                or row.get("family") in ("EDAS Polyimide", "EDAS Acrylate")
                or row["group_id"].split("/")[0] not in
                ("doped", "multicore", "multimode", "single_mode")):
            raise ValueError("Only frozen internal train-panel NPZ inputs are permitted")
        expected = plan_context_windows(int(row["channels"]))
        if panel["context_plans"][row["sample_path"]] != expected:
            raise ValueError("Frozen context plan differs from the shared planner")
    forwards = 3 * 2 * sum(len(p["windows"]) for p in panel["context_plans"].values())
    if forwards != 414 or forwards > 480:
        raise ValueError(f"Frozen forward budget differs: {forwards}, expected 414")
    return rows, forwards


@dataclass
class Packet:
    values: np.ndarray
    valid: np.ndarray
    mean: float
    scale: float
    threshold: float
    indices: np.ndarray
    escaped: np.ndarray

    def side_stream(self):
        # Diagnostic serialization only. FCD1 additionally permits exact special bits.
        return (HEADER.pack(b"FCD1", *self.values.shape, self.mean, self.scale,
                            self.threshold, len(self.indices))
                + self.indices.astype("<u4").tobytes()
                + self.escaped.astype("<f4").tobytes())


def normalize(raw):
    """Match historical robust normalization, adding exact special/signed-zero escapes."""
    raw = np.asarray(raw)
    if raw.dtype != np.float32 or raw.ndim != 2 or not raw.size:
        raise ValueError("Expected nonempty 2D float32 source")
    finite = np.isfinite(raw)
    if finite.all():
        encoded = encode_robust(raw)
        _, mean, scale, indices, escaped = parse_side_stream(encoded.side_stream)
        valid = encoded.valid.copy()
        values = encoded.values.copy()
        threshold = HEADER.unpack_from(encoded.side_stream)[5]
    else:
        # Specials never enter quantiles or affine statistics.
        from finetuning.robust_escape import robust_threshold
        threshold = robust_threshold(raw[finite].reshape(1, -1)) if finite.any() else 1e20
        valid = finite & (np.abs(raw.astype(np.float64)) <= threshold)
        ordinary = raw[valid].astype(np.float64)
        mean = float(ordinary.mean()) if ordinary.size else 0.
        scale = float(ordinary.std()) + 1e-6 if ordinary.size else 1.
        values = np.zeros(raw.shape, np.float32)
        values[valid] = ((ordinary - mean) / scale).astype(np.float32)
    valid &= ~(raw == 0)  # Preserve both zero signs exactly; no uncoded exceptions.
    # Exact zero escapes must not change the historical model input. A zero
    # sample generally normalizes to -mean/scale, not zero.
    indices = np.flatnonzero(~valid).astype("<u4")
    escaped = raw.ravel()[indices].copy()
    return Packet(values, valid, mean, scale, threshold, indices, escaped)


def slice_packet(full, start, stop):
    """Slice full normalized tensors and relocate its exact escapes; never refit."""
    if not 0 <= start < stop <= full.values.shape[0]:
        raise ValueError("Context must be contained in the source; no manufactured channels")
    samples = full.values.shape[1]
    selected = (full.indices >= start * samples) & (full.indices < stop * samples)
    return Packet(full.values[start:stop].copy(), full.valid[start:stop].copy(),
                  full.mean, full.scale, full.threshold,
                  (full.indices[selected] - start * samples).astype("<u4"),
                  full.escaped[selected].copy())


def physical_inverse(prediction, packet):
    prediction = np.asarray(prediction)
    if prediction.shape != packet.values.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid model reconstruction")
    # Match the historical inverse: affine arithmetic in float64, stored result float32.
    restored = prediction.astype(np.float64) * packet.scale + packet.mean
    restored.ravel()[packet.indices] = 0
    if np.any(np.abs(restored) > np.finfo(np.float32).max):
        raise ValueError("Physical reconstruction overflows float32")
    restored = restored.astype(np.float32)
    restored.ravel()[packet.indices] = packet.escaped
    if restored.ravel()[packet.indices].tobytes() != packet.escaped.tobytes():
        raise AssertionError("Exact escape payload changed")
    return restored


def score_core(raw, restored, plan, window, reference_scale, physical_channels):
    """Score only the identical source core; arrays are [channel, time]."""
    a, b = plan["core_start"], plan["core_stop"]
    c, d = window["score_start_in_window"], window["score_stop_in_window"]
    target, prediction = raw[a:b], restored[c:d]
    if target.shape != prediction.shape or target.shape[0] != len(physical_channels):
        raise ValueError("Scoring axes/core identity differ")
    channels = []
    total_sse, total_cells = 0., 0
    spectral = {}
    for index, (x, y) in enumerate(zip(target, prediction)):
        finite = np.isfinite(x)
        if not np.isfinite(y[finite]).all():
            raise ValueError("Nonfinite prediction on a finite target")
        special = ~finite | (x == 0)
        if x[special].tobytes() != y[special].tobytes():
            raise AssertionError("Special/zero target payload changed")
        acc = PhysicalAccumulator()
        if finite.any():
            acc.update(x[finite], y[finite])
            metrics = acc.result(reference_scale)
            total_sse += acc.sse
            total_cells += acc.count
        else:
            metrics = {"cells": 0, "rmse": None}
        available = bool(finite.all() and np.isfinite(y).all())
        stats = target_supported_spectra(x[None], y[None], 1 / 20000, reference_scale) if available else None
        if stats is not None:
            for band, values in stats.items():
                dest = spectral.setdefault(band, {key: 0 for key in values})
                for key, value in values.items():
                    dest[key] += value
        channels.append({"source_channel_index": physical_channels[index],
                         "physical": metrics, "physical_statistics": asdict(acc),
                         "constant_target": bool(finite.all() and np.ptp(x.astype(np.float64)) == 0),
                         "nonfinite_cells": int((~finite).sum()),
                         "exact_special_zero_cells": int(special.sum()),
                         "spectral_available": available,
                         "spectral_unavailable_reason": None if available else "nonfinite_target",
                         "spectral_statistics": stats,
                         "spectral": summarize_target_spectra(stats) if stats else None})
    return {"rmse": math.sqrt(total_sse / total_cells) if total_cells else None,
            "finite_cells": total_cells, "sse": total_sse,
            "channels": channels,
            "spectral": summarize_target_spectra(spectral) if spectral else None}


def context_pairs(records):
    """All signed contrasts; the gate requires the SAME pair/norm/model in two groups."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row["model"], row["normalization"], row["sample_path"]].append(row)
    pairs = []
    qualifying = defaultdict(lambda: defaultdict(set))
    for (model, norm, sample), entries in sorted(grouped.items()):
        for left, right in combinations(sorted(entries, key=lambda r: r["context_channels"]), 2):
            x, y = left["metrics"]["rmse"], right["metrics"]["rmse"]
            relative = (y - x) / x if x is not None and x > 0 and y is not None else None
            pair = (left["context_channels"], right["context_channels"])
            record = {"model": model, "normalization": norm, "sample_path": sample,
                      "drive_id": left["drive_id"], "group_id": left["group_id"],
                      "context_pair": list(pair), "rmse_left": x, "rmse_right": y,
                      "absolute_difference": y - x if x is not None and y is not None else None,
                      "relative_difference": relative,
                      "relative_difference_defined": relative is not None,
                      "threshold_met": relative is not None and abs(relative) >= .05}
            pairs.append(record)
            if record["threshold_met"]:
                qualifying[model, norm, pair][left["group_id"]].add(left["drive_id"])
    gates = []
    all_keys = {(p["model"], p["normalization"], tuple(p["context_pair"])) for p in pairs}
    for model, norm, pair in sorted(all_keys):
        groups = qualifying[model, norm, pair]
        passed = sorted(group for group, files in groups.items() if len(files) >= 2)
        gates.append({"model": model, "normalization": norm, "context_pair": list(pair),
                      "qualifying_files_by_group": {g: sorted(f) for g, f in sorted(groups.items())},
                      "qualifying_groups": passed, "descriptive_gate_met": len(passed) >= 2})
    return pairs, gates


def build_checkpoint(entry, device):
    import torch
    path = resolve(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint hash differs: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != entry["config"] or checkpoint.get("update") != entry["update"]:
        raise ValueError(f"Checkpoint config/update differs: {path}")
    config = checkpoint["config"]
    architecture = config.get("architecture", "native-v1")
    if architecture == "physical-film-v1":
        from finetuning.physical_conditioning import build, set_condition, condition
        model = build(config, checkpoint["model_state_dict"], device)
        set_condition(model, condition(config["plan"], "constant", None))
    elif architecture == "dense-full-tile-product-v1":
        from finetuning.dense_product_vq import build
        model = build(config, checkpoint["model_state_dict"], device)
    else:
        from finetuning.model_factory import build_model
        model = build_model({**config, "exact_eval_quantization": True}, checkpoint, device)
    return model.eval().float().requires_grad_(False)


def validate_checkpoint_identity(entry):
    """CPU-only identity check for ALL checkpoints before any model forward."""
    import torch
    path = resolve(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError("Checkpoint SHA256 differs from manifest")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("config") != entry["config"]
            or checkpoint.get("update") != entry["update"]
            or "model_state_dict" not in checkpoint):
        raise ValueError(f"Checkpoint config/update identity differs: {path}")


def model_state_hash(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        result.update(canonical([name, str(value.dtype), list(value.shape)]))
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("metadata_freeze.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("--output must be a new directory (no resume or overwrite)")
    metadata = json.loads(args.metadata.read_text())
    weights = validate_weights_manifest(json.loads(args.weights_manifest.read_text()))
    rows, expected = validate_panel(metadata)
    # Verify frozen metadata and its exact historical panel BEFORE touching NPZ.
    from tools.freeze_feasibility_metadata import select_f1_panel
    import csv
    for value in metadata["input_sha256"].values():
        if digest(resolve(value["path"])) != value["sha256"]:
            raise ValueError(f"Frozen metadata input changed: {value['path']}")
    config_path = resolve(metadata["input_sha256"]["config"]["path"])
    config = json.loads(config_path.read_text())
    if (digest(config_path) != metadata["config_sha256"]
            or config["f1"]["models"] != list(MODELS)
            or config["f1"]["max_forward_passes"] != 480):
        raise ValueError("Frozen F1 definition differs")
    for module in ("feasibility_context", "feasibility_metrics"):
        if digest(ROOT / "finetuning" / (module + ".py")) != metadata[module + "_sha256"]:
            raise ValueError("Frozen planner/metrics source differs")
    checks = json.loads(resolve(metadata["input_sha256"]["checks"]["path"]).read_text())
    manifest_path = resolve(metadata["input_sha256"]["manifest"]["path"])
    with manifest_path.open(newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    if select_f1_panel(manifest_rows, checks["checks"]) != rows:
        raise ValueError("Frozen panel differs from historical selection")
    reference_path = resolve(metadata["input_sha256"]["manifest_reference"]["path"])
    scales_path = reference_path.with_name("reference_scales.json")
    scales_data = json.loads(scales_path.read_text())
    if scales_data.get("protocol_sha256") != digest(reference_path):
        raise ValueError("Frozen scale lineage differs")
    scales = scales_data["scales"]
    maps_path = manifest_path.with_name("channel_maps.jsonl")
    reference = json.loads(reference_path.read_text())
    if digest(maps_path) != reference["channel_maps_sha256"]:
        raise ValueError("Physical channel maps differ")
    maps = {r["channel_map_id"]: r for r in map(json.loads, maps_path.read_text().splitlines())}
    inputs, historical = {}, {}
    for row in rows:
        name = row["sample_path"]
        inputs[name] = digest(resolve(name))
        known = checks.get("input_sha256", {}).get(name)
        historical[name] = "matched" if known is not None else "not_in_historical_protocol"
        if known is not None and known != inputs[name]:
            raise ValueError(f"Historical train input hash differs: {name}")
        if not np.isfinite(scales[row["group_id"]]) or scales[row["group_id"]] <= 0:
            raise ValueError("Invalid frozen reference scale")
        mapping = maps[row["channel_map_id"]]
        if mapping["drive_id"] != row["drive_id"] or len(mapping["source_channel_indices"]) != int(row["channels"]):
            raise ValueError("Channel map identity differs")
    for entry in weights.values():
        validate_checkpoint_identity(entry)
    # Snapshot all local inference dependencies, not just directly imported helpers.
    sources = sorted(set(ROOT.glob("finetuning/*.py")) | set(ROOT.glob("compression/*.py"))
                     | {ROOT / "tools/freeze_feasibility_metadata.py"})
    try:
        import vqvae
        vendor = Path(next(iter(vqvae.__path__))).resolve()
        if not (vendor / "train_vqvae.py").is_file() or not (vendor / "evaluate_vqvae.py").is_file():
            raise ImportError("Historical native builder sources are missing")
    except (ImportError, TypeError, AttributeError, StopIteration) as error:
        raise RuntimeError("Native vqvae builder package unavailable; install the historical vendor") from error
    sources += sorted(vendor.rglob("*.py"))
    source_hashes = {str(path): digest(path) for path in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    for path in sources:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("vendor/vqvae") / path.relative_to(vendor)
        destination = args.output / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    protocol = {"kind": "feasibility-context-diagnostic-v1", "weights": weights,
                "weights_manifest_sha256": digest(args.weights_manifest),
                "metadata_sha256": digest(args.metadata), "config_sha256": digest(config_path),
                "reference_scales_sha256": digest(scales_path), "channel_maps_sha256": digest(maps_path),
                "input_sha256": inputs, "historical_input_verification": historical,
                "source_sha256": source_hashes, "panel": metadata["f1_panel"],
                "expected_forward_passes": expected, "maximum_forward_passes": 480,
                "parameter_updates": 0, "reserved_signal_reads": 0, "device": args.device,
                "normalizations": list(NORMS), "codec_claim": False,
                "rate_note": "Nominal indices and diagnostic side only; not a complete transmitted codec",
                "zero_error_ratio_policy": "undefined; absolute difference retained"}
    write_json(args.output / "protocol.json", protocol)
    shutil.copy2(args.metadata, args.output / "metadata_freeze.json")
    shutil.copy2(args.weights_manifest, args.output / "weights_manifest.json")
    shutil.copy2(scales_path, args.output / "reference_scales.json")
    import torch
    from finetuning.joint_anchor_vq import exact_branch_forward
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.manual_seed(107)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, identities, count = [], [], 0
    with (args.output / "records.jsonl").open("x", encoding="utf-8") as log:
        for name in MODELS:
            model = build_checkpoint(weights[name], args.device)
            initial_hash = model_state_hash(model)
            for row in rows:
                path = resolve(row["sample_path"])
                if digest(path) != inputs[row["sample_path"]]:
                    raise ValueError("Train NPZ changed after freeze")
                with np.load(path, allow_pickle=False) as payload:
                    raw = payload["data"].copy()
                if raw.dtype != np.float32 or raw.shape != (int(row["channels"]), 1000):
                    raise ValueError("Frozen train tile dtype/shape differs")
                full = normalize(raw)
                plan = metadata["f1_panel"]["context_plans"][row["sample_path"]]
                mapping = maps[row["channel_map_id"]]["source_channel_indices"]
                core_channels = mapping[plan["core_start"]:plan["core_stop"]]
                replay = {}
                for window in plan["windows"]:
                    a, b = window["start"], window["stop"]
                    for norm in NORMS:
                        packet = normalize(raw[a:b]) if norm == NORMS[0] else slice_packet(full, a, b)
                        if count >= 480:
                            raise AssertionError("Forward budget exhausted")
                        tensor = torch.from_numpy(packet.values)[None, None].to(args.device)
                        with torch.inference_mode():
                            prediction, _, ids, latent, _ = exact_branch_forward(model, tensor)
                        count += 1
                        predicted = prediction[0, 0].cpu().numpy().copy()
                        indices = ids.cpu().numpy().astype("<u4")
                        restored = physical_inverse(predicted, packet)
                        if window["full_tile"]:
                            replay[norm] = (packet, indices.tobytes(), predicted.tobytes(), restored.tobytes())
                        metrics = score_core(raw, restored, plan, window, scales[row["group_id"]], core_channels)
                        bits = int(indices.size * math.ceil(math.log2(2048)))
                        record = {"model": name, "normalization": norm, **row,
                                  "context": window, "context_channels": b - a,
                                  "core_source_channels": core_channels, "metrics": metrics,
                                  "ledger": {"codec_claim": False, "input_shape": list(packet.values.shape),
                                             "latent_shape": list(latent), "index_shape": list(ids.shape),
                                             "padding_bottom": (a - b) % 16, "padding_right": (-1000) % 16,
                                             "nominal_index_bits": bits, "nominal_index_bytes": (bits + 7) // 8,
                                             "index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                                             "normalized_input_sha256": hashlib.sha256(packet.values.tobytes()).hexdigest(),
                                             "side_bytes": len(packet.side_stream()),
                                             "side_header_bytes": HEADER.size,
                                             "escape_bytes": 8 * len(packet.indices),
                                             "escape_count": len(packet.indices),
                                             "affine_mean": packet.mean, "affine_scale": packet.scale,
                                             "escape_threshold": packet.threshold,
                                             "full_affine_dependency_cells": int(raw.size) if norm == NORMS[1] else None,
                                             "context_cells": int(packet.values.size),
                                             "scored_cells": int((plan["core_stop"] - plan["core_start"]) * 1000),
                                             "frames_geometry_integrity_model_costs": "not measured; no complete-codec rate",
                                             "parameter_updates": 0}}
                        log.write(canonical(record).decode() + "\n")
                        log.flush()
                        records.append(record)
                first, second = replay[NORMS[0]], replay[NORMS[1]]
                identity = (first[0].side_stream() == second[0].side_stream()
                            and first[0].values.tobytes() == second[0].values.tobytes()
                            and first[0].valid.tobytes() == second[0].valid.tobytes()
                            and first[1:] == second[1:])
                identities.append({"model": name, "sample_path": row["sample_path"],
                                   "full_native_fixed_bit_identity": identity,
                                   "extra_forward_passes": 0})
                if not identity:
                    raise AssertionError("Full native/fixed replay is not bit-identical")
            if model_state_hash(model) != initial_hash:
                raise AssertionError("Frozen model state changed")
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    if count != expected:
        raise AssertionError("Executed forward count differs from frozen budget")
    for path, sha in source_hashes.items():
        if digest(path) != sha:
            raise ValueError("Source changed during inference")
    for entry in weights.values():
        if digest(resolve(entry["path"])) != entry["sha256"]:
            raise ValueError("Checkpoint changed during inference")
    for path, sha in inputs.items():
        if digest(resolve(path)) != sha:
            raise ValueError("Frozen train-panel input changed during inference")
    pairs, gates = context_pairs(records)
    write_json(args.output / "context_scores.json", {"pairs": pairs, "gates": gates})
    write_json(args.output / "results.json",
               {"forward_passes": count, "expected_forward_passes": expected,
                "parameter_updates": 0, "weights_unchanged": True, "reserved_signal_reads": 0,
                "record_count": len(records), "full_replay_identity": identities,
                "descriptive_context_signal": any(g["descriptive_gate_met"] for g in gates),
                "gates": gates, "codec_claim": False, "model_promoted": False,
                "independent": False, "status": "historical_seen_train_checks_not_independent"})
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
