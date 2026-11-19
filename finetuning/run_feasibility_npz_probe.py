"""CPU-only, protocol-gated F2 screening of isolated train NPZ windows.

Planning opens metadata only. Execution requires a separately approved protocol.
This is not the continuous-raw experiment of report 107, a generalist gate, or F4.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.feasibility_tables import (
    MAX_CELLS, MAX_TABLE_BYTES, MODES, RESET, SUPPORTS, Tables, contexts,
    integer_frequencies, quantize, reconstruct, score, symbols,
)
from compression.feasibility_stream import decode, encode, portable_object
from compression.feasibility_portable import decode_bytes
from tools.freeze_feasibility_metadata import validate_manifest

RATIOS = (.03, .1)
FAMILIES = ("doped", "multimode", "single_mode")
SOURCES = (
    "finetuning/run_feasibility_npz_probe.py", "tests/test_feasibility_npz_probe.py",
    "compression/feasibility_tables.py", "compression/feasibility_stream.py",
    "compression/feasibility_portable.py",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def selection_key(path):
    return digest(("feasibility-npz-v1|" + path).encode())


def local_path(root, path):
    """No remote, raw, reserve, validation, or escaping symlink inputs."""
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Only repository-relative paths are permitted")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Input leaves the repository")
    return resolved


def select_windows(rows, envelope, strict=True):
    """Keep envelope roles; hash sample_path only, without inspecting NPZ files."""
    identities, samples = validate_manifest(rows)
    if envelope.get("kind") != "feasibility-clock-envelope-sensitivity-v1":
        raise ValueError("Expected report-110 envelope")
    roles = envelope["candidate_roles"]
    eligible = {g for g, c in envelope["contrasts"].items()
                if c["has_candidate_contrast"] and g.split("/")[0] in FAMILIES}
    by_file = defaultdict(list)
    for row in samples.values():
        if row["drive_id"] in roles:
            by_file[row["drive_id"]].append(row)
    selected = []
    for fid, role in sorted(roles.items()):
        if role not in ("fit", "check"):
            continue
        if fid not in identities:
            raise ValueError(f"Envelope file missing from manifest: {fid}")
        split, group = identities[fid]
        if group not in eligible:
            continue
        if split != "train":
            raise ValueError("Envelope role references non-train data")
        row = dict(min(by_file[fid], key=lambda r: (selection_key(r["sample_path"]),
                                                   r["sample_path"])))
        path = Path(row["sample_path"])
        if (path.is_absolute() or ".." in path.parts or path.suffix != ".npz"
                or path.parts[:3] != ("data", "finetune", "spatial_band_v1_fold0_20261024")
                or path.parts[3:4] != ("train",)
                or row["family"] != group.split("/")[0]
                or int(row["channels"]) * int(row["time_samples"]) > MAX_CELLS
                or int(row["channels"]) <= 0 or int(row["time_samples"]) <= 0):
            raise ValueError("Disallowed NPZ window or geometry")
        selected.append({**row, "role": role, "selection_sha256": selection_key(str(path))})
    for group in eligible:
        if {r["role"] for r in selected if r["group_id"] == group} != {"fit", "check"}:
            raise ValueError(f"Missing candidate contrast: {group}")
    # Components may span configurations; never split/reassign their roles.
    selected_roles = {r["drive_id"]: r["role"] for r in selected}
    for component in envelope["components"]:
        component_roles = {selected_roles[fid] for fid in component["members"]
                           if fid in selected_roles}
        if len(component_roles) > 1:
            raise ValueError("Envelope component crosses fit/check")
    families = {r["group_id"].split("/")[0] for r in selected}
    if strict and (len(eligible) != 8 or families != set(FAMILIES)):
        raise ValueError("Expected exactly eight configurations and three families")
    return selected


def build_plan(rows, envelope, scales, maps, strict=True):
    selected = select_windows(rows, envelope, strict)
    sigma = {}
    geometry = {}
    for row in selected:
        group = row["group_id"]
        if scales.get("version") != "physical-pooled-train-scale-v1":
            raise ValueError("Expected frozen P0 scales")
        scale = float(scales["scales"][group])
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(100 * scale):
            raise ValueError(f"Invalid P0 scale: {group}")
        sigma[group] = 100 * scale
        mapping = maps[row["channel_map_id"]]
        ids = mapping["source_channel_indices"]
        if (mapping["drive_id"] != row["drive_id"]
                or len(ids) != int(row["channels"])
                or any(type(i) is not int or i < 0 for i in ids)
                or any(b <= a for a, b in zip(ids, ids[1:]))):
            raise ValueError("Missing or inconsistent physical channel map")
        geometry[row["sample_path"]] = ids
    cells = sum(int(r["channels"]) * int(r["time_samples"]) for r in selected)
    census = {role: {"windows": sum(r["role"] == role for r in selected),
                     "cells": sum(int(r["channels"]) * int(r["time_samples"])
                                  for r in selected if r["role"] == role)}
              for role in ("fit", "check")}
    widths = {g: {role: sorted({int(r["channels"]) for r in selected
                               if r["group_id"] == g and r["role"] == role})
                  for role in ("fit", "check")} for g in sigma}
    if strict:
        expected = {"fit": {"windows": 18, "cells": 656000},
                    "check": {"windows": 8, "cells": 271000}}
        mismatches = {g: w for g, w in widths.items() if w["fit"] != w["check"]}
        if (census != expected or any(int(r["time_samples"]) != RESET for r in selected)
                or any(float(r["dt_s"]) != .00005 for r in selected)
                or mismatches != {"single_mode/Configuration 8": {"fit": [10], "check": [23]}}):
            raise ValueError("Report-113 census/widths differ; do not read or replace windows")
    if cells > 64_000_000:
        raise ValueError("Original-cell reading budget exceeds 64M")
    return {
        "kind": "feasibility-npz-plan-v1", "rows": selected, "geometry": geometry,
        "sigma_p0": sigma, "ratios": list(RATIOS), "reset_samples": RESET,
        "planned_original_cells": cells, "waveform_cells_read": 0,
        "census": census, "widths": widths,
        "absent_families": ["multicore"], "f4_authorized": False,
        "generalist_gate": None, "independence_claim": False,
        "scale_policy": "100 * frozen P0 s_g; no NPZ fit or check recalibration",
        "fit_policy": "separate configuration; accumulate per-window causal counts; never concatenate",
        "deployment_platform": "Python 3 standard library installed; interpreter and OS excluded",
        "limits": [
            "Isolated retained train NPZ windows only; not continuous raw F2/D2c.",
            "Report-110 clock assumptions remain uncertified.",
            "Inherited P0 calibration exposure is not new held-out calibration.",
            "No cross-session independence, absent-family inference, generalist gate, or F4 decision.",
        ],
    }


def zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 18, 0, 0, 0))
            archive.writestr(info, data)
    return buffer.getvalue()


def fit_windows(windows, epsilon, k, mode):
    """Accumulate counts with each window's own geometry and reset origin.

    Geometry is not learned. Returned probability model uses the first geometry
    as a serialization template; bind_geometry supplies and charges every actual
    geometry without changing the fitted probabilities.
    """
    nctx = {"marginal": 1, "temporal": 18, "spatiotemporal": 324}[mode]
    alphabet = 2 * k + 2
    if nctx * alphabet * 24 > MAX_TABLE_BYTES:
        raise ValueError("Fit table memory ceiling exceeded")
    counts = np.zeros((nctx, alphabet), dtype=np.int64)
    if not windows:
        raise ValueError("Empty fit windows")
    groups = {w["group_id"] for w in windows}
    if len(groups) != 1 or any(w["role"] != "fit" for w in windows):
        raise ValueError("Fit requires one configuration and fit roles only")
    for window in windows:
        q, positions = window["q"], window["positions"]
        c = contexts(q, positions, window["channel_ids"], mode).reshape(-1)
        s = symbols(q, k).reshape(-1)
        counts += np.bincount(c * alphabet + s,
                              minlength=nctx * alphabet).reshape(nctx, alphabet)
    total = counts.sum(axis=0)
    marginal = (total + .5) / (int(total.sum()) + .5 * alphabet)
    probabilities = [marginal]
    mapping = np.zeros(nctx, dtype=np.uint32)
    if mode != "marginal":
        for context, row in enumerate(counts):
            n = int(row.sum())
            if n >= 256:
                p = (row + .5) / (n + .5 * alphabet)
                probabilities.append(n / (n + 256) * p + 256 / (n + 256) * marginal)
                mapping[context] = len(probabilities) - 1
    p = np.asarray(probabilities)
    return Tables(k, mode, epsilon, windows[0]["channel_ids"],
                  np.asarray([integer_frequencies(row) for row in p]), mapping, p)


def bind_geometry(model, ids):
    # Reuse probabilities, never fit/reassign a file to solve geometry differences.
    return Tables(model.k, model.mode, model.epsilon, ids, model.frequencies,
                  model.mapping, model.probabilities)


def choose_support(windows, epsilon, supports=SUPPORTS):
    candidates = []
    for k in supports:
        model = fit_windows(windows, epsilon, k, "marginal")
        installations, transport_bytes = {}, 0
        for window in windows:
            tables = bind_geometry(model, window["channel_ids"])
            blob = tables.dump()
            installations[digest(blob)] = blob
            transport, _ = encode(window["q"], window["positions"], window["bits"], tables)
            transport_bytes += len(transport)
        installation_bytes = sum(map(len, installations.values()))
        candidates.append({"k": k, "fit_transport_bytes": transport_bytes,
                           "fit_installation_bytes": installation_bytes,
                           "fit_total_bytes": transport_bytes + installation_bytes})
    return min(candidates, key=lambda c: (c["fit_total_bytes"], c["k"]))["k"], candidates


def evaluate_window(window, model, epsilon):
    """Score/encode/decode frozen tables; both decoders receive bytes only."""
    tables = bind_geometry(model, window["channel_ids"])
    installation = tables.dump()
    before = time.perf_counter()
    nll = score(tables, window["q"], window["positions"])
    score_seconds = time.perf_counter() - before
    before = time.perf_counter()
    transport, costs = encode(window["q"], window["positions"], window["bits"], tables)
    encode_seconds = time.perf_counter() - before
    before = time.perf_counter()
    q, restored = decode(transport, installation)
    decode_seconds = time.perf_counter() - before
    expected = reconstruct(window["q"], window["positions"], window["bits"], epsilon)
    pq, px = decode_bytes(transport, installation)
    if (not np.array_equal(q, window["q"])
            or not np.array_equal(restored.view("u4"), expected.view("u4"))
            or pq != q.astype("<i4").tobytes() or px != expected.astype("<f4").tobytes()):
        raise RuntimeError("Causal q/waveform roundtrip failed")
    raw = window["raw"]
    finite = np.isfinite(raw)
    error = raw[finite].astype(np.float64) - restored[finite].astype(np.float64)
    violations = int(np.count_nonzero(np.abs(error) > epsilon))
    if violations:
        raise RuntimeError("Physical quantization bound violation")
    cells = raw.size
    obj = portable_object(transport, installation)
    return {
        **nll, **costs, "cells": cells, "finite_cells": int(finite.sum()),
        "nonfinite_cells": int((~finite).sum()),
        "negative_zero_cells": int(np.count_nonzero(raw.view("u4") == 0x80000000)),
        "sse": float(np.sum(error * error)),
        "rmse": float(np.sqrt(np.mean(error * error))) if error.size else None,
        "max_finite_error": float(np.max(np.abs(error), initial=0)),
        "bound_violations": violations, "q_roundtrip": True, "waveform_bit_exact": True,
        "portable_roundtrip": True, "transport_bpc": 8 * len(transport) / cells,
        "python_stdlib_object_bytes": len(obj),
        "object_bpc": 8 * len(obj) / cells,
        "cdf_rounding_delta_bits": nll["integer_cdf_nll_bits"] - nll["float_nll_bits"],
        "transport_minus_cdf_bits": 8 * len(transport) - nll["integer_cdf_nll_bits"],
        "installation_sha256": digest(installation), "transport_sha256": digest(transport),
        "q_sha256": digest(q.astype("<i4").tobytes()),
        "waveform_sha256": digest(expected.astype("<f4").tobytes()),
        "original_float32_sha256": digest(raw.astype("<f4").tobytes()),
        "score_seconds": score_seconds, "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
    }, transport, installation, obj


def load_window(root, row, ids):
    path = local_path(root, row["sample_path"])
    with np.load(path, allow_pickle=False) as data:
        raw = data["data"]
    shape = (int(row["channels"]), int(row["time_samples"]))
    if raw.dtype != np.float32 or raw.shape != shape:
        raise ValueError(f"NPZ dtype/shape mismatch: {row['sample_path']}")
    return {**row, "raw": np.ascontiguousarray(raw.T), "channel_ids": ids}


def quantized_window(window, epsilon):
    q, positions, bits = quantize(window["raw"], epsilon)
    return {**window, "q": q, "positions": positions, "bits": bits}


def validate_authorization(plan, protocol):
    if (protocol.get("kind") != "feasibility-npz-execution-v1"
            or protocol.get("execute_npz_train_probe") is not True
            or protocol.get("plan_sha256") != digest(canonical(plan))
            or protocol.get("f4_authorized", False) is not False):
        raise ValueError("Execution requires a separately approved, exact-plan protocol")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                allow_nan=False) + "\n")


def run(plan, protocol, output, root=ROOT, loader=load_window):
    validate_authorization(plan, protocol)  # Must precede all NPZ reads.
    for name, expected in plan.get("input_sha256", {}).items():
        if digest(local_path(root, name).read_bytes()) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for name, expected in plan.get("source_sha256", {}).items():
        if digest(local_path(root, name).read_bytes()) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if "coder_preflight" in plan:
        from finetuning.run_feasibility_synthetic import verify_preflight
        verify_preflight(local_path(root, plan["coder_preflight"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "protocol.json", protocol)
    write_json(output / "freeze.json", {
        "plan_sha256": digest(canonical(plan)), "protocol_sha256": digest(canonical(protocol)),
        "waveform_cells_read": 0,
    })
    try:
        return execute_plan(plan, output, root, loader)
    except Exception as exc:
        write_json(output / "blocker.json", {"error_type": type(exc).__name__,
                                           "error": str(exc), "selection_changed": False})
        raise


def execute_plan(plan, output, root, loader):
    started = time.perf_counter()
    results, summaries = [], []
    read_cells = 0
    groups = sorted({r["group_id"] for r in plan["rows"]})
    for group_index, group in enumerate(groups):
        rows = [r for r in plan["rows"] if r["group_id"] == group]
        # Each selected NPZ read once. Ratios/arms reuse identical physical cells.
        windows = [loader(root, row, plan["geometry"][row["sample_path"]]) for row in rows]
        read_cells += sum(w["raw"].size for w in windows)
        for ratio in RATIOS:
            epsilon = ratio * plan["sigma_p0"][group]
            quantized = [quantized_window(w, epsilon) for w in windows]
            fit = [w for w in quantized if w["role"] == "fit"]
            check = [w for w in quantized if w["role"] == "check"]
            k, candidates = choose_support(fit, epsilon)
            arm_summaries = {}
            for mode in MODES:
                before = time.perf_counter()
                model = fit_windows(fit, epsilon, k, mode)
                fit_seconds = time.perf_counter() - before
                transports, installations, objects = {}, {}, {}
                bindings, records = {}, []
                for index, window in enumerate(check):
                    record, transport, installation, obj = evaluate_window(window, model, epsilon)
                    name = f"{index:03d}"
                    table_hash = digest(installation)
                    transports[f"{name}.fqs"] = transport
                    installations[f"{table_hash}.ftb"] = installation
                    objects[f"{name}.pyz"] = obj
                    bindings[name] = {"sample_path": window["sample_path"],
                                      "installation_sha256": table_hash}
                    record.update({"group_id": group, "family": group.split("/")[0],
                                   "drive_id": window["drive_id"], "sample_path": window["sample_path"],
                                   "ratio": ratio, "epsilon": epsilon, "k": k, "mode": mode})
                    records.append(record)
                # Persist actual complete installed transport and installation.
                transports["bindings.json"] = canonical(bindings)
                installations["decoder.py"] = (root / "compression/feasibility_portable.py").read_bytes()
                installations["platform.json"] = canonical({"platform": plan["deployment_platform"]})
                transport_zip = zip_bytes(transports)
                installation_zip = zip_bytes(installations)
                object_zip = zip_bytes(objects)  # Actual independent executable objects; repetitions charged.
                prefix = f"group{group_index:02d}_{ratio:g}_{mode}"
                for suffix, blob in (("transport.zip", transport_zip),
                                     ("installation.zip", installation_zip), ("objects.zip", object_zip)):
                    with (output / f"{prefix}.{suffix}").open("xb") as stream:
                        stream.write(blob)
                cells = sum(r["cells"] for r in records)
                finite = sum(r["finite_cells"] for r in records)
                summary = {
                    "transport_bytes": len(transport_zip), "installation_bytes": len(installation_zip),
                    "installed_plus_transport_bytes": len(transport_zip) + len(installation_zip),
                    "self_contained_objects_bytes": len(object_zip), "cells": cells,
                    "transport_bpc": 8 * len(transport_zip) / cells,
                    "installed_plus_transport_bpc": 8 * (len(transport_zip) + len(installation_zip)) / cells,
                    "self_contained_bpc": 8 * len(object_zip) / cells,
                    "integer_cdf_nll_bits": sum(r["integer_cdf_nll_bits"] for r in records),
                    "float_nll_bits": sum(r["float_nll_bits"] for r in records),
                    "rmse": float(np.sqrt(sum(r["sse"] for r in records) / finite)) if finite else None,
                    "fit_seconds": fit_seconds, "unique_geometry_installations": len(installations) - 2,
                    "fit_file_ids": [w["drive_id"] for w in fit],
                    "installation_sha256": digest(installation_zip),
                    "transport_sha256": digest(transport_zip), "objects_sha256": digest(object_zip),
                }
                arm_summaries[mode] = summary
                results.extend(records)
            marginal = arm_summaries["marginal"]
            for mode, summary in arm_summaries.items():
                summary["cdf_saving_bpc_vs_marginal"] = (
                    marginal["integer_cdf_nll_bits"] - summary["integer_cdf_nll_bits"]) / summary["cells"]
                summary["transport_saving_bytes_vs_marginal"] = marginal["transport_bytes"] - summary["transport_bytes"]
                summary["object_saving_bytes_vs_marginal"] = (
                    marginal["self_contained_objects_bytes"] - summary["self_contained_objects_bytes"])
            summaries.append({"group_id": group, "family": group.split("/")[0], "ratio": ratio,
                              "sigma_p0": plan["sigma_p0"][group], "epsilon": epsilon,
                              "k": k, "fit_support_candidates": candidates, "arms": arm_summaries})
            write_json(output / f"group{group_index:02d}_{ratio:g}.json", summaries[-1])
    result = {"kind": "feasibility-npz-screening-v1", "rows": results,
              "configurations": summaries, "original_cells_read": read_cells,
              "elapsed_seconds": time.perf_counter() - started,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "absent_families": plan["absent_families"], "limits": plan["limits"],
              "generalist_gate": None, "f4_authorized": False}
    write_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/finetune/spatial_band_v1_fold0_20261024/samples_manifest.csv")
    parser.add_argument("--envelope", default="results/experiments/feasibility_clock_envelopes_20261118/results.json")
    parser.add_argument("--scales", default="results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json")
    parser.add_argument("--maps", default="data/finetune/spatial_band_v1_fold0_20261024/channel_maps.jsonl")
    parser.add_argument("--scientific-protocol", default="progress_reports/archive/feasibility_protocols_20261118/113-FEBUS_SHORT_WINDOW_ENTROPY_TRIAGE.md")
    parser.add_argument("--preflight", default="results/experiments/feasibility_coder_preflight_20261117")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    inputs = {name: local_path(ROOT, name).read_bytes()
              for name in (args.manifest, args.envelope, args.scales, args.maps,
                           args.scientific_protocol, args.preflight + "/manifest.json")}
    rows = list(csv.DictReader(io.StringIO(inputs[args.manifest].decode())))
    maps = {r["channel_map_id"]: r for r in
            map(json.loads, inputs[args.maps].decode().splitlines())}
    plan = build_plan(rows, json.loads(inputs[args.envelope]), json.loads(inputs[args.scales]), maps)
    plan["input_sha256"] = {name: digest(data) for name, data in inputs.items()}
    plan["source_sha256"] = {name: digest((ROOT / name).read_bytes()) for name in SOURCES}
    plan["coder_preflight"] = args.preflight
    if args.execute:
        if args.protocol is None:
            parser.error("--execute requires a separately approved --protocol")
        run(plan, json.loads(args.protocol.read_text()), args.output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, plan)


if __name__ == "__main__":
    main()
