"""CPU-only D2c on frozen, contiguous HDF5 regions; no remote access."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.feasibility_tables import MAX_CELLS, MODES, RESET
from finetuning.run_feasibility_npz_probe import (
    canonical, choose_support, digest, evaluate_window, fit_windows, local_path,
    quantized_window, write_json, zip_bytes,
)
from tools.freeze_feasibility_metadata import validate_manifest

RATIOS = (.0001, .0003, .001, .003, .01, .03, .1, .3, 1.)
SOURCES = (
    "finetuning/run_feasibility_continuous_probe.py",
    "tests/test_feasibility_continuous_probe.py",
    "finetuning/run_feasibility_npz_probe.py",
    "compression/feasibility_tables.py", "compression/feasibility_stream.py",
    "compression/feasibility_portable.py",
)


def build_plan(revalidated, scales, manifest_rows, strict=True):
    identities, _ = validate_manifest(manifest_rows)
    if (revalidated.get("kind") != "feasibility-centered-clock-revalidation-v1"
            or revalidated.get("forbidden_intersections") != 0
            or scales.get("version") != "physical-pooled-train-scale-v1"):
        raise ValueError("Invalid clock revalidation or frozen scales")
    rows, geometry, sigma, components, seen_files = [], {}, {}, {}, set()
    files = revalidated["files"]
    for entry in files:
        fid, group, role = entry["drive_id"], entry["group_id"], entry["candidate_role"]
        if (fid in seen_files or identities.get(fid) != ("train", group)
                or role not in ("fit", "check") or not entry["executable"]
                or "/" in fid or "\\" in fid or fid in (".", "..")):
            raise ValueError("Duplicate, unauthorized or non-train source")
        seen_files.add(fid)
        component = entry["component_id"]
        if component in components and components[component] != role:
            raise ValueError("Component crosses fit/check")
        components[component] = role
        scale = 100 * float(scales["scales"][group])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid frozen P0 scale")
        sigma[group] = scale
        occupied = set()
        for region in entry["regions"]:
            ids = region["source_channel_indices"]
            if (not ids or ids != list(range(ids[0], ids[-1] + 1))
                    or ids[0] < 0 or ids[-1] >= entry["source_shape"][2]
                    or occupied.intersection(ids)
                    or region["time_start_source_index"] != 0
                    or region["time_stop_source_index_exclusive"] != 5000
                    or region["cells"] != 5000 * len(ids)):
                raise ValueError("Invalid frozen region or overlapping coverage")
            occupied.update(ids)
            # Only oversized regions are computationally split. Each split starts
            # on exactly the same source reset boundary; no channel splitting.
            length = RESET if region["cells"] > MAX_CELLS else 5000
            if length * len(ids) > MAX_CELLS:
                raise ValueError("Single reset frame exceeds coder cap")
            for start in range(0, 5000, length):
                key = f"{fid}/r{region['region_index']}/t{start:05d}"
                rows.append({
                    **entry, "regions": [region], "role": role,
                    "sample_path": key, "raw_path": f"data/finetune/raw/{fid}.h5",
                    "region_index": region["region_index"], "source_time_start": start,
                    "time_samples": length, "channels": len(ids),
                })
                geometry[key] = ids
    cells = sum(r["channels"] * r["time_samples"] for r in rows)
    counts = dict(Counter(e["candidate_role"] for e in files))
    if (cells != revalidated["cells"] or cells > 64_000_000
            or any({r["role"] for r in rows if r["group_id"] == g} != {"fit", "check"}
                   for g in sigma)):
        raise ValueError("Census, budget or configuration contrast mismatch")
    if strict and (cells != 17_575_000 or counts != {"fit": 20, "check": 10}
                   or len(sigma) != 9):
        raise ValueError("Expected report-117 frozen census")
    return {
        "kind": "feasibility-continuous-plan-v1", "rows": rows, "geometry": geometry,
        "ratios": list(RATIOS), "primary_ratios": [.03, .1], "sigma_p0": sigma,
        "planned_original_cells": cells, "role_file_counts": counts,
        "waveform_cells_read": 0, "reset_samples": RESET,
        "deployment_platform": "Python 3 standard library installed; interpreter and OS excluded",
        "framing_contract": "One region stream unless >2M cells: separate 1000-sample streams; all framing and geometry charged; source resets unchanged.",
        "absent_configurations": ["multicore/C1 and C4"], "generalist_gate": None,
        "f4_authorized": False, "independence_claim": False,
        "limits": ["Centered common clocks are user-accepted assumptions, not vendor certification.",
                   "No session independence or ten-configuration/generalist pass.",
                   "P0 scales are inherited frozen calibration; no signal-driven selection."],
    }


def checked_dataset(handle, row):
    """Validate headers, geometry and selected clock before any waveform slice."""
    ds = handle[row["dataset"]]
    if ds.shape != tuple(row["source_shape"]) or ds.dtype != np.dtype("float32"):
        raise ValueError("HDF5 source dtype/shape mismatch")
    block = row["block_index"]
    start, stop = row["source_time_start"], row["source_time_start"] + row["time_samples"]
    region = row["regions"][0]
    ids = region["source_channel_indices"]
    if not (0 <= block < ds.shape[0] and 0 <= start < stop <= 5000 <= ds.shape[1]
            and 0 <= ids[0] <= ids[-1] < ds.shape[2]):
        raise ValueError("HDF5 slice outside frozen source bounds")
    zone = ds.parent
    spacing, origin, extent = (np.asarray(zone.attrs[k]) for k in ("Spacing", "Origin", "Extent"))
    if (spacing.shape not in ((2,), (3,)) or origin.shape != spacing.shape
            or extent.shape not in ((4,), (6,))
            or not np.isclose(spacing[1] / 1000, .00005, rtol=0, atol=1e-12)
            or extent[3] - extent[2] + 1 != ds.shape[1]
            or extent[1] - extent[0] + 1 != ds.shape[2]
            or not np.isclose((origin[1] + extent[2] * spacing[1]) / 1000,
                              -.525, rtol=0, atol=1e-12)
            or float(np.asarray(zone.attrs["BlockOverlap"]).reshape(-1)[0]) != 5
            or float(np.asarray(zone.attrs["BlockRate"]).reshape(-1)[0]) != 1000):
        raise ValueError("HDF5 acquisition metadata mismatch")
    if (not np.array_equal(np.asarray(ids) + extent[0], region["source_extent_channel_indices"])
            or not np.allclose(origin[0] + (np.asarray(ids) + extent[0]) * spacing[0],
                               region["metadata_positions_m"], rtol=0, atol=1e-9)
            or not np.allclose(np.asarray(ids) * spacing[0],
                               region["dataset_relative_positions_m"], rtol=0, atol=1e-9)):
        raise ValueError("HDF5 physical channel metadata mismatch")
    clocks = zone.parent["time"]
    tick = float(np.asarray(clocks[block]).reshape(-1)[0])
    if not np.isclose(tick - .525, row["raw_block_conditional_clock_interval_s"][0],
                      rtol=0, atol=5e-7):
        raise ValueError("HDF5 selected clock mismatch")
    return ds


def inspect_sources(root, plan):
    """Check all selected files before opening any waveform dataset for reading."""
    blockers, audited = [], []
    for fid in sorted({r["drive_id"] for r in plan["rows"]}):
        rows = [r for r in plan["rows"] if r["drive_id"] == fid]
        path = local_path(root, rows[0]["raw_path"])
        try:
            if not path.is_file():
                raise FileNotFoundError(str(path))
            if not h5py.is_hdf5(path):
                raise ValueError("Not an HDF5 file")
            with h5py.File(path, "r") as handle:
                for row in rows:
                    checked_dataset(handle, row)
            audited.append({"drive_id": fid, "path": rows[0]["raw_path"],
                            "file_bytes": path.stat().st_size, "headers_verified": True})
        except (OSError, ValueError, KeyError, IndexError) as exc:
            blockers.append({"drive_id": fid, "path": rows[0]["raw_path"],
                             "error_type": type(exc).__name__, "error": str(exc)})
    return {"files": audited, "blockers": blockers, "waveform_cells_read": 0}


def load_region(root, row, ids):
    path = local_path(root, row["raw_path"])
    with h5py.File(path, "r") as handle:
        ds = checked_dataset(handle, row)
        start = row["source_time_start"]
        raw = ds[row["block_index"], start:start + row["time_samples"], ids[0]:ids[-1] + 1]
    if raw.shape != (row["time_samples"], len(ids)):
        raise ValueError("Loaded HDF5 slice shape mismatch")
    return {**row, "raw": np.ascontiguousarray(raw), "channel_ids": ids}


def summarize_guards(configurations):
    """Report-107 thresholds by contract; missing C1/C4 never becomes a pass."""
    outcomes = []
    for ratio in (.03, .1):
        selected = [c for c in configurations if c["ratio"] == ratio]
        for mode in MODES[1:]:
            for contract in ("transport", "installed_plus_transport", "objects"):
                key = contract + "_bytes"
                baseline = sum(c["arms"]["marginal"][key] for c in selected)
                candidate = sum(c["arms"][mode][key] for c in selected)
                cells = sum(c["arms"][mode]["cells"] for c in selected)
                families = {
                    family: sum(c["arms"]["marginal"][key] - c["arms"][mode][key]
                                for c in selected if c["family"] == family)
                    for family in sorted({c["family"] for c in selected})}
                regressions = {
                    c["group_id"]: c["arms"][mode][key] / c["arms"]["marginal"][key] - 1
                    for c in selected}
                saving = (baseline - candidate) / baseline
                bpc = 8 * (baseline - candidate) / cells
                outcomes.append({
                    "ratio": ratio, "mode": mode, "contract": contract,
                    "marginal_bytes": baseline, "candidate_bytes": candidate,
                    "cells": cells, "saving_fraction": saving, "saving_bpc": bpc,
                    "family_saving_bytes": families,
                    "configuration_regression_fractions": regressions,
                    "measured_coverage_thresholds_pass": (
                        len(selected) == 9 and saving >= .1 and bpc >= .05
                        and len(families) == 4 and all(v > 0 for v in families.values())
                        and all(v <= .01 for v in regressions.values())),
                    "generalist_gate": None, "f4_authorized": False,
                })
    return outcomes


def execute_plan(plan, output, root=ROOT, budget_seconds=5400):
    started = time.perf_counter()
    records, configurations, read_cells = [], [], 0
    for gi, group in enumerate(sorted(plan["sigma_p0"])):
        windows = [load_region(root, r, plan["geometry"][r["sample_path"]])
                   for r in plan["rows"] if r["group_id"] == group]
        read_cells += sum(w["raw"].size for w in windows)
        for ratio in plan["ratios"]:
            if time.perf_counter() - started > budget_seconds:
                raise TimeoutError("90-minute CPU execution budget exhausted")
            epsilon = ratio * plan["sigma_p0"][group]
            quantized = [quantized_window(w, epsilon) for w in windows]
            fit = [w for w in quantized if w["role"] == "fit"]
            check = [w for w in quantized if w["role"] == "check"]
            before = time.perf_counter()
            k, candidates = choose_support(fit, epsilon)
            support_seconds = time.perf_counter() - before
            arms, paired = {}, {}
            for mode in MODES:
                before = time.perf_counter()
                model = fit_windows(fit, epsilon, k, mode)
                fit_seconds = time.perf_counter() - before
                transports, installs, objects, bindings, mode_records = {}, {}, {}, {}, []
                for wi, window in enumerate(check):
                    record, transport, installation, obj = evaluate_window(window, model, epsilon)
                    identity = (record["q_sha256"], record["waveform_sha256"],
                                record["original_float32_sha256"])
                    if wi in paired and paired[wi] != identity:
                        raise RuntimeError("Quantized/waveform reconstruction differs across arms")
                    paired[wi] = identity
                    name = f"{wi:03d}"
                    ih = digest(installation)
                    transports[name + ".fqs"] = transport
                    installs[ih + ".ftb"] = installation
                    objects[name + ".pyz"] = obj
                    bindings[name] = {key: window[key] for key in (
                        "drive_id", "dataset", "block_index", "region_index",
                        "source_time_start", "time_samples", "sample_path")}
                    bindings[name]["installation_sha256"] = ih
                    record.update({**bindings[name], "group_id": group,
                                   "family": group.split("/")[0], "mode": mode,
                                   "ratio": ratio, "epsilon": epsilon, "k": k})
                    mode_records.append(record)
                # The complete region bindings are transmitted, not free metadata.
                transports["bindings.json"] = canonical(bindings)
                installs["decoder.py"] = (root / "compression/feasibility_portable.py").read_bytes()
                installs["platform.json"] = canonical({"platform": plan["deployment_platform"]})
                objects["bindings.json"] = canonical(bindings)
                blobs = {"transport": zip_bytes(transports), "installation": zip_bytes(installs),
                         "objects": zip_bytes(objects)}
                prefix = f"group{gi:02d}_{ratio:g}_{mode}"
                for kind, blob in blobs.items():
                    (output / f"{prefix}.{kind}.zip").write_bytes(blob)
                cells = sum(r["cells"] for r in mode_records)
                arms[mode] = {
                    "cells": cells, "fit_seconds": fit_seconds,
                    "unique_geometry_installations": len(installs) - 2,
                    **{kind + "_bytes": len(blob) for kind, blob in blobs.items()},
                    **{kind + "_sha256": digest(blob) for kind, blob in blobs.items()},
                    "installed_plus_transport_bytes": len(blobs["transport"]) + len(blobs["installation"]),
                    **{key: sum(r[key] for r in mode_records) for key in (
                        "integer_cdf_nll_bits", "float_nll_bits", "sse",
                        "encode_seconds", "decode_seconds", "score_seconds", "bound_violations")},
                    "finite_cells": sum(r["finite_cells"] for r in mode_records),
                    "max_finite_error": max(r["max_finite_error"] for r in mode_records),
                }
                finite = arms[mode]["finite_cells"]
                arms[mode]["rmse"] = float(np.sqrt(arms[mode]["sse"] / finite)) if finite else None
                records.extend(mode_records)
            marginal = arms["marginal"]
            for arm in arms.values():
                for contract in ("transport", "installed_plus_transport", "objects"):
                    key = contract + "_bytes"
                    arm[contract + "_bpc"] = 8 * arm[key] / arm["cells"]
                    arm[contract + "_saving_bytes"] = marginal[key] - arm[key]
            config = {"group_id": group, "family": group.split("/")[0], "ratio": ratio,
                      "epsilon": epsilon, "k": k, "fit_support_candidates": candidates,
                      "support_selection_seconds": support_seconds, "arms": arms}
            configurations.append(config)
            write_json(output / f"group{gi:02d}_{ratio:g}.json", config)
            print(f"completed {group} ratio={ratio:g} elapsed={time.perf_counter()-started:.1f}s",
                  flush=True)
    result = {"kind": "feasibility-continuous-results-v1", "rows": records,
              "configurations": configurations, "original_cells_read": read_cells,
              "elapsed_seconds": time.perf_counter() - started,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "generalist_gate": None, "f4_authorized": False,
              "absent_configurations": plan["absent_configurations"], "limits": plan["limits"],
              "primary_guards": summarize_guards(configurations)}
    write_json(output / "results.json", result)
    return result


def run(plan, protocol, output, root=ROOT):
    if (protocol.get("kind") != "feasibility-continuous-execution-v1"
            or protocol.get("execute_continuous_probe") is not True
            or protocol.get("plan_sha256") != digest(canonical(plan))
            or protocol.get("f4_authorized") is not False):
        raise ValueError("Exact frozen-plan authorization required before HDF5 access")
    for name, sha in {**plan["input_sha256"], **plan["source_sha256"]}.items():
        if digest(local_path(root, name).read_bytes()) != sha:
            raise ValueError(f"Frozen file changed: {name}")
    from finetuning.run_feasibility_synthetic import verify_preflight
    verify_preflight(local_path(root, plan["coder_preflight"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "protocol.json", protocol)
    write_json(output / "freeze.json", {
        "plan_sha256": digest(canonical(plan)), "protocol_sha256": digest(canonical(protocol)),
        "waveform_cells_read": 0})
    audit = inspect_sources(root, plan)
    write_json(output / "source_audit.json", audit)
    if audit["blockers"]:
        write_json(output / "blocker.json", audit)
        return audit
    try:
        return execute_plan(plan, output, root)
    except Exception as exc:
        write_json(output / "blocker.json", {
            "error_type": type(exc).__name__, "error": str(exc), "selection_changed": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revalidated", default="results/experiments/feasibility_centered_clock_revalidation_20261118/results.json")
    parser.add_argument("--manifest", default="data/finetune/spatial_band_v1_fold0_20261024/samples_manifest.csv")
    parser.add_argument("--scales", default="results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json")
    parser.add_argument("--scientific-protocol", required=True)
    parser.add_argument("--preflight", default="results/experiments/feasibility_coder_preflight_20261117")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    names = (args.revalidated, args.manifest, args.scales, args.scientific_protocol,
             args.preflight + "/manifest.json")
    inputs = {name: local_path(ROOT, name).read_bytes() for name in names}
    plan = build_plan(json.loads(inputs[args.revalidated]), json.loads(inputs[args.scales]),
                      list(csv.DictReader(inputs[args.manifest].decode().splitlines())))
    plan["input_sha256"] = {name: digest(blob) for name, blob in inputs.items()}
    plan["source_sha256"] = {name: digest((ROOT / name).read_bytes()) for name in SOURCES}
    plan["coder_preflight"] = args.preflight
    if args.execute:
        if args.protocol is None:
            parser.error("--execute requires --protocol")
        run(plan, json.loads(args.protocol.read_text()), args.output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, plan)


if __name__ == "__main__":
    main()
