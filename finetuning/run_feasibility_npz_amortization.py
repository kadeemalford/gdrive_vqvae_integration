"""CPU-only amortization of report-114 installations; no fitting or NPZ planning reads."""
import argparse
from collections import Counter, defaultdict
import csv
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

from compression.feasibility_tables import MODES, RESET, Tables, reconstruct, score
from compression.feasibility_stream import decode, encode
from compression.feasibility_portable import decode_bytes
from finetuning.run_feasibility_npz_probe import (
    canonical, digest, load_window, local_path, quantized_window, write_json, zip_bytes,
)

BASE = "results/experiments/feasibility_npz_probe_20261118"
OUTPUT = "results/experiments/feasibility_npz_amortization_20261118"
MANIFEST = "data/finetune/spatial_band_v1_fold0_20261024/samples_manifest.csv"
MAPS = "data/finetune/spatial_band_v1_fold0_20261024/channel_maps.jsonl"
PROTOCOL = "progress_reports/archive/feasibility_protocols_20261118/115-FEBUS_FROZEN_INSTALLATION_AMORTIZATION.md"
SOURCES = (
    "finetuning/run_feasibility_npz_amortization.py",
    "tests/test_feasibility_npz_amortization.py",
    "finetuning/run_feasibility_npz_probe.py",
    "compression/feasibility_tables.py", "compression/feasibility_stream.py",
    "compression/feasibility_portable.py",
)


def selection_key(path):
    return digest(("feasibility-amortization-v1|" + path).encode())


def frozen_dependency(root, relative, expected):
    """Resolve historical sources by content, including archived protocol originals."""
    archive = Path("progress_reports/archive/feasibility_protocols_20261118")
    for name in (Path(relative), archive / Path(relative).name):
        path = local_path(root, str(name))
        if path.is_file():
            content = path.read_bytes()
            if digest(content) == expected:
                return str(name), content
    raise ValueError(f"Frozen dependency changed: {relative}")


def interval(row):
    start = int(row["time_start"])
    if start < 0:
        raise ValueError("Negative time_start")
    return start, start + int(row["time_samples"])


def overlaps(a, b):
    return (a["block_index"] == b["block_index"]
            and max(interval(a)[0], interval(b)[0]) < min(interval(a)[1], interval(b)[1]))


def physical_ids(row, maps):
    mapping = maps[row["channel_map_id"]]
    ids = mapping["source_channel_indices"]
    if (mapping["drive_id"] != row["drive_id"] or len(ids) != int(row["channels"])
            or any(type(i) is not int or i < 0 for i in ids)
            or any(b <= a for a, b in zip(ids, ids[1:]))):
        raise ValueError("Invalid physical geometry")
    return ids


def select_windows(rows, original_rows, maps):
    selected, coverage = [], {}
    if len({r["sample_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate manifest paths")
    for original in sorted(original_rows, key=lambda r: r["group_id"]):
        ids = physical_ids(original, maps)
        accepted, excluded = [], Counter()
        candidates = sorted((r for r in rows if r["drive_id"] == original["drive_id"]),
                            key=lambda r: (selection_key(r["sample_path"]), r["sample_path"]))
        for row in candidates:
            if row["sample_path"] == original["sample_path"]:
                excluded["original"] += 1
                continue
            path = Path(row["sample_path"])
            if (row["split"] != "train" or path.parts[:4] != (
                    "data", "finetune", "spatial_band_v1_fold0_20261024", "train")
                    or ".." in path.parts or path.suffix != ".npz"):
                raise ValueError("Non-train or unsafe candidate")
            if (row["group_id"] != original["group_id"]
                    or row["family"] != original["family"]):
                raise ValueError("File identity changed")
            if (row["dataset"] != original["dataset"]
                    or float(row["dt_s"]) != float(original["dt_s"])
                    or int(row["channels"]) != int(original["channels"])
                    or int(row["time_samples"]) != RESET
                    or physical_ids(row, maps) != ids):
                excluded["geometry_dataset_dt"] += 1
                continue
            if any(overlaps(row, other) for other in [original, *accepted]):
                excluded["same_block_overlap"] += 1
                continue
            if len(accepted) == 20:
                excluded["cap"] += 1
                continue
            accepted.append({**row, "role": "check",
                             "selection_sha256": selection_key(row["sample_path"])})
        selected.extend(accepted)
        coverage[original["group_id"]] = {
            "drive_id": original["drive_id"], "selected": len(accepted),
            "excluded": dict(excluded),
            "blocks": sorted({r["block_index"] for r in accepted}),
            "original_sample_path": original["sample_path"],
        }
    return selected, coverage


def read_installation(blob, expected_sha, mode, epsilon, ids, decoder):
    if digest(blob) != expected_sha:
        raise ValueError("Report-114 installation ZIP SHA mismatch")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        tables = [n for n in names if n.endswith(".ftb")]
        if (len(names) != len(set(names)) or len(tables) != 1
                or set(names) != {tables[0], "decoder.py", "platform.json"}
                or archive.testzip() is not None):
            raise ValueError("Unexpected installation ZIP schema")
        table_blob = archive.read(tables[0])
        if tables[0] != digest(table_blob) + ".ftb" or archive.read("decoder.py") != decoder:
            raise ValueError("Installation table/decoder identity mismatch")
        platform = json.loads(archive.read("platform.json"))
        if set(platform) != {"platform"} or not isinstance(platform["platform"], str):
            raise ValueError("Unexpected platform schema")
    table = Tables.load(table_blob)
    if (table.mode != mode or table.epsilon != epsilon or table.channel_ids != ids
            or table.dump() != table_blob):
        raise ValueError("Exact frozen geometry/probability reuse impossible")
    return table, table_blob


def build_plan(root=ROOT):
    paths = [MANIFEST, MAPS, PROTOCOL, BASE + "/plan.json", BASE + "/results.json",
             "results/experiments/feasibility_clock_envelopes_20261118/results.json", *SOURCES]
    inputs = {p: local_path(root, p).read_bytes() for p in paths}
    original = json.loads(inputs[BASE + "/plan.json"])
    previous = json.loads(inputs[BASE + "/results.json"])
    if original["kind"] != "feasibility-npz-plan-v1" or previous["kind"] != "feasibility-npz-screening-v1":
        raise ValueError("Unexpected report-114 schema")
    checks = [r for r in original["rows"] if r["role"] == "check"]
    if (len(checks) != 8 or len({r["drive_id"] for r in checks}) != 8
            or original["ratios"] != [.03, .1]
            or any(int(r["time_samples"]) != RESET for r in checks)):
        raise ValueError("Report-114 check census mismatch")
    maps = {r["channel_map_id"]: r for r in
            map(json.loads, inputs[MAPS].decode().splitlines())}
    rows, coverage = select_windows(
        list(csv.DictReader(io.StringIO(inputs[MANIFEST].decode()))), checks, maps)
    installations = {}
    groups = sorted({r["group_id"] for r in checks})
    for summary in previous["configurations"]:
        group, ratio = summary["group_id"], summary["ratio"]
        if group not in groups or ratio not in original["ratios"]:
            raise ValueError("Unexpected report-114 configuration")
        ids = physical_ids(next(r for r in checks if r["group_id"] == group), maps)
        for mode in MODES:
            path = f"{BASE}/group{groups.index(group):02d}_{ratio:g}_{mode}.installation.zip"
            blob = local_path(root, path).read_bytes()
            expected = summary["arms"][mode]["installation_sha256"]
            model, table_blob = read_installation(
                blob, expected, mode, summary["epsilon"], ids,
                inputs["compression/feasibility_portable.py"])
            installations[f"{group}|{ratio:g}|{mode}"] = {
                "path": path, "sha256": expected, "bytes": len(blob),
                "table_sha256": digest(table_blob), "epsilon": model.epsilon,
                "k": model.k, "channel_ids": ids,
            }
            inputs[path] = blob
    if len(installations) != 48:
        raise ValueError("Expected exactly 48 frozen installations")
    return {
        "kind": "feasibility-npz-amortization-plan-v1", "rows": rows,
        "coverage": coverage, "installations": installations, "ratios": [.03, .1],
        "geometry": {r["sample_path"]: physical_ids(r, maps) for r in rows},
        "input_sha256": {p: digest(b) for p, b in inputs.items()},
        "planned_npz_cell_occurrences": sum(int(r["channels"]) * RESET for r in rows),
        "waveform_cells_read": 0, "f4_authorized": False,
        "deployment": "Exact report-114 ZIP once per group/ratio/mode; cumulative ZIP_STORED frames plus bindings",
        "adaptation": "none; fail closed unless exact original physical geometry and table bytes",
        "temporal_scope": "Same-block intervals do not overlap; cross-block clock semantics uncertified. "
                          "Byte-reuse inference only; counts are NPZ cell occurrences, NOT unique physical cells.",
        "float_probability_policy": "Unavailable in serialized installation; integer CDF only, no reconstruction/refit",
        "checkpoints": [1, 5, 10, 20], "reset_samples": RESET,
    }


def evaluate(window, model, installation):
    transport, costs = encode(window["q"], window["positions"], window["bits"], model)
    q, restored = decode(transport, installation)
    pq, px = decode_bytes(transport, installation)
    expected = reconstruct(window["q"], window["positions"], window["bits"], model.epsilon)
    if (not np.array_equal(q, window["q"]) or restored.tobytes() != expected.tobytes()
            or pq != q.astype("<i4").tobytes() or px != expected.astype("<f4").tobytes()):
        raise RuntimeError("q/waveform/portable byte roundtrip failed")
    raw = window["raw"]
    finite = np.isfinite(raw)
    error = raw[finite].astype(np.float64) - restored[finite].astype(np.float64)
    violations = int(np.count_nonzero(np.abs(error) > model.epsilon))
    if violations:
        raise RuntimeError("Quantization bound violation")
    return {
        **score(model, q, window["positions"]), **costs,
        "cells": raw.size, "finite_cells": int(finite.sum()),
        "sse": float(np.sum(error * error)), "bound_violations": violations,
        "max_finite_error": float(np.max(np.abs(error), initial=0)),
        "q_roundtrip": True, "waveform_bit_exact": True, "portable_roundtrip": True,
        "q_sha256": digest(q.astype("<i4").tobytes()),
        "waveform_sha256": digest(expected.astype("<f4").tobytes()),
        "original_float32_sha256": digest(raw.astype("<f4").tobytes()),
        "transport_sha256": digest(transport),
    }, transport


def add_contrasts(curves, scope):
    lookup = {(c.get("group_id"), c["ratio"], c["mode"], c[scope]): c for c in curves}
    thresholds = {}
    for curve in curves:
        marginal = lookup[(curve.get("group_id"), curve["ratio"], "marginal", curve[scope])]
        for key in ("transport_bytes", "total_bytes", "integer_cdf_nll_bits"):
            curve[key + "_saving_vs_marginal"] = marginal[key] - curve[key]
        curve["incremental_installation_bytes"] = (
            curve["installation_bytes"] - marginal["installation_bytes"])
        key = f"{curve.get('group_id', 'aggregate')}|{curve['ratio']:g}|{curve['mode']}"
        threshold = thresholds.setdefault(key, {
            "first_nonnegative_horizon": None, "observed_horizons_with_net_saving": []})
        if curve["total_bytes_saving_vs_marginal"] >= 0:
            if threshold["first_nonnegative_horizon"] is None:
                threshold["first_nonnegative_horizon"] = curve[scope]
            threshold["observed_horizons_with_net_saving"].append(curve[scope])
    return thresholds


def run(plan, output, root=ROOT, loader=load_window):
    # Verify every frozen metadata/source/installation byte before any NPZ value.
    for path, sha in plan["input_sha256"].items():
        if digest(local_path(root, path).read_bytes()) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "freeze.json", {
        "plan_sha256": digest(canonical(plan)), "waveform_cells_read": 0,
        "authorization": PROTOCOL, "execution": "CPU; no fit; no HDF5/val/reserves/GPU",
    })
    try:
        return execute_plan(plan, output, root, loader)
    except Exception as exc:
        write_json(output / "blocker.json", {
            "error_type": type(exc).__name__, "error": str(exc), "selection_changed": False})
        raise


def execute_plan(plan, output, root, loader):
    started = time.perf_counter()
    records, curves = [], []
    read_cells = 0
    for gi, group in enumerate(sorted(plan["coverage"])):
        rows = [r for r in plan["rows"] if r["group_id"] == group]
        windows = [loader(root, r, plan["geometry"][r["sample_path"]]) for r in rows]
        read_cells += sum(w["raw"].size for w in windows)
        for ratio in plan["ratios"]:
            signatures = {}
            for mode in MODES:
                spec = plan["installations"][f"{group}|{ratio:g}|{mode}"]
                blob = local_path(root, spec["path"]).read_bytes()
                model, installation = read_installation(
                    blob, spec["sha256"], mode, spec["epsilon"], spec["channel_ids"],
                    (root / "compression/feasibility_portable.py").read_bytes())
                # Persist the exact artifact actually charged and used.
                prefix = f"group{gi:02d}_{ratio:g}_{mode}"
                (output / f"{prefix}.installation.zip").write_bytes(blob)
                entries, bindings, arm_records = {}, {}, []
                for index, w in enumerate(windows, 1):
                    before = time.perf_counter()
                    qw = quantized_window(w, model.epsilon)
                    record, transport = evaluate(qw, model, installation)
                    signature = (record["q_sha256"], record["waveform_sha256"])
                    if index in signatures and signatures[index] != signature:
                        raise RuntimeError("Cross-mode roundtrip disagreement")
                    signatures[index] = signature
                    record.update({
                        "group_id": group, "ratio": ratio, "mode": mode, "window_index": index,
                        "sample_path": w["sample_path"], "drive_id": w["drive_id"],
                        "epsilon": model.epsilon, "k": model.k,
                        "elapsed_seconds": time.perf_counter() - before})
                    records.append(record)
                    arm_records.append(record)
                    name = f"{index:03d}.fqs"
                    entries[name] = transport
                    bindings[name] = {"sample_path": w["sample_path"],
                                      "installation_sha256": spec["table_sha256"]}
                    payload = zip_bytes({**entries, "bindings.json": canonical(bindings)})
                    cells = sum(r["cells"] for r in arm_records)
                    curve = {
                        "group_id": group, "ratio": ratio, "mode": mode, "windows": index,
                        "cells": cells, "frame_bytes": sum(len(b) for b in entries.values()),
                        "transport_bytes": len(payload), "installation_bytes": len(blob),
                        "total_bytes": len(payload) + len(blob),
                        "transport_sha256": digest(payload),
                        "integer_cdf_nll_bits": sum(r["integer_cdf_nll_bits"] for r in arm_records),
                        "integer_cdf_bpc": sum(r["integer_cdf_nll_bits"] for r in arm_records) / cells,
                        "transport_bpc": 8 * len(payload) / cells,
                        "total_bpc": 8 * (len(payload) + len(blob)) / cells,
                    }
                    curves.append(curve)
                    if index in plan["checkpoints"] or index == len(windows):
                        (output / f"{prefix}.n{index:02d}.transport.zip").write_bytes(payload)
                write_json(output / f"{prefix}.json", {
                    "installation": spec, "rows": arm_records,
                    "curves": [c for c in curves if c["group_id"] == group
                               and c["ratio"] == ratio and c["mode"] == mode]})
    aggregates = []
    for ratio in plan["ratios"]:
        for horizon in range(1, 21):
            for mode in MODES:
                selected = [c for c in curves if c["ratio"] == ratio and c["mode"] == mode
                            and c["windows"] == min(horizon, plan["coverage"][c["group_id"]]["selected"])]
                aggregate = {"ratio": ratio, "mode": mode, "horizon": horizon,
                             "groups": len(selected), "available_windows_by_group":
                             {c["group_id"]: c["windows"] for c in selected}}
                for key in ("windows", "cells", "frame_bytes", "transport_bytes",
                            "installation_bytes", "total_bytes", "integer_cdf_nll_bits"):
                    aggregate[key] = sum(c[key] for c in selected)
                if aggregate["cells"]:
                    aggregate["total_bpc"] = 8 * aggregate["total_bytes"] / aggregate["cells"]
                    aggregate["transport_bpc"] = 8 * aggregate["transport_bytes"] / aggregate["cells"]
                    aggregate["integer_cdf_bpc"] = aggregate["integer_cdf_nll_bits"] / aggregate["cells"]
                aggregates.append(aggregate)
    thresholds = {
        "by_group": add_contrasts(curves, "windows"),
        "aggregate": add_contrasts(aggregates, "horizon"),
        "policy": "Observed cumulative prefixes only; first nonnegative is not extrapolation or sustained guarantee",
    }
    result = {
        "kind": "feasibility-npz-amortization-results-v1", "rows": records,
        "curves": curves, "aggregates": aggregates, "coverage": plan["coverage"],
        "npz_cell_occurrences_read": read_cells,
        "temporal_scope": plan["temporal_scope"], "float_probability_policy": plan["float_probability_policy"],
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "f4_authorized": False, "fit_calls": 0, "observed_amortization": thresholds,
    }
    write_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    plan = build_plan()
    if args.execute:
        run(plan, ROOT / args.output)
    else:
        print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
