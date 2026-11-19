"""Conditional CPU transfer of exact report-114 installations, importing frozen 115."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuning import run_feasibility_npz_amortization as amort

PROTOCOL = "progress_reports/116-FEBUS_FROZEN_INSTALLATION_TRANSFER_PROTOCOL.md"
OUTPUT = "results/experiments/feasibility_npz_transfer_20261118"
ENVELOPE = "results/experiments/feasibility_clock_envelopes_20261118/results.json"
SOURCES = ("finetuning/run_feasibility_npz_transfer.py", "tests/test_feasibility_npz_transfer.py")


def selection_key(path):
    return amort.digest(("feasibility-transfer-v1|" + path).encode())


def gate115(result):
    if result.get("kind") != "feasibility-npz-amortization-results-v1":
        raise ValueError("Unexpected report-115 result schema")
    decisions = {}
    for ratio in (.03, .1):
        arms = {a["mode"]: a for a in result["aggregates"]
                if a["ratio"] == ratio and a["horizon"] == 20}
        if set(arms) != set(amort.MODES):
            raise ValueError("Incomplete report-115 gate")
        marginal, contextual = arms["marginal"], arms["spatiotemporal"]
        if (marginal["groups"] != 8 or contextual["groups"] != 8
                or marginal["cells"] != contextual["cells"]
                or marginal["windows"] != contextual["windows"]):
            raise ValueError("Report-115 populations differ")
        saving = marginal["total_bytes"] - contextual["total_bytes"]
        if saving <= 0:
            raise ValueError(f"Report-115 gate failed at {ratio:g}; no transfer authorized")
        decisions[str(ratio)] = {"net_saving_bytes": saving, "strictly_positive": True}
    return decisions


def select_windows(rows, originals, maps, envelope, fit_ids, strict=True):
    if envelope["kind"] != "feasibility-clock-envelope-sensitivity-v1":
        raise ValueError("Unexpected clock-envelope schema")
    # Conservatively exclude any component touching any fit-role file.
    fits = set(fit_ids) | {fid for fid, role in envelope["candidate_roles"].items()
                           if role == "fit"}
    candidate_ids = {fid for component in envelope["components"]
                     if not set(component["members"]) & fits for fid in component["check_ids"]}
    by_file = defaultdict(list)
    if len({r["sample_path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate manifest paths")
    for row in rows:
        if row["drive_id"] in candidate_ids and row["group_id"] in originals:
            if row["split"] != "train":
                raise ValueError("Historical check candidate outside train")
            by_file[row["drive_id"]].append(row)
    selected, files = [], {}
    previous_ids = {r["drive_id"] for r in originals.values()} | fits
    for fid, candidates in sorted(by_file.items()):
        if fid in previous_ids:
            raise ValueError("Transfer candidate already used by report-114 coder")
        group = candidates[0]["group_id"]
        ref = originals[group]
        ids = amort.physical_ids(ref, maps)
        accepted, incompatible, overlap = [], 0, 0
        for row in sorted(candidates, key=lambda r: (selection_key(r["sample_path"]), r["sample_path"])):
            path = Path(row["sample_path"])
            if (path.parts[:4] != ("data", "finetune", "spatial_band_v1_fold0_20261024", "train")
                    or path.suffix != ".npz" or ".." in path.parts):
                raise ValueError("Unsafe transfer NPZ path")
            if row["group_id"] != group or row["family"] != ref["family"]:
                raise ValueError("Inconsistent transfer file identity")
            if (row["dataset"] != ref["dataset"] or float(row["dt_s"]) != float(ref["dt_s"])
                    or int(row["time_samples"]) != amort.RESET
                    or int(row["channels"]) != int(ref["channels"])
                    or amort.physical_ids(row, maps) != ids):
                incompatible += 1
                continue
            if any(amort.overlaps(row, other) for other in accepted):
                overlap += 1
                continue
            if len(accepted) < 20:
                accepted.append({**row, "role": "check",
                                 "selection_sha256": selection_key(row["sample_path"])})
        selected.extend(accepted)
        files[fid] = {"group_id": group, "selected": len(accepted),
                      "incompatible_windows": incompatible, "overlapping_windows": overlap,
                      "candidate_windows": len(candidates),
                      "compatible": bool(accepted),
                      "blocks": sorted({r["block_index"] for r in accepted})}
    groups = {g: {"candidate_files": sum(f["group_id"] == g for f in files.values()),
                  "compatible_files": sum(f["group_id"] == g and f["compatible"] for f in files.values()),
                  "windows": sum(f["selected"] for f in files.values() if f["group_id"] == g)}
              for g in originals}
    if strict and (len(files) != 25 or sum(f["compatible"] for f in files.values()) != 19
                   or len(groups) != 8 or any(not g["compatible_files"] for g in groups.values())):
        raise ValueError("Report-116 prospective census differs; do not substitute files")
    return selected, files, groups


def build_plan(root=ROOT):
    prior_plan_path = amort.OUTPUT + "/plan.json"
    prior_result_path = amort.OUTPUT + "/results.json"
    prior_freeze_path = amort.OUTPUT + "/freeze.json"
    paths = (prior_plan_path, prior_result_path, prior_freeze_path, PROTOCOL, *SOURCES)
    inputs = {p: amort.local_path(root, p).read_bytes() for p in paths}
    prior_plan = json.loads(inputs[prior_plan_path])
    freeze = json.loads(inputs[prior_freeze_path])
    if freeze["plan_sha256"] != amort.digest(amort.canonical(prior_plan)):
        raise ValueError("Report-115 plan freeze mismatch")
    gate = gate115(json.loads(inputs[prior_result_path]))  # Must precede even selection.
    for path, sha in prior_plan["input_sha256"].items():
        data = amort.local_path(root, path).read_bytes()
        if amort.digest(data) != sha:
            raise ValueError(f"Frozen report-115 dependency changed: {path}")
        inputs[path] = data
    original = json.loads(inputs[amort.BASE + "/plan.json"])
    originals = {r["group_id"]: r for r in original["rows"] if r["role"] == "check"}
    fits = {r["drive_id"] for r in original["rows"] if r["role"] == "fit"}
    maps = {r["channel_map_id"]: r for r in map(json.loads, inputs[amort.MAPS].decode().splitlines())}
    rows, files, groups = select_windows(
        list(csv.DictReader(io.StringIO(inputs[amort.MANIFEST].decode()))),
        originals, maps, json.loads(inputs[ENVELOPE]), fits)
    return {
        "kind": "feasibility-npz-transfer-plan-v1", "gate115": gate,
        "rows": rows, "files": files, "coverage": groups,
        "geometry": {r["sample_path"]: amort.physical_ids(r, maps) for r in rows},
        "installations": prior_plan["installations"], "ratios": [.03, .1],
        "input_sha256": {p: amort.digest(data) for p, data in inputs.items()},
        "waveform_cells_read": 0, "f4_authorized": False,
        "planned_npz_cell_occurrences": sum(int(r["channels"]) * amort.RESET for r in rows),
        "temporal_scope": prior_plan["temporal_scope"] + " No cross-file/session independence claim.",
        "float_probability_policy": prior_plan["float_probability_policy"],
        "checkpoints": [1, 5, 10, 20],
        "deployment": "One exact installation ZIP per GROUP/ratio/mode, not per file. "
                      "Group ZIP_STORED contains per-file ZIP_STORED frames+bindings. "
                      "File transport curves exclude shared installation; group/aggregate totals include it.",
        "success_policy": "spatiotemporal vs marginal: >=10% and >=0.05 bpc aggregate net saving "
                          "at horizon 20 in both ratios; no covered configuration loses >1%",
    }


def run(plan, output, root=ROOT, loader=amort.load_window):
    if plan.get("kind") != "feasibility-npz-transfer-plan-v1" or plan.get("f4_authorized") is not False:
        raise ValueError("Invalid transfer plan")
    for path, sha in plan["input_sha256"].items():
        if amort.digest(amort.local_path(root, path).read_bytes()) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    # Recompute the gate; do not trust the recorded decision in the plan.
    gate115(json.loads(amort.local_path(root, amort.OUTPUT + "/results.json").read_bytes()))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    amort.write_json(output / "plan.json", plan)
    amort.write_json(output / "freeze.json", {
        "plan_sha256": amort.digest(amort.canonical(plan)), "waveform_cells_read": 0,
        "authorization": PROTOCOL, "gate115": plan["gate115"],
        "execution": "CPU; exact installations; no fit/calibration/HDF5/val/reserves/GPU"})
    try:
        return execute_plan(plan, output, root, loader)
    except Exception as exc:
        amort.write_json(output / "blocker.json", {
            "error_type": type(exc).__name__, "error": str(exc), "selection_changed": False})
        raise


def transfer_decision(aggregates, curves):
    decisions = {}
    for ratio in (.03, .1):
        arms = {c["mode"]: c for c in aggregates if c["ratio"] == ratio and c["horizon"] == 20}
        marginal, contextual = arms["marginal"], arms["spatiotemporal"]
        saving = marginal["total_bytes"] - contextual["total_bytes"]
        fraction = saving / marginal["total_bytes"]
        bpc = 8 * saving / marginal["cells"]
        group_losses = {
            c["group_id"]: -c["total_bytes_saving_vs_marginal"] /
            (c["total_bytes"] + c["total_bytes_saving_vs_marginal"])
            for c in curves if c["ratio"] == ratio and c["mode"] == "spatiotemporal"
            and c["horizon"] == 20}
        decisions[str(ratio)] = {
            "net_saving_bytes": saving, "net_saving_fraction": fraction, "net_saving_bpc": bpc,
            "configuration_loss_fractions": group_losses,
            "passes": fraction >= .1 and bpc >= .05 and all(loss <= .01 for loss in group_losses.values()),
        }
    return {"ratios": decisions, "passes_both": all(d["passes"] for d in decisions.values()),
            "f4_authorized": False}


def execute_plan(plan, output, root, loader):
    started = time.perf_counter()
    records, file_curves, curves = [], [], []
    read_cells = 0
    for gi, group in enumerate(sorted(plan["coverage"])):
        rows = [r for r in plan["rows"] if r["group_id"] == group]
        windows = [loader(root, row, plan["geometry"][row["sample_path"]]) for row in rows]
        read_cells += sum(w["raw"].size for w in windows)
        file_ids = sorted({w["drive_id"] for w in windows})
        for ratio in plan["ratios"]:
            signatures = {}
            for mode in amort.MODES:
                spec = plan["installations"][f"{group}|{ratio:g}|{mode}"]
                blob = amort.local_path(root, spec["path"]).read_bytes()
                model, installation = amort.read_installation(
                    blob, spec["sha256"], mode, spec["epsilon"], spec["channel_ids"],
                    (root / "compression/feasibility_portable.py").read_bytes())
                prefix = f"group{gi:02d}_{ratio:g}_{mode}"
                (output / f"{prefix}.installation.zip").write_bytes(blob)
                payloads, file_stats = {}, {}
                for fid in file_ids:
                    entries, bindings, arm_records = {}, {}, []
                    fw = [w for w in windows if w["drive_id"] == fid]
                    for index, w in enumerate(fw, 1):
                        before = time.perf_counter()
                        qw = amort.quantized_window(w, model.epsilon)
                        record, transport = amort.evaluate(qw, model, installation)
                        signature = (record["q_sha256"], record["waveform_sha256"])
                        if w["sample_path"] in signatures and signatures[w["sample_path"]] != signature:
                            raise RuntimeError("Cross-mode roundtrip disagreement")
                        signatures[w["sample_path"]] = signature
                        record.update({"group_id": group, "drive_id": fid, "ratio": ratio,
                                       "mode": mode, "window_index": index,
                                       "sample_path": w["sample_path"], "epsilon": model.epsilon,
                                       "k": model.k, "elapsed_seconds": time.perf_counter() - before})
                        records.append(record)
                        arm_records.append(record)
                        name = f"{index:03d}.fqs"
                        entries[name] = transport
                        bindings[name] = {"sample_path": w["sample_path"],
                                          "installation_sha256": spec["table_sha256"]}
                        payload = amort.zip_bytes({**entries, "bindings.json": amort.canonical(bindings)})
                        payloads[(fid, index)] = payload
                        cells = sum(r["cells"] for r in arm_records)
                        stat = {
                            "group_id": group, "drive_id": fid, "ratio": ratio, "mode": mode,
                            "windows": index, "cells": cells, "transport_bytes": len(payload),
                            "frame_bytes": sum(len(b) for b in entries.values()),
                            "integer_cdf_nll_bits": sum(r["integer_cdf_nll_bits"] for r in arm_records),
                            "transport_sha256": amort.digest(payload),
                            "installation_charge": "shared once per GROUP; excluded from file transport",
                        }
                        file_stats[(fid, index)] = stat
                        file_curves.append(stat)
                arm_curves = []
                if file_ids:
                    for horizon in range(1, 21):
                        selected = [(fid, min(horizon, plan["files"][fid]["selected"])) for fid in file_ids]
                        payload = amort.zip_bytes({fid + ".transport.zip": payloads[(fid, n)]
                                                   for fid, n in selected})
                        stats = [file_stats[(fid, n)] for fid, n in selected]
                        curve = {
                            "group_id": group, "ratio": ratio, "mode": mode, "horizon": horizon,
                            "files": len(stats), "windows": sum(n for _, n in selected),
                            "available_windows_by_file": dict(selected),
                            "transport_bytes": len(payload), "installation_bytes": len(blob),
                            "total_bytes": len(payload) + len(blob),
                            "transport_sha256": amort.digest(payload),
                        }
                        for key in ("cells", "frame_bytes", "integer_cdf_nll_bits"):
                            curve[key] = sum(s[key] for s in stats)
                        for key, numerator in (("transport_bpc", 8 * len(payload)),
                                               ("total_bpc", 8 * curve["total_bytes"]),
                                               ("integer_cdf_bpc", curve["integer_cdf_nll_bits"])):
                            curve[key] = numerator / curve["cells"]
                        curves.append(curve)
                        arm_curves.append(curve)
                        if horizon in plan["checkpoints"]:
                            (output / f"{prefix}.n{horizon:02d}.transport.zip").write_bytes(payload)
                amort.write_json(output / f"{prefix}.json", {
                    "installation": spec, "curves": arm_curves,
                    "file_curves": [s for s in file_curves if s["group_id"] == group
                                    and s["ratio"] == ratio and s["mode"] == mode]})
    aggregates = []
    for ratio in plan["ratios"]:
        for horizon in range(1, 21):
            for mode in amort.MODES:
                selected = [c for c in curves if c["ratio"] == ratio
                            and c["horizon"] == horizon and c["mode"] == mode]
                a = {"ratio": ratio, "horizon": horizon, "mode": mode, "groups": len(selected)}
                for key in ("files", "windows", "cells", "frame_bytes", "transport_bytes",
                            "installation_bytes", "total_bytes", "integer_cdf_nll_bits"):
                    a[key] = sum(c[key] for c in selected)
                if a["cells"]:
                    a.update({"total_bpc": 8 * a["total_bytes"] / a["cells"],
                              "transport_bpc": 8 * a["transport_bytes"] / a["cells"],
                              "integer_cdf_bpc": a["integer_cdf_nll_bits"] / a["cells"]})
                aggregates.append(a)
    amort.add_contrasts(curves, "horizon")
    amort.add_contrasts(aggregates, "horizon")
    file_lookup = {(c["drive_id"], c["ratio"], c["mode"], c["windows"]): c for c in file_curves}
    for c in file_curves:
        m = file_lookup[(c["drive_id"], c["ratio"], "marginal", c["windows"])]
        c["transport_bytes_saving_vs_marginal"] = m["transport_bytes"] - c["transport_bytes"]
        c["integer_cdf_nll_bits_saving_vs_marginal"] = m["integer_cdf_nll_bits"] - c["integer_cdf_nll_bits"]
    result = {
        "kind": "feasibility-npz-transfer-results-v1", "gate115": plan["gate115"],
        "rows": records, "file_curves": file_curves, "curves": curves, "aggregates": aggregates,
        "coverage": plan["coverage"], "files": plan["files"],
        "npz_cell_occurrences_read": read_cells,
        "temporal_scope": plan["temporal_scope"], "deployment": plan["deployment"],
        "float_probability_policy": plan["float_probability_policy"],
        "decision": transfer_decision(aggregates, curves),
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "fit_calls": 0, "f4_authorized": False,
    }
    amort.write_json(output / "results.json", result)
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
