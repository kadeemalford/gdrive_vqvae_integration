"""Single frozen spatial-sign pooling candidate; CPU/train NPZ only (report 120)."""
import argparse
import io
import json
import platform
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
    Tables, contexts, integer_frequencies, symbols, SUPPORTS,
)
from finetuning import run_feasibility_npz_transfer as transfer

a = transfer.amort
OUTPUT = "results/experiments/feasibility_pooled_context_20261118"
REFERENCE = transfer.OUTPUT + "/retry_01"
SOURCES = ("finetuning/run_feasibility_pooled_context.py",
           "tests/test_feasibility_pooled_context.py")
# Temporal sign/magnitude is unchanged. Spatial sign alone is removed.
SPATIAL = (0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9)
CONTEXT_GROUPS = np.asarray([t * 10 + s for t in range(18) for s in SPATIAL])
CANDIDATE = {
    "id": "temporal18-spatial10-sign-pool-v1", "context_groups": CONTEXT_GROUPS.tolist(),
    "groups": 180, "wire_contexts": 324, "minimum_count": 256,
    "jeffreys": .5, "backoff_weight": 256, "support_epsilon": "exact report114",
    "selection": "none; exact report116 retry_01 plan rows in original order",
    "decoder": "unchanged FTB1; shared mapping rows", "variants": 1,
}


def fit_pooled(windows, epsilon, k, ids):
    if (not windows or k not in SUPPORTS
            or any(w["role"] != "fit" for w in windows)
            or len({w["group_id"] for w in windows}) != 1):
        raise ValueError("Only original fit windows from one group are permitted")
    alphabet = 2 * k + 2
    counts = np.zeros((180, alphabet), dtype=np.int64)
    for w in windows:
        c = CONTEXT_GROUPS[contexts(w["q"], w["positions"], w["channel_ids"],
                                    "spatiotemporal").reshape(-1)]
        s = symbols(w["q"], k).reshape(-1)
        counts += np.bincount(c * alphabet + s, minlength=180 * alphabet).reshape(180, alphabet)
    total = counts.sum(axis=0)
    marginal = (total + .5) / (int(total.sum()) + .5 * alphabet)
    probabilities, mapping = [marginal], np.zeros(180, dtype=np.uint32)
    for group, row in enumerate(counts):
        n = int(row.sum())
        if n >= 256:
            p = (row + .5) / (n + .5 * alphabet)
            probabilities.append(n / (n + 256) * p + 256 / (n + 256) * marginal)
            mapping[group] = len(probabilities) - 1
    p = np.asarray(probabilities)
    return Tables(k, "spatiotemporal", epsilon, ids,
                  np.asarray([integer_frequencies(row) for row in p]),
                  mapping[CONTEXT_GROUPS], p)


def authorization(root):
    paths = sorted((root / "progress_reports").glob("120-*.md"))
    if not paths:
        paths = sorted((root / "progress_reports/archive/feasibility_protocols_20261118").glob("120-*.md"))
    if len(paths) != 1:
        raise ValueError("Exactly one final report120 is required")
    text = paths[0].read_text()
    title = next((line for line in text.splitlines() if line.startswith("# ")), "")
    if "compatibilidade de tabela contextual" not in title.casefold():
        raise ValueError("Final report120 title is not authorized")
    if "metas provisórias" in title.casefold():
        raise ValueError("Provisional report120 is not authorization")
    return str(paths[0].relative_to(root))


def build_plan(root=ROOT):
    protocol = authorization(root)
    paths = [protocol, *SOURCES, "finetuning/run_feasibility_npz_transfer.py",
             "finetuning/run_feasibility_npz_amortization.py",
             REFERENCE + "/plan.json", REFERENCE + "/freeze.json",
             REFERENCE + "/results.json", a.BASE + "/plan.json"]
    inputs = {p: a.local_path(root, p).read_bytes() for p in paths}
    ref = json.loads(inputs[REFERENCE + "/plan.json"])
    freeze = json.loads(inputs[REFERENCE + "/freeze.json"])
    if (ref["kind"] != "feasibility-npz-transfer-plan-v1"
            or freeze["plan_sha256"] != a.digest(a.canonical(ref))):
        raise ValueError("Frozen retry116 plan mismatch")
    original = json.loads(inputs[a.BASE + "/plan.json"])
    for plan in (ref, original):
        for path, sha in {**plan.get("input_sha256", {}),
                          **plan.get("source_sha256", {})}.items():
            resolved, data = a.frozen_dependency(root, path, sha)
            inputs[resolved] = data
    fits = [r for r in original["rows"] if r["role"] == "fit"]
    if (len(fits) != 18 or len(ref["rows"]) != 380 or len(ref["coverage"]) != 8
            or ref["ratios"] != [.03, .1]):
        raise ValueError("Original fit/check census mismatch")
    for row in [*fits, *ref["rows"]]:
        path = Path(row["sample_path"])
        if (row["split"] != "train" or path.suffix != ".npz" or ".." in path.parts
                or path.parts[:4] != ("data", "finetune",
                                     "spatial_band_v1_fold0_20261024", "train")):
            raise ValueError("Disallowed waveform path")
    return {
        "kind": "feasibility-pooled-context-plan-v1", "candidate": CANDIDATE,
        "authorization": protocol, "fit_rows": fits, "rows": ref["rows"],
        "fit_geometry": original["geometry"], "geometry": ref["geometry"],
        "coverage": ref["coverage"], "files": ref["files"],
        "installations": ref["installations"], "ratios": ref["ratios"],
        "checkpoints": [1, 5, 10, 20], "waveform_cells_read": 0,
        "input_sha256": {p: a.digest(b) for p, b in inputs.items()},
        "source_sha256": {p: a.digest(b) for p, b in inputs.items() if p.endswith(".py")},
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "platform": platform.platform(), "executable": sys.executable,
                    "executable_sha256": a.digest(Path(sys.executable).read_bytes())},
        "f4_authorized": False, "independent_confirmation": False,
        "scope": "Reused development; no selection/tuner/check calibration; no F4",
    }


def check_hashes(record, baseline):
    for key in ("q_sha256", "waveform_sha256", "original_float32_sha256"):
        if record[key] != baseline[key]:
            raise RuntimeError(f"Report116 hash disagreement: {key}")


def primary_decision(aggregates, curves):
    primary = next(c for c in aggregates if c["ratio"] == .03 and c["horizon"] == 20)
    saving = primary["vs_spatiotemporal"]["total_bytes_saving"]
    fraction = saving / (primary["total_bytes"] + saving)
    losses = {}
    for c in curves:
        if c["ratio"] == .03 and c["horizon"] == 20:
            delta = c["vs_spatiotemporal"]["total_bytes_saving"]
            losses[c["group_id"]] = -delta / (c["total_bytes"] + delta)
    return {"primary_ratio": .03, "primary_horizon": 20,
            "net_saving_bytes": saving, "net_saving_fraction": fraction,
            "configuration_loss_fractions": losses,
            "passes": fraction >= .03 and all(loss <= .01 for loss in losses.values()),
            "f4_authorized": False, "independent_confirmation": False}


def execute_plan(plan, output, root, loader):
    started = time.perf_counter()
    baseline = json.loads(a.local_path(root, REFERENCE + "/results.json").read_bytes())
    lookup = {(r["sample_path"], r["ratio"], r["mode"]): r for r in baseline["rows"]}
    records, curves, file_curves, installations = [], [], [], []
    read_cells = 0
    for gi, group in enumerate(sorted(plan["coverage"])):
        fit_rows = [r for r in plan["fit_rows"] if r["group_id"] == group]
        fits = [loader(root, r, plan["fit_geometry"][r["sample_path"]]) for r in fit_rows]
        read_cells += sum(w["raw"].size for w in fits)
        # All installations for a group are frozen before reading any group check.
        models = {}
        for ratio in plan["ratios"]:
            spec = plan["installations"][f"{group}|{ratio:g}|spatiotemporal"]
            model = fit_pooled([a.quantized_window(w, spec["epsilon"]) for w in fits],
                               spec["epsilon"], spec["k"], spec["channel_ids"])
            table = model.dump()
            with zipfile.ZipFile(io.BytesIO(a.local_path(root, spec["path"]).read_bytes())) as z:
                blob = a.zip_bytes({a.digest(table) + ".ftb": table,
                                    "decoder.py": z.read("decoder.py"),
                                    "platform.json": z.read("platform.json")})
            # Use the same deserialization and byte-only decoder as the baseline.
            loaded, wire = a.read_installation(
                blob, a.digest(blob), "spatiotemporal", spec["epsilon"], spec["channel_ids"],
                (root / "compression/feasibility_portable.py").read_bytes())
            prefix = f"group{gi:02d}_{ratio:g}_pooled"
            (output / (prefix + ".installation.zip")).write_bytes(blob)
            installations.append({"group_id": group, "ratio": ratio, "k": loaded.k,
                                  "epsilon": loaded.epsilon, "rows": len(loaded.frequencies),
                                  "contexts": len(loaded.mapping), "bytes": len(blob),
                                  "sha256": a.digest(blob), "table_sha256": a.digest(table)})
            models[ratio] = loaded, wire, blob, prefix
        a.write_json(output / f"group{gi:02d}.fit_freeze.json", {
            "check_waveform_cells_read": 0,
            "installations": [i for i in installations if i["group_id"] == group]})
        rows = [r for r in plan["rows"] if r["group_id"] == group]
        windows = [loader(root, r, plan["geometry"][r["sample_path"]]) for r in rows]
        read_cells += sum(w["raw"].size for w in windows)
        file_ids = sorted({w["drive_id"] for w in windows})
        for ratio, (model, table, blob, prefix) in models.items():
            payloads, stats = {}, {}
            for fid in file_ids:
                entries, bindings, arm = {}, {}, []
                for index, w in enumerate([w for w in windows if w["drive_id"] == fid], 1):
                    record, transport = a.evaluate(a.quantized_window(w, model.epsilon), model, table)
                    for mode in ("marginal", "spatiotemporal"):
                        check_hashes(record, lookup[(w["sample_path"], ratio, mode)])
                    record.update(group_id=group, drive_id=fid, ratio=ratio, mode="pooled",
                                  sample_path=w["sample_path"], window_index=index,
                                  epsilon=model.epsilon, k=model.k, matches_report116_hashes=True)
                    records.append(record)
                    arm.append(record)
                    name = f"{index:03d}.fqs"
                    entries[name] = transport
                    bindings[name] = {"sample_path": w["sample_path"],
                                      "installation_sha256": a.digest(table)}
                    payload = a.zip_bytes({**entries, "bindings.json": a.canonical(bindings)})
                    payloads[(fid, index)] = payload
                    stat = dict(group_id=group, drive_id=fid, ratio=ratio, mode="pooled",
                                windows=index, cells=sum(r["cells"] for r in arm),
                                transport_bytes=len(payload), frame_bytes=sum(map(len, entries.values())),
                                integer_cdf_nll_bits=sum(r["integer_cdf_nll_bits"] for r in arm),
                                transport_sha256=a.digest(payload))
                    stats[(fid, index)] = stat
                    file_curves.append(stat)
            for horizon in range(1, 21):
                selected = [(fid, min(horizon, plan["files"][fid]["selected"])) for fid in file_ids]
                payload = a.zip_bytes({fid + ".transport.zip": payloads[(fid, n)] for fid, n in selected})
                curve = dict(group_id=group, ratio=ratio, mode="pooled", horizon=horizon,
                             files=len(file_ids), windows=sum(n for _, n in selected),
                             transport_bytes=len(payload), installation_bytes=len(blob),
                             total_bytes=len(payload) + len(blob), transport_sha256=a.digest(payload))
                for key in ("cells", "frame_bytes", "integer_cdf_nll_bits"):
                    curve[key] = sum(stats[(fid, n)][key] for fid, n in selected)
                curves.append(curve)
                if horizon in plan["checkpoints"]:
                    (output / f"{prefix}.n{horizon:02d}.transport.zip").write_bytes(payload)
            a.write_json(output / (prefix + ".json"), {
                "curves": [c for c in curves if c["group_id"] == group and c["ratio"] == ratio]})
    aggregates = []
    for ratio in plan["ratios"]:
        for horizon in range(1, 21):
            selected = [c for c in curves if c["ratio"] == ratio and c["horizon"] == horizon]
            aggregate = dict(ratio=ratio, horizon=horizon, mode="pooled", groups=len(selected))
            for key in ("files", "windows", "cells", "frame_bytes", "transport_bytes",
                        "installation_bytes", "total_bytes", "integer_cdf_nll_bits"):
                aggregate[key] = sum(c[key] for c in selected)
            aggregates.append(aggregate)
    for collection, reference, scope in (
            (curves, baseline["curves"], "horizon"),
            (aggregates, baseline["aggregates"], "horizon"),
            (file_curves, baseline["file_curves"], "windows")):
        keys = ("transport_bytes", "integer_cdf_nll_bits")
        if scope == "horizon":
            keys += ("total_bytes", "installation_bytes")
        refs = {(c.get("group_id"), c.get("drive_id"), c["ratio"], c[scope], c["mode"]): c
                for c in reference}
        for c in collection:
            c["transport_bpc"] = 8 * c["transport_bytes"] / c["cells"]
            c["integer_cdf_bpc"] = c["integer_cdf_nll_bits"] / c["cells"]
            if "total_bytes" in c:
                c["total_bpc"] = 8 * c["total_bytes"] / c["cells"]
            for mode in ("marginal", "spatiotemporal"):
                ref = refs[(c.get("group_id"), c.get("drive_id"), c["ratio"], c[scope], mode)]
                if (c["cells"], c["windows"]) != (ref["cells"], ref["windows"]):
                    raise RuntimeError("Baseline population mismatch")
                c["vs_" + mode] = {key + "_saving": ref[key] - c[key] for key in keys}
    result = dict(kind="feasibility-pooled-context-results-v1", candidate=CANDIDATE,
                  rows=records, curves=curves, aggregates=aggregates, file_curves=file_curves,
                  installations=installations, coverage=plan["coverage"], files=plan["files"],
                  decision=primary_decision(aggregates, curves),
                  npz_cell_occurrences_read=read_cells, fit_windows=len(plan["fit_rows"]),
                  fit_calls=len(installations), check_windows=len(plan["rows"]),
                  baseline_reference=REFERENCE, elapsed_seconds=time.perf_counter() - started,
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  f4_authorized=False, independent_confirmation=False)
    a.write_json(output / "results.json", result)
    return result


def run(plan, output, root=ROOT, loader=a.load_window):
    if (plan["candidate"] != CANDIDATE or plan["f4_authorized"] is not False
            or plan["authorization"] != authorization(root)):
        raise ValueError("Invalid frozen candidate/authorization")
    if a.digest(Path(sys.executable).read_bytes()) != plan["runtime"]["executable_sha256"]:
        raise ValueError("Python executable changed")
    for path, sha in plan["input_sha256"].items():
        if a.digest(a.local_path(root, path).read_bytes()) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    a.write_json(output / "plan.json", plan)
    a.write_json(output / "freeze.json", {
        "plan_sha256": a.digest(a.canonical(plan)),
        "candidate_sha256": a.digest(a.canonical(CANDIDATE)),
        "waveform_cells_read": 0, "execution": "CPU; original fit114 + exact check116 retry"})
    try:
        return execute_plan(plan, output, root, loader)
    except Exception as exc:
        a.write_json(output / "blocker.json", {"error": str(exc), "selection_changed": False})
        raise


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
