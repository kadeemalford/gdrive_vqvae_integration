"""One prospective CPU fit-only routing policy; checks are reused development."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.feasibility_installation_envelope import pack, unpack
from finetuning.run_feasibility_continuous_probe import inspect_sources, load_region
from finetuning.run_feasibility_npz_probe import (
    canonical, choose_support, digest, evaluate_window, fit_windows,
    quantized_window, zip_bytes,
)

SOURCE = ROOT / "results/experiments/feasibility_continuous_probe_20261118"
ENVELOPES = ROOT / "results/experiments/feasibility_installation_envelope_20261118"
OUTPUT = ROOT / "results/experiments/feasibility_fit_routing_20261118"
REPORT = ROOT / "progress_reports/120-FEBUS_FIT_ONLY_INSTALLATION_ROUTING.md"
RATIOS = (.03, .1)
MODES = ("marginal", "spatiotemporal")
OWN = ("finetuning/run_feasibility_fit_routing.py", "tests/test_feasibility_fit_routing.py")
CODEC = "compression/feasibility_installation_envelope.py"
BINDING_KEYS = ("drive_id", "dataset", "block_index", "region_index",
                "source_time_start", "time_samples", "sample_path")


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": digest(data)}


def verify(path, expected):
    if identity(path) != expected:
        raise ValueError(f"Frozen bytes changed: {path}")


def check_budget(started, budget=3600):
    if time.monotonic() - started >= budget:
        raise TimeoutError("60-minute execution budget exhausted; no policy revision authorized")


def select_mode(folds, fit_file_count):
    """Integer threshold, every whole held-out FIT file; equality is accepted."""
    if fit_file_count < 2:
        return "marginal"
    if (len(folds) != fit_file_count
            or len({f["held_out_drive_id"] for f in folds}) != fit_file_count):
        raise ValueError("Incomplete leave-one-FIT-file-out evidence")
    return ("spatiotemporal" if all(
        f["marginal_total_bytes"] > 0
        and 100 * (f["marginal_total_bytes"] - f["spatiotemporal_total_bytes"])
        >= 3 * f["marginal_total_bytes"] for f in folds) else "marginal")


def split_fit(windows, held_out):
    if any(w["role"] != "fit" for w in windows):
        raise ValueError("Only FIT waveforms may enter routing")
    train = [w for w in windows if w["drive_id"] != held_out]
    held = [w for w in windows if w["drive_id"] == held_out]
    if not train or not held:
        raise ValueError("Empty training or held-out FIT file")
    return train, held


def freeze(output=OUTPUT):
    """Hash opaque historical ledgers, not their evaluated contents, before waves."""
    if not REPORT.is_file():
        raise ValueError("Active prospective report120 must exist before execution")
    plan = json.loads((SOURCE / "plan.json").read_text())
    old_freeze = json.loads((SOURCE / "freeze.json").read_text())
    if digest(canonical(plan)) != old_freeze["plan_sha256"]:
        raise ValueError("D2c plan does not match its original freeze")
    for name, sha in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != sha:
            raise ValueError(f"Historical D2c implementation changed: {name}")
    # Report117 acquired a results section after its execution, so its original
    # prospective document digest is provenance, not an immutable current file.
    for name, sha in plan["input_sha256"].items():
        if name.startswith("progress_reports/"):
            continue
        if digest((ROOT / name).read_bytes()) != sha:
            raise ValueError(f"D2c metadata changed: {name}")
    fit_rows = [r for r in plan["rows"] if r["role"] == "fit"]
    if (len({r["drive_id"] for r in fit_rows}) != 20
            or len(plan["sigma_p0"]) != 9
            or sum(r["time_samples"] * r["channels"] for r in fit_rows) != 11_670_000):
        raise ValueError("Unexpected frozen FIT census")
    source_names = set(plan["source_sha256"]) | set(OWN) | {
        CODEC, "tools/freeze_feasibility_metadata.py"}
    inputs = {str(p.relative_to(ROOT)): identity(p) for p in (
        SOURCE / "plan.json", SOURCE / "freeze.json", SOURCE / "protocol.json",
        SOURCE / "results.json", SOURCE / "completion_evidence.json",
        ENVELOPES / "results.json", ENVELOPES / "completion_evidence.json",
        ENVELOPES / "protocol.json", ENVELOPES / "freeze.json",
        REPORT, ROOT / "progress_reports/107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md",
        ROOT / "progress_reports/118-FEBUS_LOSSLESS_INSTALLATION_ENVELOPE.md",
        ROOT / "progress_reports/119-FEBUS_LOSSLESS_INSTALLATION_RESULTS.md",
    )}
    for name in plan["input_sha256"]:
        inputs[name] = identity(ROOT / name)
    frozen = {
        "kind": "feasibility-fit-routing-freeze-v1",
        "ratios": list(RATIOS), "modes": list(MODES), "threshold_percent": 3,
        "fit_plan": {**plan, "rows": fit_rows},
        "source_files": {name: identity(ROOT / name) for name in sorted(source_names)},
        "inputs": inputs, "route_bytes_per_group_ratio_both_arms": 1,
        "source_charge": "full routing runner plus envelope decoder source uncompressed, once per shared arm/ratio; once per isolated configuration; identical in both arms and FIT contrasts",
        "source_charge_bytes": sum((ROOT / p).stat().st_size for p in (OWN[0], CODEC)),
        "selection": "all whole held-out FIT-file installed+transport contrasts >=3%; <2 FIT files marginal; support/CDF use other FIT files only",
        "envelope": "exact report118 pack, preset6 XZ/raw, no new variants",
        "check_policy": "read frozen report117/119 ledgers and artifacts only after decisions are saved and hash-frozen; no check HDF5 or encoder calls",
        "check_population": "reused development only, not independent confirmation",
        "fit_slice_integrity": "historical117 has no FIT waveform digest; verify original plan/metadata/source hashes and headers, record newly read exact FIT slice hashes; no historical bit-identity claim",
        "budget_seconds": 3600, "gpu_used": False, "f4_authorized": False,
        "no_posthoc_adjustment": True, "waveform_cells_read": 0,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", frozen)
    write_json(output / "freeze.json", identity(output / "protocol.json"))
    return frozen


def verify_freeze(output):
    verify(output / "protocol.json", json.loads((output / "freeze.json").read_text()))
    frozen = json.loads((output / "protocol.json").read_text())
    for name, expected in {**frozen["source_files"], **frozen["inputs"]}.items():
        verify(ROOT / name, expected)
    return frozen


def fit_arm(held, model, epsilon, platform, source_bytes, artifact_prefix):
    """Exact D2c ZIP layout and bindings, wrapped with the unchanged 118 codec."""
    transports, installs, bindings, records = {}, {}, {}, []
    for wi, window in enumerate(held):
        record, transport, installation, _ = evaluate_window(window, model, epsilon)
        name, ih = f"{wi:03d}", digest(installation)
        transports[name + ".fqs"] = transport
        installs[ih + ".ftb"] = installation
        bindings[name] = {k: window[k] for k in BINDING_KEYS}
        bindings[name]["installation_sha256"] = ih
        records.append({**bindings[name], **record})
    transports["bindings.json"] = canonical(bindings)
    installs["decoder.py"] = (ROOT / "compression/feasibility_portable.py").read_bytes()
    installs["platform.json"] = canonical({"platform": platform})
    transport, installation = zip_bytes(transports), zip_bytes(installs)
    envelope = pack(installation)
    if unpack(envelope) != installation:
        raise ValueError("FIT installation roundtrip failed")
    artifacts = {}
    for suffix, data in ((".transport.zip", transport), (".installation.zip.fie", envelope)):
        path = Path(str(artifact_prefix) + suffix)
        with path.open("xb") as stream:
            stream.write(data)
        artifacts[path.name] = {"bytes": len(data), "sha256": digest(data)}
    return {
        "transport_bytes": len(transport), "envelope_bytes": len(envelope),
        "installation_original_bytes": len(installation),
        "installation_original_sha256": digest(installation),
        "source_bytes": source_bytes, "route_bytes": 1,
        "total_bytes": len(transport) + len(envelope) + source_bytes + 1,
        "cells": sum(w["raw"].size for w in held), "records": records,
        "artifacts": artifacts,
    }


def fit_decisions(frozen, output, started):
    plan, decisions, slice_hashes, read_cells = frozen["fit_plan"], [], [], 0
    audit = inspect_sources(ROOT, plan)
    write_json(output / "source_audit.json", audit)
    if audit["blockers"]:
        raise ValueError("FIT source header audit blocked; see source_audit.json")
    for gi, group in enumerate(sorted(plan["sigma_p0"])):
        check_budget(started)
        windows = []
        for row in plan["rows"]:
            if row["group_id"] != group:
                continue
            if row["role"] != "fit":
                raise ValueError("Non-FIT row reached waveform loader")
            w = load_region(ROOT, row, plan["geometry"][row["sample_path"]])
            windows.append(w)
            read_cells += w["raw"].size
            slice_hashes.append({"sample_path": row["sample_path"], "cells": w["raw"].size,
                                 "original_float32_sha256": digest(w["raw"].astype("<f4").tobytes())})
        file_ids = sorted({w["drive_id"] for w in windows})
        for ratio in RATIOS:
            epsilon = ratio * plan["sigma_p0"][group]
            quantized = [quantized_window(w, epsilon) for w in windows]
            folds = []
            if len(file_ids) >= 2:
                for fi, fid in enumerate(file_ids):
                    check_budget(started)
                    train, held = split_fit(quantized, fid)
                    k, candidates = choose_support(train, epsilon)
                    arms = {}
                    for mode in MODES:
                        check_budget(started)
                        model = fit_windows(train, epsilon, k, mode)
                        prefix = output / f"fit_group{gi:02d}_{ratio:g}_fold{fi:02d}_{mode}"
                        arms[mode] = fit_arm(held, model, epsilon, plan["deployment_platform"],
                                             frozen["source_charge_bytes"], prefix)
                    for left, right in zip(arms["marginal"]["records"],
                                           arms["spatiotemporal"]["records"]):
                        for key in ("q_sha256", "waveform_sha256", "original_float32_sha256"):
                            if left[key] != right[key]:
                                raise ValueError("FIT arm reconstruction mismatch")
                    fold = {
                        "held_out_drive_id": fid,
                        "training_drive_ids": sorted({w["drive_id"] for w in train}),
                        "held_out_sample_paths": [w["sample_path"] for w in held],
                        "k": k, "support_candidates_training_only": candidates, "arms": arms,
                        "marginal_total_bytes": arms["marginal"]["total_bytes"],
                        "spatiotemporal_total_bytes": arms["spatiotemporal"]["total_bytes"],
                    }
                    fold["saving_fraction"] = (
                        fold["marginal_total_bytes"] - fold["spatiotemporal_total_bytes"]
                    ) / fold["marginal_total_bytes"]
                    folds.append(fold)
                    write_json(output / f"fit_group{gi:02d}_{ratio:g}_fold{fi:02d}.json", fold)
            decision = {"group_id": group, "family": group.split("/")[0], "ratio": ratio,
                        "fit_file_count": len(file_ids), "folds": folds,
                        "mode": select_mode(folds, len(file_ids))}
            decisions.append(decision)
            write_json(output / f"decision_group{gi:02d}_{ratio:g}.json", decision)
            print(f"frozen FIT decision {group} ratio={ratio:g}: {decision['mode']}", flush=True)
    payload = {"decisions": decisions, "fit_slice_hashes": slice_hashes,
               "fit_cells_read": read_cells, "check_waveform_cells_read": 0,
               "evaluated_check_ledgers_read_before_decisions": 0}
    write_json(output / "decisions.json", payload)
    write_json(output / "decisions.freeze.json", identity(output / "decisions.json"))
    return payload


def aggregate(rows, source_bytes):
    summaries = []
    for ratio in RATIOS:
        selected = [r for r in rows if r["ratio"] == ratio]
        for deployment in ("shared_once", "isolated"):
            charge = source_bytes * (1 if deployment == "shared_once" else len(selected))
            base = sum(r["marginal_payload_bytes"] + 1 for r in selected) + charge
            candidate = sum(r["selected_payload_bytes"] + 1 for r in selected) + charge
            cells = sum(r["cells"] for r in selected)
            family = {f: sum(r["saving_bytes"] for r in selected if r["family"] == f)
                      for f in sorted({r["family"] for r in selected})}
            regressions = {r["group_id"]: -r["saving_bytes"] /
                           (r["marginal_payload_bytes"] + 1 + source_bytes) for r in selected}
            saving, bpc = (base - candidate) / base, 8 * (base - candidate) / cells
            rate_pass = saving >= .1 and bpc >= .05
            strict = (len(selected) == 9 and len(family) == 4 and rate_pass
                      and all(v > 0 for v in family.values())
                      and all(v <= .01 for v in regressions.values()))
            summaries.append({
                "ratio": ratio, "deployment": deployment, "marginal_total_bytes": base,
                "selected_total_bytes": candidate, "cells": cells,
                "source_charge_bytes_per_arm": charge, "route_bytes_per_arm": len(selected),
                "saving_bytes": base - candidate, "saving_fraction": saving, "saving_bpc": bpc,
                "family_saving_bytes": family, "configuration_regression_fractions": regressions,
                "aggregate_rate_thresholds_pass": rate_pass,
                "no_configuration_regression": all(v <= 0 for v in regressions.values()),
                "strictly_positive_all_families": len(family) == 4 and all(v > 0 for v in family.values()),
                "measured_coverage_thresholds_pass": strict,
                "generalist_gate": None, "f4_authorized": False,
            })
    return summaries


def evaluate_frozen_check(frozen, output, started):
    """No raw inputs, encoder, model fitting or selection from check bytes here."""
    verify(output / "decisions.json", json.loads((output / "decisions.freeze.json").read_text()))
    payload = json.loads((output / "decisions.json").read_text())
    decisions = payload["decisions"]
    expected = {(g, r) for g in frozen["fit_plan"]["sigma_p0"] for r in RATIOS}
    if {(d["group_id"], d["ratio"]) for d in decisions} != expected or len(decisions) != 18:
        raise ValueError("Incomplete frozen routing panel")
    for d in decisions:
        if select_mode(d["folds"], d["fit_file_count"]) != d["mode"]:
            raise ValueError("Decision differs from frozen FIT-only rule")
    old = json.loads((SOURCE / "results.json").read_text())
    new = json.loads((ENVELOPES / "results.json").read_text())
    evidence = json.loads((SOURCE / "completion_evidence.json").read_text())
    if (evidence.get("complete") is not True
            or evidence.get("all_q_waveform_portable_roundtrips") is not True
            or evidence.get("bound_violations") != 0
            or evidence.get("result_sha256") != identity(SOURCE / "results.json")["sha256"]):
        raise ValueError("Original D2c completion evidence is not valid")
    historical_artifacts = {Path(a["path"]).name: {k: a[k] for k in ("bytes", "sha256")}
                            for a in evidence["artifacts"]}
    envelope_evidence = json.loads((ENVELOPES / "completion_evidence.json").read_text())
    if (envelope_evidence.get("complete") is not True
            or envelope_evidence.get("results") != identity(ENVELOPES / "results.json")
            or envelope_evidence.get("decoder_source", {}).get("sha256")
            != digest((ROOT / CODEC).read_bytes())):
        raise ValueError("Envelope completion/source evidence is not valid")
    verify(ENVELOPES / "installation_decoder.py",
           {k: envelope_evidence["decoder_source"][k] for k in ("bytes", "sha256")})
    envelope_artifacts = {Path(a["path"]).name: {k: a[k] for k in ("bytes", "sha256")}
                          for a in envelope_evidence["artifacts"]}
    old_configs = {(c["group_id"], c["ratio"]): c for c in old["configurations"]}
    new_rows = {(r["group_id"], r["ratio"], r["mode"]): r for r in new["rows"]}
    rows = []
    route_file = bytearray()
    for d in decisions:
        check_budget(started)
        group, ratio = d["group_id"], d["ratio"]
        config = old_configs[group, ratio]
        arms = {}
        for mode in MODES:
            r = new_rows[group, ratio, mode]
            original = config["arms"][mode]
            installation_name = r["original_installation_path"]
            transport_name = installation_name.replace(".installation.zip", ".transport.zip")
            for name, kind in ((installation_name, "installation"), (transport_name, "transport")):
                expected_id = {k: original[kind + "_" + k] for k in ("bytes", "sha256")}
                if historical_artifacts[name] != expected_id:
                    raise ValueError("Original completion inventory/ledger differs")
                verify(SOURCE / name, expected_id)
            env_path = ENVELOPES / r["envelope_path"]
            env_id = {k: r["envelope_" + k] for k in ("bytes", "sha256")}
            if envelope_artifacts[env_path.name] != env_id:
                raise ValueError("Envelope completion inventory/ledger differs")
            verify(env_path, env_id)
            restored = unpack(env_path.read_bytes())
            if restored != (SOURCE / installation_name).read_bytes():
                raise ValueError("Check installation envelope is not exact original ZIP")
            if (r["transport_sha256"] != original["transport_sha256"]
                    or r["transport_bytes"] != original["transport_bytes"]):
                raise ValueError("Check transport changed")
            records = [x for x in old["rows"] if x["group_id"] == group
                       and x["ratio"] == ratio and x["mode"] == mode]
            arms[mode] = {
                "transport_bytes": original["transport_bytes"],
                "envelope_bytes": r["envelope_bytes"],
                "payload_bytes": original["transport_bytes"] + r["envelope_bytes"],
                "cells": original["cells"],
                "paired_hashes": {x["sample_path"]: tuple(x[k] for k in (
                    "q_sha256", "waveform_sha256", "original_float32_sha256")) for x in records},
                "artifacts": {str(env_path.relative_to(ROOT)): env_id,
                              str((SOURCE / transport_name).relative_to(ROOT)):
                                  historical_artifacts[transport_name]},
            }
        if (not arms["marginal"]["paired_hashes"]
                or arms["marginal"]["paired_hashes"] != arms["spatiotemporal"]["paired_hashes"]
                or arms["marginal"]["cells"] != arms["spatiotemporal"]["cells"]):
            raise ValueError("Original check arm q/waveform hash identity is not established")
        selected, base = arms[d["mode"]], arms["marginal"]
        route_file.append(int(d["mode"] == "spatiotemporal"))
        rows.append({
            "group_id": group, "family": d["family"], "ratio": ratio,
            "mode": d["mode"], "cells": selected["cells"],
            "marginal_payload_bytes": base["payload_bytes"],
            "selected_payload_bytes": selected["payload_bytes"],
            "saving_bytes": base["payload_bytes"] - selected["payload_bytes"],
            "marginal_isolated_total_bytes": base["payload_bytes"] + 1 + frozen["source_charge_bytes"],
            "selected_isolated_total_bytes": selected["payload_bytes"] + 1 + frozen["source_charge_bytes"],
            "route_bytes_both_arms": 1, "arms": arms,
            "q_waveform_identity_inherited_verified": True,
        })
    with (output / "selected_routes.bin").open("xb") as stream:
        stream.write(route_file)
    with (output / "marginal_routes.bin").open("xb") as stream:
        stream.write(bytes(len(route_file)))
    summaries = aggregate(rows, frozen["source_charge_bytes"])
    result = {
        "kind": "feasibility-fit-routing-results-v1", "rows": rows, "summaries": summaries,
        "decisions_sha256": identity(output / "decisions.json")["sha256"],
        "fit_cells_read": payload["fit_cells_read"], "check_waveform_cells_read": 0,
        "check_encoder_calls": 0, "check_population": "reused development only",
        "gpu_used": False, "new_training_updates": 0, "posthoc_adjustments": 0,
        "generalist_gate": None, "f4_authorized": False,
        "joint_primary_measured_coverage_pass": {
            deployment: all(s["measured_coverage_thresholds_pass"] for s in summaries
                            if s["deployment"] == deployment)
            for deployment in ("shared_once", "isolated")},
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "limits": ["Check was already consumed: development, not independent confirmation.",
                   "No multicore C1/C4, generalist approval or F4 authorization.",
                   frozen["fit_slice_integrity"],
                   "Check q/waveform identity is verified from historical completion and per-frame hashes, not newly decoded.",
                   "Full routing runner source is conservatively paid in both arms; platform dependencies follow 118."],
    }
    verify_freeze(output)
    verify(output / "decisions.json", json.loads((output / "decisions.freeze.json").read_text()))
    write_json(output / "results.json", result)
    return result


def execute(output=OUTPUT):
    started, output = time.monotonic(), Path(output)
    if (output / "execution_started.json").exists():
        raise FileExistsError("Single execution already started; no rerun or tuning authorized")
    frozen = verify_freeze(output)
    write_json(output / "execution_started.json", {"protocol": identity(output / "protocol.json")})
    try:
        fit_decisions(frozen, output, started)
        check_budget(started)
        result = evaluate_frozen_check(frozen, output, started)
        artifacts = {p.name: identity(p) for p in sorted(output.iterdir()) if p.is_file()}
        write_json(output / "completion_evidence.json", {
            "complete": True, "result_sha256": identity(output / "results.json")["sha256"],
            "decisions_frozen_before_check_ledgers": True, "source_freeze_verified_after": True,
            "artifacts": artifacts, "check_waveform_cells_read": 0, "check_encoder_calls": 0,
            "f4_authorized": False,
        })
        print(json.dumps({"summaries": result["summaries"],
                          "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
        return result
    except Exception as exc:
        write_json(output / "blocker.json", {
            "error_type": type(exc).__name__, "error": str(exc),
            "policy_changed": False, "no_rerun_authorized": True, "f4_authorized": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else execute()


if __name__ == "__main__":
    main()
