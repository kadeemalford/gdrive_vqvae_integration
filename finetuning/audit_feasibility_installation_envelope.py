"""Single frozen CPU contrast: wrap intact D2c installation ZIPs, never signals."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.feasibility_installation_envelope import HEADER, OVERHEAD, pack, unpack

SOURCE = ROOT / "results/experiments/feasibility_continuous_probe_20261118"
OUTPUT = ROOT / "results/experiments/feasibility_installation_envelope_20261118"
MODES = ("marginal", "temporal", "spatiotemporal")
RATIOS = (.0001, .0003, .001, .003, .01, .03, .1, .3, 1.)
SOURCES = (
    "compression/feasibility_installation_envelope.py",
    "finetuning/audit_feasibility_installation_envelope.py",
    "tests/test_feasibility_installation_envelope.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha(data)}


def verify(path, expected):
    if identity(path) != expected:
        raise ValueError(f"Frozen input changed: {path}")


def freeze(source=SOURCE, output=OUTPUT, report=None):
    """Freeze policy and all input bytes before any compression measurement."""
    source, output = Path(source), Path(output)
    if source.resolve() != SOURCE.resolve():
        raise ValueError("Only the frozen D2c source is authorized")
    reports = sorted((ROOT / "progress_reports").glob("118-*.md"))
    if report is None and len(reports) == 1:
        report = reports[0]
    if len(reports) > 1 and report is None:
        raise ValueError("Ambiguous active report118")
    inventory = {p.name: identity(p) for p in sorted(source.glob("*.zip"))}
    if len(inventory) != 729:
        raise ValueError("Expected all 729 frozen D2c ZIPs")
    metadata = {name: identity(source / name) for name in (
        "results.json", "completion_evidence.json", "freeze.json", "plan.json", "protocol.json")}
    protocol = {
        "kind": "feasibility-installation-envelope-frozen-v1",
        "source": str(source.relative_to(ROOT)),
        "codec": "stdlib lzma FORMAT_XZ LZMA2 preset=6 CHECK_CRC64; one attempt",
        "fallback": "raw when XZ payload is not strictly smaller",
        "exact_input": "complete original installation ZIP bytes, without opening or rewriting ZIP",
        "envelope_overhead_bytes": OVERHEAD,
        "ratios": list(RATIOS), "primary_ratios": [.03, .1], "modes": list(MODES),
        "platform": "Python 3 standard library including lzma/liblzma preinstalled; interpreter, stdlib and OS excluded",
        "decoder_charge": "complete new codec source bytes, uncompressed, once per mode/ratio shared deployment",
        "isolated_charge": "also report each configuration deployed alone, source once each",
        "guard107": {"minimum_saving_fraction": .1, "minimum_saving_bpc": .05,
                     "maximum_configuration_regression": .01,
                     "positive_families": 4, "configurations": 9},
        "guard_scope": "shared deployment aggregate; per-configuration regressions from isolated deployments; additionally publish isolated aggregate guards",
        "transport": "unchanged exact ZIP bytes and ledger; no entropy decoding",
        "objects": "historical objects integrity only; no new self-contained-object claim",
        "no_posthoc_adjustment": True, "waveform_cells_read": 0, "gpu_used": False,
        "remote_used": False, "new_training_updates": 0, "f4_authorized": False,
        "generalist_gate": None, "absent_configurations": ["multicore/C1 and C4"],
        "report118": ({"path": str(Path(report).resolve().relative_to(ROOT)),
                       **identity(Path(report))}
                      if report else {"status": "not present at freeze; archived 118 is not substituted"}),
        "guard107_document": identity(ROOT / "progress_reports/107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md"),
        "inputs": metadata, "zip_inventory": inventory,
        "source_files": {name: identity(ROOT / name) for name in SOURCES},
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "freeze.json", identity(output / "protocol.json"))
    return protocol


def guards(rows, summaries):
    outcomes = []
    for ratio in (.03, .1):
        for mode in MODES[1:]:
            selected = [r for r in rows if r["ratio"] == ratio and r["mode"] == mode]
            for deployment in ("shared_once", "isolated"):
                base = next(s for s in summaries if s["ratio"] == ratio and s["mode"] == "marginal")
                candidate = next(s for s in summaries if s["ratio"] == ratio and s["mode"] == mode)
                key = deployment + "_total_bytes"
                saving_bytes = base[key] - candidate[key]
                saving_fraction = saving_bytes / base[key]
                saving_bpc = 8 * saving_bytes / candidate["cells"]
                family = {f: sum(r["saving_vs_new_marginal_bytes"] for r in selected
                                 if r["family"] == f)
                          for f in sorted({r["family"] for r in selected})}
                regressions = {r["group_id"]: -r["saving_vs_new_marginal_bytes"]
                               / r["new_marginal_total_bytes"] for r in selected}
                passed = (len(selected) == 9 and saving_fraction >= .1 and saving_bpc >= .05
                          and len(family) == 4 and all(v > 0 for v in family.values())
                          and all(v <= .01 for v in regressions.values()))
                outcomes.append({
                    "ratio": ratio, "mode": mode, "deployment": deployment,
                    "marginal_bytes": base[key], "candidate_bytes": candidate[key],
                    "cells": candidate["cells"], "saving_bytes": saving_bytes,
                    "saving_fraction": saving_fraction, "saving_bpc": saving_bpc,
                    "family_saving_bytes": family,
                    "configuration_regression_fractions": regressions,
                    "measured_coverage_thresholds_pass": passed,
                    "generalist_gate": None, "f4_authorized": False,
                })
    return outcomes


def run(source=SOURCE, output=OUTPUT):
    source, output = Path(source), Path(output)
    if source.resolve() != SOURCE.resolve():
        raise ValueError("Only frozen D2c source is authorized")
    if (output / "results.json").exists():
        raise FileExistsError("Single contrast already completed; no rerun authorized")
    started = time.perf_counter()
    verify(output / "protocol.json", json.loads((output / "freeze.json").read_text()))
    protocol = json.loads((output / "protocol.json").read_text())
    for name, expected in protocol["inputs"].items():
        verify(source / name, expected)
    for name, expected in protocol["source_files"].items():
        verify(ROOT / name, expected)
    verify(ROOT / "progress_reports/107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md",
           protocol["guard107_document"])
    if "path" in protocol["report118"]:
        verify(ROOT / protocol["report118"]["path"],
               {k: protocol["report118"][k] for k in ("bytes", "sha256")})
    elif list((ROOT / "progress_reports").glob("118-*.md")):
        raise ValueError("Report118 appeared after freeze; obtain a fresh pre-execution freeze")
    for name, expected in protocol["zip_inventory"].items():
        verify(source / name, expected)
    historical = json.loads((source / "results.json").read_text())
    evidence = json.loads((source / "completion_evidence.json").read_text())
    if (evidence.get("complete") is not True
            or evidence.get("all_q_waveform_portable_roundtrips") is not True
            or evidence.get("bound_violations") != 0
            or evidence.get("result_sha256") != protocol["inputs"]["results.json"]["sha256"]):
        raise ValueError("D2c completion or result integrity is not established")
    evidence_by_name = {Path(a["path"]).name: {k: a[k] for k in ("bytes", "sha256")}
                        for a in evidence["artifacts"]}
    if evidence_by_name != protocol["zip_inventory"]:
        raise ValueError("D2c completion inventory differs from exact ZIP bytes")
    configurations = historical["configurations"]
    groups = sorted({c["group_id"] for c in configurations})
    if (len(groups) != 9 or len(configurations) != 81
            or {c["ratio"] for c in configurations} != set(RATIOS)
            or len({(c["group_id"], c["ratio"]) for c in configurations}) != 81):
        raise ValueError("Incomplete or duplicate frozen D2c panel")
    decoder_source = (ROOT / SOURCES[0]).read_bytes()
    decoder_path = output / "installation_decoder.py"
    with decoder_path.open("xb") as stream:
        stream.write(decoder_source)
    spec = importlib.util.spec_from_file_location("paid_installation_decoder", decoder_path)
    installed_decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installed_decoder)
    rows = []
    for config in configurations:
        prefix = f"group{groups.index(config['group_id']):02d}_{config['ratio']:g}"
        for mode in MODES:
            old = config["arms"][mode]
            for kind in ("installation", "transport", "objects"):
                name = f"{prefix}_{mode}.{kind}.zip"
                expected = {"bytes": old[kind + "_bytes"], "sha256": old[kind + "_sha256"]}
                if protocol["zip_inventory"][name] != expected:
                    raise ValueError("Frozen per-arm ledger mismatch")
            if old["installed_plus_transport_bytes"] != old["installation_bytes"] + old["transport_bytes"]:
                raise ValueError("Original installed-plus-transport ledger mismatch")
            name = f"{prefix}_{mode}.installation.zip"
            original = (source / name).read_bytes()
            before = time.perf_counter()
            envelope = pack(original)
            encode_seconds = time.perf_counter() - before
            path = output / (name + ".fie")
            with path.open("xb") as stream:
                stream.write(envelope)
            before = time.perf_counter()
            restored = installed_decoder.unpack(path.read_bytes())
            decode_seconds = time.perf_counter() - before
            if restored != original or unpack(envelope) != original:
                raise ValueError("Installation ZIP roundtrip changed original bytes")
            magic, flag, length, stored, digest = HEADER.unpack_from(envelope)
            new_total = old["transport_bytes"] + len(envelope) + len(decoder_source)
            row = {
                "group_id": config["group_id"], "family": config["family"],
                "ratio": config["ratio"], "mode": mode, "cells": old["cells"],
                "original_installation_path": name,
                "original_installation_bytes": len(original),
                "original_installation_sha256": sha(original),
                "restored_installation_sha256": sha(restored),
                "transport_sha256": old["transport_sha256"],
                "transport_bytes": old["transport_bytes"], "transport_unchanged": True,
                "envelope_path": path.name, "envelope_bytes": len(envelope),
                "envelope_sha256": sha(envelope), "payload_bytes": stored,
                "envelope_overhead_bytes": OVERHEAD, "flag": flag,
                "branch": "xz" if flag else "raw", "exact_zip_roundtrip": True,
                "incremental_decoder_source_bytes": len(decoder_source),
                "original_total_bytes": old["installed_plus_transport_bytes"],
                "new_isolated_total_bytes": new_total,
                "saving_vs_original_own_bytes": old["installed_plus_transport_bytes"] - new_total,
                "new_isolated_bpc": 8 * new_total / old["cells"],
                "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
            }
            if length != len(original) or len(envelope) != stored + OVERHEAD:
                raise ValueError("Envelope ledger mismatch")
            rows.append(row)
    for row in rows:
        marginal = next(r for r in rows if r["group_id"] == row["group_id"]
                        and r["ratio"] == row["ratio"] and r["mode"] == "marginal")
        row["new_marginal_total_bytes"] = marginal["new_isolated_total_bytes"]
        row["saving_vs_new_marginal_bytes"] = marginal["new_isolated_total_bytes"] - row["new_isolated_total_bytes"]
    summaries = []
    for ratio in RATIOS:
        for mode in MODES:
            selected = [r for r in rows if r["ratio"] == ratio and r["mode"] == mode]
            old = sum(r["original_total_bytes"] for r in selected)
            shared = sum(r["transport_bytes"] + r["envelope_bytes"] for r in selected) + len(decoder_source)
            isolated = sum(r["new_isolated_total_bytes"] for r in selected)
            cells = sum(r["cells"] for r in selected)
            summaries.append({
                "ratio": ratio, "mode": mode, "cells": cells,
                "original_total_bytes": old, "shared_once_total_bytes": shared,
                "isolated_total_bytes": isolated,
                "shared_once_bpc": 8 * shared / cells, "isolated_bpc": 8 * isolated / cells,
                "shared_once_saving_vs_original_own_bytes": old - shared,
                "isolated_saving_vs_original_own_bytes": old - isolated,
                "original_installation_bytes": sum(r["original_installation_bytes"] for r in selected),
                "envelope_bytes": sum(r["envelope_bytes"] for r in selected),
                "transport_bytes": sum(r["transport_bytes"] for r in selected),
                "shared_decoder_bytes": len(decoder_source),
                "isolated_decoder_bytes": len(decoder_source) * len(selected),
                "xz_count": sum(r["flag"] == 1 for r in selected),
                "raw_count": sum(r["flag"] == 0 for r in selected),
            })
    for summary in summaries:
        marginal = next(s for s in summaries if s["ratio"] == summary["ratio"] and s["mode"] == "marginal")
        for contract in ("shared_once", "isolated"):
            saving = marginal[contract + "_total_bytes"] - summary[contract + "_total_bytes"]
            summary[contract + "_saving_vs_new_marginal_bytes"] = saving
            summary[contract + "_saving_fraction_vs_new_marginal"] = saving / marginal[contract + "_total_bytes"]
            summary[contract + "_saving_bpc_vs_new_marginal"] = 8 * saving / summary["cells"]
    # Recheck all original bytes and implementation after the experiment.
    for name, expected in protocol["zip_inventory"].items():
        verify(source / name, expected)
    for name, expected in protocol["source_files"].items():
        verify(ROOT / name, expected)
    for name, expected in protocol["inputs"].items():
        verify(source / name, expected)
    if "path" in protocol["report118"]:
        verify(ROOT / protocol["report118"]["path"],
               {k: protocol["report118"][k] for k in ("bytes", "sha256")})
    primary_guards = guards(rows, summaries)
    result = {
        "kind": "feasibility-installation-envelope-results-v1",
        "protocol_sha256": sha((output / "protocol.json").read_bytes()),
        "source_results_sha256": protocol["inputs"]["results.json"]["sha256"],
        "decoder_source": {"path": decoder_path.name, **identity(decoder_path)},
        "platform_contract": protocol["platform"],
        "execution_platform": {"python": sys.version, "platform": platform.platform()},
        "zip_artifacts_verified_before_and_after": len(protocol["zip_inventory"]),
        "installation_roundtrips": len(rows), "transport_ledger_checks": len(rows),
        "source_zip_mutations": 0, "waveform_cells_read": 0,
        "entropy_symbols_decoded": 0, "gpu_used": False, "remote_used": False,
        "new_training_updates": 0, "posthoc_adjustments": 0,
        "generalist_gate": None, "f4_authorized": False,
        "absent_configurations": protocol["absent_configurations"],
        "primary_guards": primary_guards, "summaries": summaries, "rows": rows,
        "joint_primary_measured_coverage_pass": {
            mode: {deployment: all(g["measured_coverage_thresholds_pass"] for g in primary_guards
                                   if g["mode"] == mode and g["deployment"] == deployment)
                   for deployment in ("shared_once", "isolated")}
            for mode in MODES[1:]},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "limits": ["Artifact-specific installation repackaging only; not an entropy-model improvement.",
                   "No waveform decoded or fidelity remeasured; exact installation ZIP and transport bytes preserve D2c.",
                   "No independent-session, full-configuration, generalist or F4 pass.",
                   "Preinstalled stdlib/liblzma is a declared platform dependency, not a standalone native decoder."],
    }
    write_json(output / "results.json", result)
    print(json.dumps({"roundtrips": len(rows), "decoder_bytes": len(decoder_source),
                      "elapsed_seconds": result["elapsed_seconds"],
                      "primaries": [s for s in summaries if s["ratio"] in (.03, .1)],
                      "guards": result["primary_guards"]}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report118", type=Path)
    args = parser.parse_args()
    if args.freeze == args.execute:
        parser.error("Choose exactly one of --freeze and --execute")
    if args.freeze:
        freeze(report=args.report118)
    else:
        run()


if __name__ == "__main__":
    main()
