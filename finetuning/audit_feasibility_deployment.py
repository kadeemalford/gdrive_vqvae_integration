"""Translate frozen rate/error measurements into deployment screening quantities."""
import argparse
import hashlib
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path


def compare(marginal, contextual):
    """Keep transport and fixed installation separate; never infer a future rate."""
    cells = marginal["cells"]
    if cells <= 0 or cells != contextual["cells"]:
        raise ValueError("Expected the same positive coverage")
    for record in (marginal, contextual):
        if record["total_bytes"] != record["installation_bytes"] + record["transport_bytes"]:
            raise ValueError("Byte ledger does not reconcile")
    target = .9 * marginal["total_bytes"]
    transport = contextual["transport_bytes"]
    return {
        "cells": cells,
        "marginal_total_bpc": 8 * marginal["total_bytes"] / cells,
        "contextual_total_bpc": 8 * contextual["total_bytes"] / cells,
        "contextual_float32_ratio": 4 * cells / contextual["total_bytes"],
        "measured_net_saving_fraction": 1 - contextual["total_bytes"] / marginal["total_bytes"],
        "transport_only_saving_fraction": 1 - transport / marginal["transport_bytes"],
        "saving_fraction_with_contextual_installation_free": 1 - transport / marginal["total_bytes"],
        "additional_bytes_needed_for_10pct_at_measured_installation": max(
            0, contextual["total_bytes"] - int(target)),
        "additional_transport_bytes_needed_for_10pct_even_with_free_installation": max(
            0, transport - int(target)),
    }


def audit(data):
    if data["kind"] != "feasibility-npz-transfer-results-v1" or data["fit_calls"] != 0:
        raise ValueError("Expected frozen transfer evidence")
    groups = defaultdict(list)
    identities = {}
    for row in data["rows"]:
        key = row["ratio"], row["sample_path"]
        identity = tuple(row[field] for field in (
            "cells", "finite_cells", "epsilon", "q_sha256", "waveform_sha256", "sse"))
        if (row["bound_violations"] or not row["q_roundtrip"] or not row["portable_roundtrip"]
                or not row["waveform_bit_exact"] or identities.setdefault(key, identity) != identity):
            raise ValueError("Non-identical or invalid reconstruction")
        if row["mode"] == "marginal":
            groups[row["ratio"], row["group_id"]].append(row)
    output = []
    at_horizon = {(row["ratio"], row["mode"]): row for row in data["aggregates"]
                  if row["horizon"] == 20}
    for ratio in (.03, .1):
        marginal, contextual = (at_horizon[ratio, mode] for mode in ("marginal", "spatiotemporal"))
        physical = {}
        for (precision, group), rows in sorted(groups.items()):
            if precision != ratio:
                continue
            finite = sum(row["finite_cells"] for row in rows)
            eps, = {row["epsilon"] for row in rows}
            physical[group] = {
                "windows": len(rows), "finite_cells": finite,
                "epsilon_native": eps,
                "observed_max_abs_error_native": max(row["max_finite_error"] for row in rows),
                "observed_rmse_native": sqrt(sum(row["sse"] for row in rows) / finite) if finite else None,
            }
        if sum(v["windows"] for v in physical.values()) != contextual["windows"]:
            raise ValueError("Physical statistics do not cover the transfer panel")
        output.append({"epsilon_over_sigma_cal": ratio, **compare(marginal, contextual),
                       "physical_by_configuration": physical})
    return {"kind": "febus-feasibility-deployment-audit-v1", "points": output,
            "horizon": "20 prepared windows per file; installation once per group",
            "operational_tolerance_validated": False, "new_waveform_reads": 0,
            "new_parameter_updates": 0, "new_gate": False,
            "phase_amplitude_tail_require_application_tolerances": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = audit(json.loads(raw))
    result["source"] = str(args.input)
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != text:
            raise ValueError("Refusing to overwrite different evidence")
    else:
        with args.output.open("x") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
