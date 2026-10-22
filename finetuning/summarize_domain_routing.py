#!/usr/bin/env python3
"""Join completed candidate predictions by preset metadata route; charge route bytes."""
import argparse
import csv
import json
from pathlib import Path

try:
    from .domain_routing import FAMILIES, route_label
    from .benchmark_matched_temporal_rates import completed_windows
    from .reconstruction_objective import aggregate_validation, path_fingerprint
    from .run_sampling_experiment import ROOT
    from .analyze_sampling_results import write_csv
except ImportError:
    from domain_routing import FAMILIES, route_label
    from benchmark_matched_temporal_rates import completed_windows
    from reconstruction_objective import aggregate_validation, path_fingerprint
    from run_sampling_experiment import ROOT
    from analyze_sampling_results import write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir
    audit = root / "full_validation"
    protocol = json.loads((audit / "protocol.json").read_text())
    paths = sorted((ROOT / "data/finetune/fold_1/val").glob("*.npz"))
    if protocol["split"] != "val" or protocol["validation_hash"] != path_fingerprint(paths):
        raise ValueError("Canonical validation required")
    expected = {str(p.resolve()) for p in paths}
    candidates = {c["label"]: completed_windows(audit, c["label"], expected)[0] for c in protocol["candidates"]}
    if not {"shared", "balanced", "frozen_generalist", *(f"expert_{f}" for f in FAMILIES)}.issubset(candidates):
        raise ValueError("All experts and generalist references required")
    rows, summaries, groups = [], [], []
    # Metric field lists are taken from scored windows, excluding provenance/rates.
    excluded = {"sample_path", "drive_id", "title", "input_cells", "fixed_index_bits", "side_bytes", "family", "configuration"}
    for policy in ("frozen_generalist", "shared", "balanced", "metadata_routed"):
        values = []
        cells = bits = side = route_bytes = 0
        for path in sorted(expected):
            reference = candidates["shared"][path]
            label = route_label(reference["family"]) if policy == "metadata_routed" else policy
            row = candidates[label][path]
            if (row["family"], row["configuration"], row["input_cells"]) != (reference["family"], reference["configuration"], reference["input_cells"]):
                raise ValueError("Candidate metadata mismatch")
            numeric = {key: float(value) for key, value in row.items() if key not in excluded}
            values.append({"family": row["family"], "configuration": row["configuration"], **numeric})
            routing = 1 if policy == "metadata_routed" else 0
            rows.append({"policy": policy, "route": label, "route_bytes": routing, **row})
            cells += int(row["input_cells"])
            bits += int(row["fixed_index_bits"])
            side += int(row["side_bytes"])
            route_bytes += routing
        macro, grouped = aggregate_validation(values)
        weights = list({c["sha256"] for c in protocol["candidates"]
                        if c["label"] == policy or (policy == "metadata_routed" and (c["label"] == "shared" or c["label"].startswith("expert_")))})
        model_bytes = sum(Path(c["checkpoint"]).stat().st_size for c in protocol["candidates"]
                          if c["label"] == policy or (policy == "metadata_routed" and (c["label"] == "shared" or c["label"].startswith("expert_"))))
        parameter_bytes = sum(4 * int(json.loads((audit / f"{c['label']}.json").read_text())["model_parameters"])
                              for c in protocol["candidates"]
                              if c["label"] == policy or (policy == "metadata_routed" and (c["label"] == "shared" or c["label"].startswith("expert_"))))
        summaries.append({"policy": policy, **macro, "input_cells": cells, "fixed_index_bits": bits,
                          "side_bytes": side, "route_bytes": route_bytes,
                          "pooled_index_side_route_ratio": 32 * cells / (bits + 8 * (side + route_bytes)),
                          "distinct_checkpoint_hashes": len(weights), "training_checkpoint_registry_bytes": model_bytes,
                          "float32_parameter_bytes_for_registry": parameter_bytes})
        groups += [{"policy": policy, **row} for row in grouped]
    write_csv(root / "routed.windows.csv", rows)
    write_csv(root / "routed.groups.csv", groups)
    write_csv(root / "routed.summary.csv", summaries)
    (root / "routing.json").write_text(json.dumps({"routes": {f:route_label(f) for f in FAMILIES}, "fallback": "shared",
        "route_bytes_per_window": 1, "warning": "Development validation with preset metadata routes, not error oracle or external audit. Nominal indices+side+route; registry bytes include training optimizer/history metadata, not deployed decoder size. New/absent-family fallback behavior implemented but unknown-domain fidelity unmeasured."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
