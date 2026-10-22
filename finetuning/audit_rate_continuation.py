#!/usr/bin/env python3
"""Paired, validation-only audit of frozen before/after rate candidates."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("relative_rmse", "correlation", "ordinary_channel_p90_relative_rmse",
           "ordinary_nonweak_channel_macro_relative_rmse", "reconstruction_rms_ratio",
           "fft_phase_alignment", "high_nyquist_power_ratio", "ordinary_raw_rmse")
INVARIANTS = ("drive_id", "family", "configuration", "input_cells", "fixed_index_bits", "side_bytes",
              "escaped_cell_fraction", "diagnostic_channel_std_floor", "variable_ordinary_channels")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def keyed(rows):
    result = {str(Path(r["sample_path"]).resolve()): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate sample coordinates")
    return result


def pair_rows(before, after):
    before, after = keyed(before), keyed(after)
    if not before or before.keys() != after.keys():
        raise ValueError("Paired coordinates differ or are empty")
    pairs = []
    for path in sorted(before):
        a, b = before[path], after[path]
        if any(a[k] != b[k] for k in INVARIANTS):
            raise ValueError("Paired reference/rate/metadata mismatch")
        row = {"sample_path": path, **{k: a[k] for k in INVARIANTS}}
        for metric in METRICS:
            x, y = float(a[metric]), float(b[metric])
            if not np.isfinite([x, y]).all():
                raise ValueError("Nonfinite paired metric")
            row.update({f"before_{metric}": x, f"after_{metric}": y, f"delta_{metric}": y - x})
        pairs.append(row)
    return pairs


def describe(rows):
    delta = np.array([r["delta_relative_rmse"] for r in rows])
    result = {"windows": len(rows), "files": len({r["drive_id"] for r in rows}),
              "window_rmse_regression_fraction": float(np.mean(delta > 1e-12)),
              "window_delta_rmse_p10": float(np.quantile(delta, .1)),
              "window_delta_rmse_p50": float(np.quantile(delta, .5)),
              "window_delta_rmse_p90": float(np.quantile(delta, .9))}
    for metric in METRICS:
        for state in ("before", "after", "delta"):
            result[f"{state}_{metric}"] = float(np.mean([r[f"{state}_{metric}"] for r in rows]))
    return result


def file_bootstrap(rows, repetitions, seed):
    """Resample file clusters, retaining window weights within the configuration."""
    files = defaultdict(list)
    for row in rows:
        files[row["drive_id"]].append(row["delta_relative_rmse"])
    if len(files) < 2:
        return {"bootstrap_available": False, "file_bootstrap_delta_rmse_low": None,
                "file_bootstrap_delta_rmse_high": None}
    totals = np.array([(sum(values), len(values)) for _, values in sorted(files.items())])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(totals), size=(repetitions, len(totals)))
    sampled = totals[draws].sum(axis=1)
    low, high = np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975])
    return {"bootstrap_available": True, "file_bootstrap_delta_rmse_low": float(low),
            "file_bootstrap_delta_rmse_high": float(high)}


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions required")
    full = args.experiment_dir.resolve() / "full_validation"
    if not (args.experiment_dir / "COMPLETE").is_file() or not (full / "COMPLETE").is_file():
        raise ValueError("Complete experiment and full validation required")
    protocol = json.loads((full / "protocol.json").read_text())
    if protocol["fold"] != 0 or protocol["split"] != "val" or not protocol.get("score_reference", "").startswith("robust-escape-v1"):
        raise ValueError("Common robust fold-0 validation reference required")
    manifest = ROOT / "data/finetune/fold_0/samples_manifest.csv"
    metadata = keyed([r for r in read_rows(manifest) if r["split"] == "val"])
    if len(metadata) != protocol["windows"]:
        raise ValueError("Validation manifest size mismatch")
    inputs = {str(p): digest(p) for p in (full / "protocol.json", full / "summary.csv", manifest, Path(__file__))}
    outputs, configurations = {}, []
    for arm in ("2x", "4x"):
        candidates = []
        for label in (f"initial_rate_{arm}", f"continued_rate_{arm}"):
            if not (full / f"{label}.json").is_file():
                raise ValueError("Missing completed candidate summary")
            attempts = sorted(full.glob(f"{label}.windows.attempt_*.csv"))
            complete = [(p, read_rows(p)) for p in attempts]
            complete = [(p, rows) for p, rows in complete if len(rows) == protocol["windows"]]
            if len(complete) != 1:
                raise ValueError("Expected exactly one complete attempt per candidate")
            p, rows = complete[0]
            inputs[str(p)] = digest(p)
            candidates.append(rows)
        paired = pair_rows(*candidates)
        if {r["sample_path"] for r in paired} != metadata.keys():
            raise ValueError("Candidate coordinates differ from validation manifest")
        grouped, files, strata = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in paired:
            meta = metadata[row["sample_path"]]
            if any(meta[k] != row[k] for k in ("family", "configuration", "drive_id")):
                raise ValueError("Manifest/candidate provenance mismatch")
            group = row["family"], row["configuration"]
            grouped[group].append(row)
            files[(*group, row["drive_id"])].append(row)
            strata[(*group, meta["dt_s"], meta["channels"])].append(row)
        summaries = []
        for (family, configuration), rows in sorted(grouped.items()):
            summary = {"arm": arm, "family": family, "configuration": configuration, **describe(rows),
                       **file_bootstrap(rows, args.bootstrap_repetitions, 2026)}
            summaries.append(summary)
            configurations.append(summary)
        outputs[f"{arm}.windows.csv"] = paired
        outputs[f"{arm}.files.csv"] = [{"family": f, "configuration": c, "drive_id": d, **describe(rows)}
                                         for (f, c, d), rows in sorted(files.items())]
        outputs[f"{arm}.strata.csv"] = [{"family": f, "configuration": c, "dt_s": dt, "channels": ch, **describe(rows)}
                                          for (f, c, dt, ch), rows in sorted(strata.items())]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit_protocol = {"kind": "paired_robust_rate_continuation_audit_v1", "training_updates": 0,
                      "seed": 2026, "bootstrap_repetitions": args.bootstrap_repetitions, "inputs_sha256": inputs,
                      "warning": "Exploratory reused validation; file-cluster intervals are descriptive, not independent-acquisition confidence. Copies/correlated files may remain. No test selection or model promotion."}
    (output / "protocol.json").write_text(json.dumps(audit_protocol, indent=2) + "\n")
    write_csv(output / "configurations.csv", configurations)
    for name, rows in outputs.items():
        write_csv(output / name, rows)
    (output / "COMPLETE").write_text("Paired validation-only audit complete; no inference/training/test access.\n")
    for row in configurations:
        print(row["arm"], row["family"], row["configuration"], "delta", round(row["delta_relative_rmse"], 6),
              "regressed_windows", round(row["window_rmse_regression_fraction"], 3))


if __name__ == "__main__":
    main()
