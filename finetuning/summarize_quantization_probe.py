#!/usr/bin/env python3
"""Pair frozen diagnostic windows against completed VQ reference outputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

try:
    from .audit_rate_continuation import digest, keyed, read_rows, write_csv
except ImportError:
    from audit_rate_continuation import digest, keyed, read_rows, write_csv


METRICS = ("relative_rmse", "correlation", "ordinary_channel_p90_relative_rmse",
           "reconstruction_rms_ratio", "fft_phase_alignment")


def candidate(root, label, hashes):
    summary_path = root / f"{label}.json"
    summary = json.loads(summary_path.read_text())
    path = Path(summary["windows_csv"])
    if digest(path) != summary["windows_sha256"]:
        raise ValueError("Candidate window checksum mismatch")
    hashes[str(summary_path)] = digest(summary_path)
    hashes[str(path)] = digest(path)
    return keyed(read_rows(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    reference = (args.reference_dir or root).resolve()
    if not all((p / "COMPLETE").is_file() for p in (root, reference)):
        raise ValueError("Completed control and reference required")
    protocols = [json.loads((p / "protocol.json").read_text()) for p in (root, reference)]
    if any(protocols[0][k] != protocols[1][k] for k in ("sources", "validation_hash", "normalization")):
        raise ValueError("Control/reference protocol mismatch")
    hashes = {str(p / "protocol.json"): digest(p / "protocol.json") for p in (root, reference)}
    hashes[str(Path(__file__))] = digest(Path(__file__))
    paired, grouped = [], defaultdict(list)
    summaries = read_rows(root / "summary.csv")
    for summary in summaries:
        if summary["mode"] == "vq":
            continue
        label = summary["label"]
        source = label.split("_", 1)[0]
        before = candidate(reference, f"{source}_vq", hashes)
        after = candidate(root, label, hashes)
        if not before or before.keys() != after.keys():
            raise ValueError("Control/reference coordinates differ")
        for path in sorted(before):
            a, b = before[path], after[path]
            invariants = ("family", "configuration", "drive_id", "input_cells", "side_bytes",
                          "escaped_cell_fraction", "relative_quantization_error", "latent_positions", "latent_dimensions")
            if any(a[k] != b[k] for k in invariants):
                raise ValueError("Control/reference provenance or quantization differs")
            row = {"label": label, "sample_path": path, "drive_id": a["drive_id"], "family": a["family"],
                   "configuration": a["configuration"], "escaped": int(float(a["escaped_cell_fraction"]) > 0),
                   "relative_quantization_error": float(a["relative_quantization_error"])}
            for metric in METRICS:
                x, y = float(a[metric]), float(b[metric])
                if not np.isfinite([x, y]).all():
                    raise ValueError("Nonfinite paired reconstruction metrics")
                row.update({f"vq_{metric}": x, f"control_{metric}": y, f"delta_{metric}": y - x})
            paired.append(row)
            for stratum in ("all", "escaped" if row["escaped"] else "escape_free"):
                grouped[(label, a["family"], a["configuration"], stratum)].append(row)
    results = []
    for (label, family, configuration, stratum), rows in sorted(grouped.items()):
        result = {"label": label, "family": family, "configuration": configuration, "stratum": stratum,
                  "windows": len(rows), "files": len({r["drive_id"] for r in rows}),
                  "rmse_regression_fraction": float(np.mean([r["delta_relative_rmse"] > 0 for r in rows])),
                  "mean_relative_quantization_error": float(np.mean([r["relative_quantization_error"] for r in rows]))}
        for metric in METRICS:
            for state in ("vq", "control", "delta"):
                result[f"{state}_{metric}"] = float(np.mean([r[f"{state}_{metric}"] for r in rows]))
        results.append(result)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "paired.windows.csv", paired)
    write_csv(output / "groups.csv", results)
    (output / "protocol.json").write_text(json.dumps({"kind": "paired_quantization_probe_summary_v1",
        "inputs_sha256": hashes, "training_updates": 0,
        "warning": "Descriptive reused validation, not independent acquisition evidence or rate-matched codec selection."}, indent=2) + "\n")
    (output / "COMPLETE").write_text("Paired frozen diagnostic summary complete; no inference or training.\n")
    print(f"{len(paired)} paired rows, {len(results)} group/escape strata; {output}")


if __name__ == "__main__":
    main()
