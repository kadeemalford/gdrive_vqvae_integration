#!/usr/bin/env python3
"""Pair two completed physical-evaluator exports without new inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

try:
    from .audit_rate_continuation import digest, read_rows, pair_rows, describe, file_bootstrap, write_csv
except ImportError:
    from audit_rate_continuation import digest, read_rows, pair_rows, describe, file_bootstrap, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-windows", type=Path, required=True)
    parser.add_argument("--after-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.before_windows.resolve(), args.after_windows.resolve()]
    protocols = []
    hashes = {str(p): digest(p) for p in inputs}
    for path in inputs:
        if not (path.parent / "COMPLETE").is_file():
            raise ValueError("Completed physical validation exports required")
        p = path.parent / "protocol.json"
        protocols.append(json.loads(p.read_text()))
        hashes[str(p)] = digest(p)
    for protocol in protocols:
        if protocol["kind"] != "selected_candidates_full_physical_validation_v1" or protocol["split"] != "val":
            raise ValueError("Expected validation-only physical evaluator protocol")
    for key in ("fold", "validation_hash", "windows"):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError("Paired dataset/protocol mismatch")
    if protocols[0].get("score_reference", "escape-v1") != protocols[1].get("score_reference", "escape-v1"):
        raise ValueError("Paired score references differ")
    data = [read_rows(p) for p in inputs]
    if any(len(rows) != protocols[0]["windows"] for rows in data):
        raise ValueError("Incomplete export attempt")
    paired = pair_rows(*data)
    groups, files = defaultdict(list), defaultdict(list)
    for row in paired:
        key = row["family"], row["configuration"]
        groups[key].append(row)
        files[(*key, row["drive_id"])].append(row)
    summaries = [{"family": family, "configuration": configuration, **describe(rows),
                  **file_bootstrap(rows, 5000, 2026)}
                 for (family, configuration), rows in sorted(groups.items())]
    file_rows = [{"family": f, "configuration": c, "drive_id": d, **describe(rows)}
                 for (f, c, d), rows in sorted(files.items())]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("windows.csv", paired), ("configurations.csv", summaries), ("files.csv", file_rows)):
        write_csv(output / name, rows)
    hashes[str(Path(__file__))] = digest(Path(__file__))
    hashes[str(Path(__file__).with_name("audit_rate_continuation.py"))] = digest(Path(__file__).with_name("audit_rate_continuation.py"))
    (output / "protocol.json").write_text(json.dumps({"kind": "paired_physical_export_audit_v1",
        "inputs_sha256": hashes, "training_updates": 0, "bootstrap_repetitions": 5000, "seed": 2026,
        "warning": "Exploratory reused validation. File-cluster intervals are descriptive, not independent-acquisition inference. No acceptance tolerance or automatic promotion."}, indent=2) + "\n")
    (output / "COMPLETE").write_text("Paired physical-export audit complete; no inference/training/test access.\n")
    print(f"{len(paired)} paired windows, {len(groups)} configurations; {output}")


if __name__ == "__main__":
    main()
