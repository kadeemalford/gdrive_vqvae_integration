#!/usr/bin/env python3
"""Read-only, model-free roundtrip audit on canonical prepared coordinates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from .reversible_normalization import encode, decode
    from .analyze_sampling_results import write_csv
    from .run_capacity_probe import ROOT
except ImportError:
    from reversible_normalization import encode, decode
    from analyze_sampling_results import write_csv
    from run_capacity_probe import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    with (ROOT / "data/finetune/fold_0/samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    results = []
    for row in tqdm(rows, desc="[normalization] roundtrip only", unit="window"):
        with np.load(ROOT / row["sample_path"], allow_pickle=False) as payload:
            raw = payload["data"]
        packet = encode(raw)
        restored = decode(packet.values, packet.side_stream)
        bit_exact = raw[~packet.valid].tobytes() == restored[~packet.valid].tobytes()
        if not bit_exact:
            raise AssertionError("Escape is not bit exact")
        untouched = packet.valid.all(axis=1)
        varying = np.ptp(raw[untouched].astype(np.float64), axis=1) > 0
        erased = (np.ptp(packet.values[untouched], axis=1) == 0) & varying
        reference = raw[packet.valid].astype(np.float64)
        error = restored[packet.valid].astype(np.float64) - reference
        relative = float(np.sqrt(np.sum(error ** 2) / max(float(np.sum(reference ** 2)), 1e-60)))
        results.append({"sample_path": row["sample_path"], "family": row["family"], "configuration": row["configuration"],
                        "escaped_cells": int((~packet.valid).sum()), "escape_bit_exact": bit_exact,
                        "ordinary_roundtrip_relative_rmse": relative, "erased_ordinary_channel_instances": int(erased.sum()),
                        "side_stream_bytes": len(packet.side_stream), "cells": raw.size})
    summary = {"kind": "model-free preprocessing roundtrip; not reconstruction quality or total codec rate",
               "windows": len(results), "affected_windows": sum(r["escaped_cells"] > 0 for r in results),
               "escaped_cells": sum(r["escaped_cells"] for r in results), "escape_bit_exact": all(r["escape_bit_exact"] for r in results),
               "max_ordinary_roundtrip_relative_rmse": max(r["ordinary_roundtrip_relative_rmse"] for r in results),
               "erased_ordinary_channel_instances": sum(r["erased_ordinary_channel_instances"] for r in results),
               "side_stream_bytes": sum(r["side_stream_bytes"] for r in results),
               "side_stream_bits_per_input_cell": 8 * sum(r["side_stream_bytes"] for r in results) / sum(r["cells"] for r in results)}
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "windows.csv", results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
