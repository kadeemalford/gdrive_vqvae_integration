#!/usr/bin/env python3
"""Validation-only DCT streams bounded by each neural candidate's exact nominal budget."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from .diagnose_reconstruction import select_validation, spectral_metrics
    from .evaluate_checkpoints import metrics
    from .evaluate_physical_candidates import file_hash
    from .physical_validation import ordinary_fidelity
    from .reconstruction_objective import aggregate_validation, path_fingerprint
    from .reversible_normalization import encode, parse_side_stream
    from .run_sampling_experiment import ROOT, check_splits
    from .sparse_dct_codec import compress, decompress
    from .analyze_sampling_results import write_csv
except ImportError:
    from diagnose_reconstruction import select_validation, spectral_metrics
    from evaluate_checkpoints import metrics
    from evaluate_physical_candidates import file_hash
    from physical_validation import ordinary_fidelity
    from reconstruction_objective import aggregate_validation, path_fingerprint
    from reversible_normalization import encode, parse_side_stream
    from run_sampling_experiment import ROOT, check_splits
    from sparse_dct_codec import compress, decompress
    from analyze_sampling_results import write_csv

KEYS = ("relative_rmse", "correlation", "reconstruction_rms_ratio", "fft_magnitude_relative_error",
        "fft_phase_alignment", "ordinary_channel_median_relative_rmse", "ordinary_channel_p90_relative_rmse",
        "ordinary_nonweak_channel_macro_relative_rmse", "ordinary_raw_rmse")


def nominal_budget(row):
    """Round only the fixed-index bitstream to whole bytes, once per window."""
    bits, side = int(row["fixed_index_bits"]), int(row["side_bytes"])
    if bits < 0 or side < 0:
        raise ValueError("Negative rate components")
    return math.ceil(bits / 8) + side


def completed_windows(root, label, expected_paths):
    if not (root / "COMPLETE").is_file():
        raise ValueError("Completed detailed evaluation required")
    summary = json.loads((root / f"{label}.json").read_text())
    for path in sorted(root.glob(f"{label}.windows.attempt_*.csv"), reverse=True):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        keyed = {str(Path(row["sample_path"]).resolve()): row for row in rows}
        if len(rows) == len(keyed) == len(expected_paths) and set(keyed) == expected_paths:
            if sum(int(row["input_cells"]) for row in rows) != summary["input_cells"]:
                raise ValueError("Window/summary accounting mismatch")
            return keyed, path
    raise ValueError(f"No complete, unique window table for {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve existing evidence; use a new output directory")
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    manifest = samples / "samples_manifest.csv"
    selected = select_validation(manifest, 2, 2026)
    if {row["family"] for row in selected} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Four-family validation required")
    paths = sorted((samples / "val").glob("*.npz"))
    expected = {str(path.resolve()) for path in paths}
    root = args.evaluation_dir.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["split"] != "val" or protocol["validation_hash"] != path_fingerprint(paths):
        raise ValueError("Evaluation does not match canonical validation")
    references, provenance = {}, []
    for candidate in protocol["candidates"]:
        label = candidate["label"]
        if candidate["policy"] != "escape-v1":
            raise ValueError("This paired control requires global escape-v1")
        references[label], table = completed_windows(root, label, expected)
        provenance.append({"label": label, "windows": str(table), "sha256": file_hash(table)})
    rows = []
    for meta in tqdm(selected, desc="[matched rates] DCT, validation only", unit="window"):
        path = (ROOT / meta["sample_path"]).resolve()
        with np.load(path, allow_pickle=False) as payload:
            raw = np.asarray(payload["data"], dtype=np.float32)
        packet = encode(raw)
        _, center, scale, _, _ = parse_side_stream(packet.side_stream)
        for label, reference in references.items():
            neural = reference[str(path)]
            if int(neural["input_cells"]) != raw.size or int(neural["side_bytes"]) != len(packet.side_stream):
                raise ValueError("Raw/side-stream reference mismatch")
            budget = nominal_budget(neural)
            stream, _ = compress(raw, budget, quant_bits=8)
            restored = decompress(stream)
            if restored[~packet.valid].tobytes() != raw[~packet.valid].tobytes():
                raise AssertionError("Escape bits changed")
            normalized = ((restored.astype(np.float64) - center) / scale).astype(np.float32)
            normalized[~packet.valid] = 0
            fidelity = {**metrics(packet.values[packet.valid], normalized[packet.valid]),
                        **spectral_metrics(packet.values, normalized),
                        **ordinary_fidelity(raw, restored, packet.valid)}
            base = {"sample_path": str(path), "drive_id": meta["drive_id"], "family": meta["family"],
                    "configuration": meta["configuration"], "input_cells": raw.size, "budget_bytes": budget}
            for policy, values, size in ((f"neural_{label}", neural, budget), (f"dct_at_{label}", fidelity, len(stream))):
                rows.append({**base, "policy": policy, "encoded_or_nominal_bytes": size,
                             **{key: float(values[key]) for key in KEYS}})
    summaries, groups = [], []
    for policy in sorted({row["policy"] for row in rows}):
        subset = [row for row in rows if row["policy"] == policy]
        macro, grouped = aggregate_validation([{key: row[key] for key in ("family", "configuration", *KEYS)} for row in subset])
        cells = sum(row["input_cells"] for row in subset)
        size = sum(row["encoded_or_nominal_bytes"] for row in subset)
        summaries.append({"policy": policy, **macro, "input_cells": cells, "encoded_or_nominal_bytes": size,
                          "pooled_compression_ratio": 4 * cells / size})
        groups.extend({"policy": policy, **group} for group in grouped)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "windows.csv", rows)
    write_csv(args.output_dir / "groups.csv", groups)
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "protocol.json").write_text(json.dumps({"kind": "matched_temporal_dct_control_v1",
        "split": "val", "seed": 2026, "samples_per_file": 2, "windows": len(selected), "training_updates": 0,
        "manifest_sha256": file_hash(manifest), "references": provenance, "selected": selected,
        "warning": "Paired windows and per-window byte ceilings, not equal deployed rates: DCT actual stream vs VQ fixed indices+side nominal; neither includes external physical maps. Exploratory validation, not independent test."}, indent=2) + "\n")
    (args.output_dir / "COMPLETE").write_text("Matched-budget validation control complete; no training or test access.\n")


if __name__ == "__main__":
    main()
