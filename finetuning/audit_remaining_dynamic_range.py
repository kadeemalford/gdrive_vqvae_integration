#!/usr/bin/env python3
"""CPU-only identity roundtrip audit on fold-0 train/val, no model/test scoring."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

try:
    from .analyze_sampling_results import write_csv
    from .latent_geometry import latent_positions
    from .physical_validation import ordinary_fidelity
    from .reconstruction_objective import aggregate_validation
    from .reversible_normalization import encode, decode, parse_side_stream, HEADER
    from .robust_escape import encode_robust
    from .run_sampling_experiment import ROOT, check_splits
    from .run_temporal_density_long import digest
except ImportError:
    from analyze_sampling_results import write_csv
    from latent_geometry import latent_positions
    from physical_validation import ordinary_fidelity
    from reconstruction_objective import aggregate_validation
    from reversible_normalization import encode, decode, parse_side_stream, HEADER
    from robust_escape import encode_robust
    from run_sampling_experiment import ROOT, check_splits
    from run_temporal_density_long import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    samples = ROOT / f"data/finetune/fold_{args.fold}"
    check_splits(samples, args.fold)
    manifest = samples / "samples_manifest.csv"
    with manifest.open() as stream:
        selected = [r for r in csv.DictReader(stream) if r["split"] in ("train", "val")]
    if any(not (ROOT / r["sample_path"]).is_file() for r in selected):
        raise FileNotFoundError("Complete train/val arrays required")
    args.output_dir.mkdir(parents=True)
    protocol = {"kind": "remaining_dynamic_range_identity_audit_v1", "fold": args.fold, "splits": ["train", "val"],
                "manifest_sha256": digest(manifest), "windows": len(selected), "model_updates": 0,
                "adaptive_threshold": "min(1e20, max(1e-30, 1000*p99(abs(raw)))); if p99=0, positive-only p99; all zero -> 1e20",
                "reference_mask": "robust candidate ordinary mask, identical for BOTH identity roundtrips",
                "code_sha256": {name: digest(ROOT / "finetuning" / name) for name in
                                ("audit_remaining_dynamic_range.py", "robust_escape.py", "reversible_normalization.py", "physical_validation.py")},
                "warning": "Numeric preprocessing diagnostic, not physical invalidity classification, model performance or deployed codec. No test arrays scored. All excluded cells remain in charged bit-exact streams."}
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    model = SimpleNamespace(downsample_levels=4, latent_geometry="temporal-4x")
    with (args.output_dir / "windows.csv").open("w", newline="") as stream:
        writer = None
        for meta in tqdm(selected, desc="[numeric identity audit] train/val only", unit="window"):
            path = (ROOT / meta["sample_path"]).resolve()
            with np.load(path, allow_pickle=False) as payload:
                raw = np.asarray(payload["data"], dtype=np.float32)
            baseline, robust = encode(raw), encode_robust(raw)
            if np.any(robust.valid & ~baseline.valid):
                raise AssertionError("Adaptive ordinary region must be a baseline subset")
            changed = int((baseline.valid & ~robust.valid).sum())
            for policy, packet in (("escape-v1", baseline), ("robust-candidate", robust)):
                identity = decode(packet.values, packet.side_stream)
                if identity[~packet.valid].tobytes() != raw[~packet.valid].tobytes():
                    raise AssertionError("Escaped bits changed")
                fidelity = ordinary_fidelity(raw, identity, robust.valid)
                _, center, scale, _, _ = parse_side_stream(packet.side_stream)
                row = {"sample_path": str(path), "drive_id": meta["drive_id"], "split": meta["split"],
                       "family": meta["family"], "configuration": meta["configuration"], "policy": policy,
                       "input_cells": raw.size, "fixed_index_bits": 11 * latent_positions(model, *raw.shape),
                       "side_bytes": len(packet.side_stream), "additional_escaped_cells": changed,
                       "escaped_cells": int((~packet.valid).sum()), "reference_ordinary_cells": int(robust.valid.sum()),
                       "normalization_center": center, "normalization_scale": scale,
                       "identity_relative_rmse": fidelity["ordinary_global_centered_relative_rmse"],
                       "identity_correlation": fidelity["ordinary_correlation"],
                       "identity_channel_p90_relative_rmse": fidelity["ordinary_channel_p90_relative_rmse"],
                       "identity_raw_rmse": fidelity["ordinary_raw_rmse"],
                       "threshold": HEADER.unpack_from(packet.side_stream)[5]}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
    summaries, groups = [], []
    keys = ("identity_relative_rmse", "identity_channel_p90_relative_rmse", "identity_raw_rmse")
    for split in ("train", "val"):
        for policy in ("escape-v1", "robust-candidate"):
            values = [r for r in rows if r["split"] == split and r["policy"] == policy]
            # aggregate_validation requires its canonical fidelity keys.
            metric_rows = [{"family": r["family"], "configuration": r["configuration"],
                            "relative_rmse": r["identity_relative_rmse"], "correlation": r["identity_correlation"],
                            **{k:r[k] for k in keys}} for r in values]
            macro, grouped = aggregate_validation(metric_rows)
            bits = sum(r["fixed_index_bits"] + 8 * r["side_bytes"] for r in values)
            cells = sum(r["input_cells"] for r in values)
            summaries.append({"split": split, "policy": policy, **macro, "input_cells": cells,
                "side_bytes": sum(r["side_bytes"] for r in values), "escaped_cells": sum(r["escaped_cells"] for r in values),
                "windows_with_additional_escapes": sum(r["additional_escaped_cells"] > 0 for r in values),
                "pooled_nominal_index_side_ratio": 32 * cells / bits})
            groups += [{"split": split, "policy": policy, **r} for r in grouped]
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "groups.csv", groups)
    (args.output_dir / "COMPLETE").write_text("Identity dynamic-range audit complete; no model updates/test score.\n")


if __name__ == "__main__":
    main()
