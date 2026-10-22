#!/usr/bin/env python3
"""Frozen fold-0 model, paired range policies and one shared ordinary reference."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

try:
    from .analyze_sampling_results import write_csv
    from .diagnose_reconstruction import spectral_metrics
    from .evaluate_checkpoints import metrics
    from .latent_geometry import latent_positions
    from .model_factory import build_model
    from .physical_validation import ordinary_fidelity
    from .reconstruction_objective import aggregate_validation, path_fingerprint
    from .reversible_normalization import encode, decode, parse_side_stream
    from .robust_escape import encode_robust
    from .run_sampling_experiment import ROOT, check_splits
    from .run_temporal_density_long import digest, require_cuda
except ImportError:
    from analyze_sampling_results import write_csv
    from diagnose_reconstruction import spectral_metrics
    from evaluate_checkpoints import metrics
    from latent_geometry import latent_positions
    from model_factory import build_model
    from physical_validation import ordinary_fidelity
    from reconstruction_objective import aggregate_validation, path_fingerprint
    from reversible_normalization import encode, decode, parse_side_stream
    from robust_escape import encode_robust
    from run_sampling_experiment import ROOT, check_splits
    from run_temporal_density_long import digest, require_cuda


def common_scores(raw, restored, reference):
    _, center, scale, _, _ = parse_side_stream(reference.side_stream)
    predicted = ((restored.astype(np.float64) - center) / scale).astype(np.float32)
    predicted[~reference.valid] = 0
    target = reference.values
    outside = ~reference.valid
    error = restored[outside].astype(np.float64) - raw[outside].astype(np.float64)
    return {**metrics(target[reference.valid], predicted[reference.valid]), **spectral_metrics(target, predicted),
            **ordinary_fidelity(raw, restored, reference.valid),
            "reference_ordinary_cell_fraction": float(reference.valid.mean()),
            "outside_reference_raw_rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else 0.}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preserve previous evidence; new output directory required")
    audit = ROOT / "results/experiments/remaining_dynamic_range_audit_20261014"
    if not (audit / "COMPLETE").is_file():
        raise ValueError("Completed numeric audit required")
    require_cuda()
    samples = ROOT / "data/finetune/fold_0"
    check_splits(samples, 0)
    paths = sorted((samples / "val").glob("*.npz"))
    with (samples / "samples_manifest.csv").open() as stream:
        metadata = {str((ROOT / r["sample_path"]).resolve()):r for r in csv.DictReader(stream)}
    if any(metadata[str(p.resolve())]["split"] != "val" for p in paths):
        raise ValueError("Only validation arrays may be scored")
    source = ROOT / "results/experiments/rate_allocation_fold0_seed2027/temporal/best_macro.pt"
    protocol = {"kind": "frozen_range_policy_common_reference_v1", "fold": 0, "split": "val", "windows": len(paths),
                "source": str(source), "source_sha256": digest(source), "validation_hash": path_fingerprint(paths),
                "audit_protocol_sha256": digest(audit / "protocol.json"), "training_updates": 0,
                "reference": "robust candidate ordinary mask/mean/std for BOTH policies; tails separately scored and charged",
                "warning": "Not comparable directly to legacy denominator/mask scores; changed normalizer input, not learned weights. Reused development validation, not external audit or automatic promotion.",
                "code_sha256": {name:digest(ROOT / "finetuning" / name) for name in
                                ("evaluate_frozen_range_policies.py", "robust_escape.py", "reversible_normalization.py", "physical_validation.py")}}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    sys.path.insert(0, args.repo_root)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint["finetune"]["normalization"] != "escape-v1":
        raise ValueError("Frozen source must use baseline escape-v1")
    model = build_model(checkpoint["config"], checkpoint, "cuda").eval()
    del checkpoint
    rows = []
    with (args.output_dir / "windows.csv").open("w", newline="") as stream, torch.no_grad():
        writer = None
        for path in tqdm(paths, desc="[frozen range policies] full val", unit="window"):
            with np.load(path, allow_pickle=False) as payload:
                raw = np.asarray(payload["data"], dtype=np.float32)
            baseline, robust = encode(raw), encode_robust(raw)
            meta = metadata[str(path.resolve())]
            previous_values, previous_prediction = None, None
            for label, packet in (("escape-v1", baseline), ("robust-candidate", robust)):
                if previous_values is not None and np.array_equal(packet.values, previous_values):
                    prediction = previous_prediction
                else:
                    tensor = torch.from_numpy(packet.values).cuda()[None, None]
                    decoded, _, indices, _ = model(tensor)
                    if indices.numel() != latent_positions(model, *raw.shape):
                        raise AssertionError("Index accounting differs from forward")
                    prediction = decoded[0, 0].cpu().numpy()
                previous_values, previous_prediction = packet.values, prediction
                restored = decode(prediction, packet.side_stream)
                if restored[~packet.valid].tobytes() != raw[~packet.valid].tobytes():
                    raise AssertionError("Escaped source bits changed")
                row = {"policy": label, "sample_path": str(path.resolve()), "drive_id": meta["drive_id"],
                       "family": meta["family"], "configuration": meta["configuration"],
                       "input_cells": raw.size, "fixed_index_bits": 11 * latent_positions(model, *raw.shape),
                       "side_bytes": len(packet.side_stream), "policy_escaped_cells": int((~packet.valid).sum()),
                       "changed_window": bool(np.any(baseline.valid & ~robust.valid)),
                       **common_scores(raw, restored, robust)}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
    summaries, groups = [], []
    excluded = {"policy", "sample_path", "drive_id", "family", "configuration", "input_cells", "fixed_index_bits", "side_bytes", "policy_escaped_cells", "changed_window"}
    for label in ("escape-v1", "robust-candidate"):
        for scope in ("all", "changed", "unchanged"):
            subset = [r for r in rows if r["policy"] == label and (scope == "all" or r["changed_window"] == (scope == "changed"))]
            values = [{"family":r["family"], "configuration":r["configuration"], **{k:v for k,v in r.items() if k not in excluded}} for r in subset]
            macro, grouped = aggregate_validation(values)
            cells = sum(r["input_cells"] for r in subset)
            side = sum(r["side_bytes"] for r in subset)
            bits = sum(r["fixed_index_bits"] for r in subset)
            summaries.append({"policy": label, "scope": scope, "windows": len(subset), **macro,
                              "side_bytes": side, "input_cells": cells,
                              "pooled_index_side_ratio": 32 * cells / (bits + 8 * side)})
            groups += [{"policy":label, "scope":scope, **r} for r in grouped]
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "groups.csv", groups)
    (args.output_dir / "COMPLETE").write_text("Frozen paired normalization evaluation complete, no learned-weight changes/test access.\n")


if __name__ == "__main__":
    main()
