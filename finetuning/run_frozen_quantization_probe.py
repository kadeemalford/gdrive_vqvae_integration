#!/usr/bin/env python3
"""VQ versus raw encoder latents through an unchanged decoder; validation only."""
import argparse
import csv
import fcntl
from functools import partial
import json
from pathlib import Path
import shutil
import sys

import torch
from tqdm import tqdm

try:
    from .finetune_vqvae import load_sample, load_valid_mask
    from .model_factory import build_model
    from .latent_geometry import apply_geometry, checkpoint_geometry
    from .physical_validation import restore_for_validation
    from .quantization_probe import QuantizationProbe, payload_bits
    from .norm_matched_probe import NormMatchedProbe
    from .reconstruction_objective import evaluate_detailed, validation_metadata, path_fingerprint
    from .run_sampling_experiment import ROOT, check_splits
    from .run_temporal_density_long import digest, require_cuda
except ImportError:
    from finetune_vqvae import load_sample, load_valid_mask
    from model_factory import build_model
    from latent_geometry import apply_geometry, checkpoint_geometry
    from physical_validation import restore_for_validation
    from quantization_probe import QuantizationProbe, payload_bits
    from norm_matched_probe import NormMatchedProbe
    from reconstruction_objective import evaluate_detailed, validation_metadata, path_fingerprint
    from run_sampling_experiment import ROOT, check_splits
    from run_temporal_density_long import digest, require_cuda


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--norm-matched-only", action="store_true", help="Follow-up diagnostic at quantized latent energy, using prior completed controls.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    gain = ROOT / "results/experiments/frozen_gain_calibration_20261018"
    if not (gain / "COMPLETE").is_file():
        raise ValueError("Completed train-only gain study required")
    sources = {"rate2": ROOT / "results/experiments/frozen_rate_stability_20261018/midpoint_2x.pt",
               "rate4": ROOT / "results/experiments/high_rate_fold0_robust_20261017/rate_4x/best_macro.pt"}
    smoke = ROOT / "results/experiments/frozen_quantization_probe_smoke_20261019"
    if not (smoke / "COMPLETE").is_file():
        raise ValueError("VQ/continuous GPU and escape smoke required")
    smoke_result = json.loads((smoke / "summary.json").read_text())
    if (smoke_result["source_sha256"] != {label: digest(p) for label, p in sources.items()}
            or smoke_result["code_sha256"] != digest(Path(__file__).with_name("quantization_probe.py"))):
        raise ValueError("Smoke source/probe mismatch")
    samples = ROOT / "data/finetune/fold_0"
    check_splits(samples, 0)
    paths = sorted((samples / "val").glob("*.npz"))
    metadata = validation_metadata(paths, samples / "samples_manifest.csv")
    with (samples / "samples_manifest.csv").open() as stream:
        expected = {str((ROOT / r["sample_path"]).resolve()) for r in csv.DictReader(stream) if r["split"] == "val"}
    if not paths or {str(p.resolve()) for p in paths} != expected:
        raise ValueError("Complete validation coordinates required")
    code = [ROOT / f"finetuning/{name}.py" for name in (
        "run_frozen_quantization_probe", "quantization_probe", "norm_matched_probe", "finetune_vqvae", "model_factory", "latent_geometry",
        "physical_validation", "robust_escape", "reversible_normalization", "channel_normalization",
        "reconstruction_objective", "diagnose_reconstruction", "evaluate_checkpoints", "run_sampling_experiment")]
    code += [Path(args.repo_root) / f"vqvae/{n}.py" for n in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "frozen_quantization_probe_v1", "fold": 0, "split": "val", "windows": len(paths),
                "validation_hash": path_fingerprint(paths), "network_training_updates": 0,
                "repo_root": str(Path(args.repo_root).resolve()), "normalization": "robust-escape-v1",
                "sources": {label: {"path": str(p), "sha256": digest(p)} for label, p in sources.items()},
                "manifest_sha256": digest(samples / "samples_manifest.csv"),
                "smoke_sha256": digest(smoke / "summary.json"),
                "code_sha256": {str(p): digest(p) for p in code},
                "warning": "Raw encoder latents are off the decoder's training distribution; a negative result does not prove quantization is harmless. A positive result is not an achievable rate-matched codec. Continuous control charges all float32 latent components plus side, not retained diagnostic VQ indices. Reference VQ computation remains active, so timing is not deployed-codec compute. Reused validation, no test access/promotion."}
    if args.norm_matched_only:
        reference = ROOT / "results/experiments/frozen_quantization_probe_20261019"
        if not (reference / "COMPLETE").is_file():
            raise ValueError("Completed raw continuous/VQ controls required")
        old = json.loads((reference / "protocol.json").read_text())
        if old["sources"] != protocol["sources"] or old["validation_hash"] != protocol["validation_hash"]:
            raise ValueError("Prior controls source/validation mismatch")
        norm_smoke = ROOT / "results/experiments/norm_matched_quantization_smoke_20261019"
        if not (norm_smoke / "COMPLETE").is_file():
            raise ValueError("Norm-matched GPU/input/escape smoke required")
        norm_result = json.loads((norm_smoke / "summary.json").read_text())
        if (norm_result["source_sha256"] != {label: digest(p) for label, p in sources.items()}
                or norm_result["helper_sha256"] != digest(Path(__file__).with_name("norm_matched_probe.py"))):
            raise ValueError("Norm-matched smoke source/helper mismatch")
        protocol.update(kind="frozen_norm_matched_quantization_probe_v1", modes=["norm_matched"],
                        norm_smoke_sha256=digest(norm_smoke / "summary.json"),
                        reference_experiment=str(reference), reference_summary_sha256=digest(reference / "summary.csv"),
                        reference_protocol_sha256=digest(reference / "protocol.json"),
                        control_rule="Multiply encoder z by norm(zq)/norm(z) per window, preserving z direction and total quantized energy. No fitted coefficient, target score, family routing or alpha sweep. Transmit scaled float32 z in full; no gain side is needed because scaling is already in transmitted components. Still off-manifold; norm matching does not align distribution/covariance.")
    else:
        protocol["modes"] = ["vq", "continuous"]
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[quantization probe] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("New directory or --resume required")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 2 * 1024 ** 3:
        raise RuntimeError("2 GiB free required; no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            atomic_json(output / "protocol.json", protocol)
        sys.path.insert(0, args.repo_root)
        summaries = []
        for source_label, path in sources.items():
            for mode in protocol["modes"]:
                label = f"{source_label}_{mode}"
                summary_path = output / f"{label}.json"
                if summary_path.exists():
                    summaries.append(json.loads(summary_path.read_text()))
                    continue
                saved = torch.load(path, map_location="cpu", weights_only=False)
                state = saved["finetune"]
                if state["normalization"] != "robust-escape-v1" or Path(state["samples_dir"]).resolve() != samples.resolve():
                    raise ValueError("Source normalization/fold mismatch")
                model = apply_geometry(build_model(saved["config"], saved, "cuda"), checkpoint_geometry(saved)).eval()
                model.requires_grad_(False)
                del saved
                probe = NormMatchedProbe() if mode == "norm_matched" else QuantizationProbe(continuous=mode == "continuous")
                handle = model.quantizer.register_forward_hook(probe)
                totals = {"input_cells": 0, "latent_payload_bits": 0, "side_bytes": 0}
                probe_sums = {k: 0. for k in ("relative_quantization_error", "encoder_latent_rms", "quantized_latent_rms", "reference_vq_loss")}
                attempt = len(list(output.glob(f"{label}.windows.attempt_*.csv"))) + 1
                windows_path = output / f"{label}.windows.attempt_{attempt:03d}.csv"
                progress = tqdm(total=len(paths), desc=f"[quantization probe] {label}", unit="window")
                with windows_path.open("x", newline="") as stream:
                    writer = None
                    def on_sample(sample_path, metrics):
                        nonlocal writer
                        meta = metadata[sample_path]
                        cells = int(meta["channels"]) * int(meta["time_samples"])
                        diagnostic = probe.latest
                        bits = payload_bits(diagnostic["latent_positions"], diagnostic["latent_dimensions"], model.num_embeddings, mode != "vq")
                        side = round(metrics["side_bits_per_cell"] * cells / 8)
                        row = {"sample_path": str(sample_path), "drive_id": meta["drive_id"], "title": meta["title"],
                               **metrics, **diagnostic, "input_cells": cells, "latent_payload_bits": bits, "side_bytes": side,
                               "payload_plus_side_nominal_ratio": 32 * cells / (bits + 8 * side)}
                        # The generic physical transform assumes VQ indices; suppress its rate estimate here.
                        del row["index_plus_side_nominal_ratio"]
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        for key, value in (("input_cells", cells), ("latent_payload_bits", bits), ("side_bytes", side)):
                            totals[key] += value
                        for key in probe_sums:
                            probe_sums[key] += diagnostic[key]
                        progress.update()
                    def transform(sample_path, decoded):
                        target, restored, extras = restore_for_validation(sample_path, decoded, "robust-escape-v1", model, detailed_channels=True)
                        return target, restored, extras
                    try:
                        macro, groups = evaluate_detailed(model, paths, metadata, "cuda",
                            partial(load_sample, normalization="robust-escape-v1"),
                            load_valid_mask=partial(load_valid_mask, normalization="robust-escape-v1"),
                            fidelity_transform=transform, on_sample=on_sample)
                    finally:
                        handle.remove()
                        progress.close()
                del macro["macro_index_plus_side_nominal_ratio"]
                for group in groups:
                    del group["index_plus_side_nominal_ratio"]
                summary = {"label": label, "mode": mode, "is_continuous_control": mode != "vq",
                           "windows_csv": str(windows_path), "windows_sha256": digest(windows_path),
                           **macro, **totals, "pooled_payload_plus_side_ratio": 32 * totals["input_cells"] /
                           (totals["latent_payload_bits"] + 8 * totals["side_bytes"]),
                           **{f"window_mean_{k}": v / len(paths) for k, v in probe_sums.items()}}
                write_csv(output / f"{label}.groups.csv", groups)
                atomic_json(summary_path, summary)
                summaries.append(summary)
                del model
                torch.cuda.empty_cache()
        write_csv(output / "summary.csv", summaries)
        (output / "COMPLETE").write_text("Frozen VQ/continuous-latent diagnostic complete; not a rate-matched codec or test result.\n")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
