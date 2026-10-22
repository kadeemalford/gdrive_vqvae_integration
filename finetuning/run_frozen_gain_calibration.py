#!/usr/bin/env python3
"""Fit one gain on training only, then evaluate fixed-gain/identity on validation."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil
import sys

import torch
from tqdm import tqdm

try:
    from .finetune_vqvae import load_sample, load_valid_mask, save_checkpoint
    from .frozen_gain import window_moments, fitted_gain, fold_gain
    from .model_factory import build_model
    from .latent_geometry import apply_geometry, checkpoint_geometry
    from .reconstruction_objective import path_fingerprint
    from .run_sampling_experiment import ROOT, check_splits, run
    from .run_temporal_density_long import digest, require_cuda
except ImportError:
    from finetune_vqvae import load_sample, load_valid_mask, save_checkpoint
    from frozen_gain import window_moments, fitted_gain, fold_gain
    from model_factory import build_model
    from latent_geometry import apply_geometry, checkpoint_geometry
    from reconstruction_objective import path_fingerprint
    from run_sampling_experiment import ROOT, check_splits, run
    from run_temporal_density_long import digest, require_cuda


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    stability = ROOT / "results/experiments/frozen_rate_stability_20261018"
    audit = ROOT / "results/experiments/checkpoint_selection_audit_20261018"
    if not all((p / "COMPLETE").is_file() for p in (stability, audit)):
        raise ValueError("Completed stability and selection audits required")
    sources = {"rate2": stability / "midpoint_2x.pt",
               "rate4": ROOT / "results/experiments/high_rate_fold0_robust_20261017/rate_4x/best_macro.pt"}
    smoke = ROOT / "results/experiments/frozen_gain_smoke_20261018"
    if not (smoke / "COMPLETE").is_file():
        raise ValueError("Fixed-gain GPU/escape/index smoke required")
    smoke_result = json.loads((smoke / "summary.json").read_text())
    if smoke_result["source_sha256"] != {label: digest(path) for label, path in sources.items()}:
        raise ValueError("Smoke/source mismatch")
    samples = ROOT / "data/finetune/fold_0"
    check_splits(samples, 0)
    manifest = samples / "samples_manifest.csv"
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    train_paths = sorted((ROOT / r["sample_path"]).resolve() for r in rows if r["split"] == "train")
    if not train_paths or len(set(train_paths)) != len(train_paths) or any(not p.is_file() for p in train_paths):
        raise ValueError("Complete unique training coordinates required")
    if set(train_paths) != set(p.resolve() for p in (samples / "train").glob("*.npz")):
        raise ValueError("Train directory/manifest mismatch")
    code = [ROOT / f"finetuning/{name}.py" for name in (
        "run_frozen_gain_calibration", "frozen_gain", "finetune_vqvae", "model_factory", "latent_geometry",
        "physical_validation", "robust_escape", "reversible_normalization", "channel_normalization",
        "reconstruction_objective", "evaluate_physical_candidates", "run_sampling_experiment", "run_temporal_density_long")]
    code += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "train_only_frozen_global_gain_v1", "fold": 0, "fit_split": "train", "evaluation_split": "val",
                "repo_root": str(Path(args.repo_root).resolve()), "network_training_updates": 0,
                "fitted_scalars_per_model": 1, "scalar_bits_per_model": 32, "fit_windows": len(train_paths),
                "fit_path_sha256": path_fingerprint(train_paths), "manifest_sha256": digest(manifest),
                "smoke_sha256": digest(smoke / "summary.json"),
                "sources": {label: {"path": str(p), "sha256": digest(p)} for label, p in sources.items()},
                "code_sha256": {str(p): digest(p) for p in code},
                "note": "Single positive float32 gain minimizes mean per-window ordinary normalized MSE on all train windows, without SGD. No validation fitting, metadata routing, per-window gain, bias, forced RMS matching or alpha search. Gain may be below one. Fold gain into final affine decoder head, same indices/parameters. Charge 32 global control bits conservatively; no per-window overhead. Reused exploratory validation, no test access/promotion."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume source/protocol/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[frozen gain] already complete: {output}")
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
        for label, path in sources.items():
            source = torch.load(path, map_location="cpu", weights_only=False)
            state = source["finetune"]
            if state["normalization"] != "robust-escape-v1" or Path(state["samples_dir"]).resolve() != samples.resolve():
                raise ValueError("Source fold/normalization mismatch")
            model = apply_geometry(build_model(source["config"], source, "cuda"), checkpoint_geometry(source)).eval()
            model.requires_grad_(False)
            progress_file = output / f"{label}.fit.json"
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {
                "completed_windows": 0, "xy": 0., "yy": 0., "xx": 0.,
                "source_sha256": protocol["sources"][label]["sha256"], "fit_path_sha256": protocol["fit_path_sha256"]}
            if (progress["source_sha256"] != protocol["sources"][label]["sha256"]
                    or progress["fit_path_sha256"] != protocol["fit_path_sha256"]
                    or not 0 <= progress["completed_windows"] <= len(train_paths)):
                raise ValueError("Fit cursor/source mismatch")
            with torch.inference_mode():
                for index in tqdm(range(progress["completed_windows"], len(train_paths)), desc=f"[fit gain] {label}", unit="window"):
                    p = train_paths[index]
                    values = load_sample(p, normalization="robust-escape-v1")
                    valid = load_valid_mask(p, normalization="robust-escape-v1")
                    tensor = torch.from_numpy(values).cuda()[None, None]
                    prediction = model(tensor)[0][0, 0].cpu().numpy()
                    moments = window_moments(values, prediction, valid)
                    for key, value in moments.items():
                        progress[key] += value
                    progress["completed_windows"] = index + 1
                    if (index + 1) % 50 == 0:
                        atomic_json(progress_file, progress)
            gain = fitted_gain(progress)
            progress.update(gain=gain, objective="equal-window ordinary normalized MSE",
                            identity_train_mse=(progress["xx"] - 2 * progress["xy"] + progress["yy"]) / len(train_paths),
                            calibrated_train_mse=(progress["xx"] - 2 * gain * progress["xy"] + gain * gain * progress["yy"]) / len(train_paths))
            atomic_json(progress_file, progress)
            fold_gain(model, gain)
            calibrated = {"config": source["config"], "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                          "epoch": 0, "finetune": {"normalization": "robust-escape-v1",
                          "latent_geometry": checkpoint_geometry(source), "samples_dir": str(samples.resolve()), "physical_validation": True},
                          "frozen_global_gain": {"gain": gain, "fit_split": "train", "fit_path_sha256": protocol["fit_path_sha256"],
                          "source_sha256": protocol["sources"][label]["sha256"], "network_training_updates": 0}}
            destination = output / f"{label}.calibrated.pt"
            if not destination.exists():
                save_checkpoint(calibrated, destination)
            else:
                existing = torch.load(destination, map_location="cpu", weights_only=False)
                if (existing.get("frozen_global_gain") != calibrated["frozen_global_gain"]
                        or existing["config"] != calibrated["config"] or existing["finetune"] != calibrated["finetune"]
                        or existing["model_state_dict"].keys() != calibrated["model_state_dict"].keys()
                        or any(not torch.equal(v, existing["model_state_dict"][k]) for k, v in calibrated["model_state_dict"].items())):
                    raise ValueError("Existing calibrated checkpoint differs")
                del existing
            print(f"[gain] {label}: {gain:.9g}", flush=True)
            del source, model, calibrated
            torch.cuda.empty_cache()
        full = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--fold", "0", "--output-dir", str(full)]
        for label, path in sources.items():
            for suffix, candidate in (("identity", path), ("calibrated", output / f"{label}.calibrated.pt")):
                command += ["--candidate", f"{label}_{suffix}=robust-escape-v1={candidate}"]
        if full.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        with (full / "summary.csv").open() as stream:
            measured = list(csv.DictReader(stream))
        charged = []
        for row in measured:
            bits = 32 if row["label"].endswith("_calibrated") else 0
            charged.append({"label": row["label"], "model_gain_bits": bits,
                            "pooled_ratio_with_global_gain": 32 * int(row["input_cells"]) /
                            (int(row["fixed_index_bits"]) + 8 * int(row["side_bytes"]) + bits)})
        atomic_json(output / "charged_rate.json", charged)
        (output / "COMPLETE").write_text("Train-only scalar calibration/full validation complete; no network training or promotion.\n")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
