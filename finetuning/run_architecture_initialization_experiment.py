#!/usr/bin/env python3
"""Three-arm FEBUS-only architecture/initialization screening, with val-only audit."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import random
import shutil
import sys

import torch
import torch.nn.functional as F

try:
    from .finetune_vqvae import load_sample, save_checkpoint
    from .model_factory import architecture, new_model
    from .run_sampling_experiment import ROOT, check_splits, run
    from .run_temporal_density_long import digest, require_cuda, remaining_epochs
except ImportError:
    from finetune_vqvae import load_sample, save_checkpoint
    from model_factory import architecture, new_model
    from run_sampling_experiment import ROOT, check_splits, run
    from run_temporal_density_long import digest, require_cuda, remaining_epochs

ARMS = {"transferred_4x": "native-v1", "scratch_4x": "native-v1", "scratch_temporal": "temporal-dilated-v1"}


def initialize_codebook(model, paths, seed, normalization="escape-v1"):
    """Training-only latent reservoir, no gradient updates or validation calibration."""
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reservoir = []
    model.eval()
    with torch.no_grad():
        for path in paths:
            values = load_sample(path, normalization=normalization)
            tensor = torch.from_numpy(values).to(device)[None, None]
            multiple = 2 ** model.downsample_levels
            padded = F.pad(tensor, (0, (-values.shape[1]) % multiple, 0, (-values.shape[0]) % multiple))
            z = model.encoder(padded).permute(0, 2, 3, 1).reshape(-1, model.embedding_dim).cpu()
            if not torch.isfinite(z).all():
                raise ValueError("Nonfinite codebook calibration latents")
            reservoir.append(z[torch.randperm(len(z), generator=generator)[:256]])
        vectors = torch.cat(reservoir)
        if len(vectors) < model.num_embeddings:
            raise ValueError("Insufficient training latent reservoir for codebook initialization")
        chosen = vectors[torch.randperm(len(vectors), generator=generator)[:model.num_embeddings]]
        model.quantizer.embedding.weight.copy_(chosen.to(device))
    return len(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = ROOT / "results/experiments/temporal_density_long_20261008/temporal-4x/best_macro.pt"
    if not (source.parent.parent / "COMPLETE").is_file():
        raise ValueError("Completed 4x reference required")
    samples = ROOT / "data/finetune/fold_1"
    check_splits(samples, 1)
    with (samples / "samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    train = sorted((ROOT / row["sample_path"]).resolve() for row in rows if row["split"] == "train")
    if any(not (ROOT / row["sample_path"]).is_file() for row in rows if row["split"] in ("train", "val")):
        raise FileNotFoundError("Complete train/validation arrays required")
    for split in ("train", "val"):
        if {row["family"] for row in rows if row["split"] == split} != {"doped", "multicore", "multimode", "single_mode"}:
            raise ValueError("Four-family train and validation required")
    rng = random.Random(2026)
    calibration = []
    for family in ("doped", "multicore", "multimode", "single_mode"):
        candidates = sorted((ROOT / row["sample_path"]).resolve() for row in rows
                            if row["split"] == "train" and row["family"] == family)
        calibration.extend(rng.sample(candidates, 8))
    calibration = sorted(calibration)
    smoke = ROOT / "results/experiments/architecture_initialization_smoke_20261010"
    if not (smoke / "COMPLETE").is_file():
        raise ValueError("Maximum-size architecture gradient/memory smoke required")
    smoke_summary = json.loads((smoke / "summary.json").read_text())
    if smoke_summary["source_sha256"] != digest(source):
        raise ValueError("Smoke/reference checkpoint mismatch")
    code_paths = [ROOT / f"finetuning/{name}.py" for name in
                  ("model_factory", "finetune_vqvae", "latent_geometry", "physical_validation",
                   "reconstruction_objective", "reversible_normalization", "evaluate_physical_candidates",
                   "sampling", "run_architecture_initialization_experiment")]
    code_paths += [Path(args.repo_root) / f"vqvae/{name}.py" for name in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "architecture_initialization_screening_v1", "fold": 1, "arms": list(ARMS),
                "architectures": ARMS, "source": str(source), "source_sha256": digest(source),
                "repo_root": str(Path(args.repo_root).resolve()), "seed": 2026, "epochs": 16,
                "steps_per_epoch": 4096, "learning_rate": 1e-4, "weight_decay": 1e-5,
                "patience": 17, "min_delta": 1e-4, "normalization": "escape-v1",
                "latent_geometry": "temporal-4x", "sampling": "window", "physical_validation": True,
                "evaluate_initial": True, "initial_validation_expected_equal": False, "trainable": "full",
                "val_samples_per_file": 2, "evaluation_split": "val", "codebook_diagnostics": True,
                "channel_diagnostics": True, "train_windows": len(train), "manifest_sha256": digest(samples / "samples_manifest.csv"),
                "scratch_codebook_initialization": {"kind": "training_latent_reservoir_v1", "windows": [str(p) for p in calibration],
                                                     "windows_per_family": 8, "vectors_per_window_max": 256, "optimizer_updates": 0},
                "smoke_sha256": digest(smoke / "summary.json"),
                "code_sha256": {str(path): digest(path) for path in code_paths},
                "note": "Same additional 65,536 updates per arm, geometry, codebook size, objective and data order. Scratch has no pretrained tensors; train-only codebook seeding in both scratch arms. Transferred history is NOT matched total compute. Screening, not convergence/final acceptance; no early stopping within fixed budget."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[architecture] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("Use a new directory or --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 8 * 1024 ** 3:
        raise RuntimeError("8 GiB free required; no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        sys.path.insert(0, args.repo_root)
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if architecture(saved["config"]) != "native-v1":
            raise ValueError("Expected native reference architecture")
        config = {key: saved["config"][key] for key in
                  ("base_channels", "embedding_dim", "num_embeddings", "downsample_levels", "commitment_cost")}
        config["latent_geometry"] = "temporal-4x"
        del saved
        init_root = output / "initializations"
        init_root.mkdir(exist_ok=True)
        ledger = output / "initializations.json"
        sources = {"transferred_4x": source, **{arm: init_root / f"{arm}.pt" for arm in ARMS if arm != "transferred_4x"}}
        if ledger.exists():
            for arm, info in json.loads(ledger.read_text()).items():
                if digest(sources[arm]) != info["sha256"]:
                    raise ValueError("Initialization checkpoint checksum mismatch")
        else:
            counts = {}
            for arm in ARMS:
                if arm == "transferred_4x":
                    torch.manual_seed(2026)
                    reference = new_model(config, "cpu")
                    counts[arm] = sum(p.numel() for p in reference.parameters())
                    del reference
                    continue
                if sources[arm].exists():
                    initial = torch.load(sources[arm], map_location="cpu", weights_only=False)
                    if initial["initialization_protocol"] != protocol:
                        raise ValueError("Interrupted initialization protocol mismatch")
                    counts[arm] = initial["parameter_count"]
                    del initial
                    continue
                torch.manual_seed(2026)
                arm_config = {**config, "architecture": ARMS[arm]}
                model = new_model(arm_config, "cuda")
                vectors = initialize_codebook(model, calibration, 2026)
                counts[arm] = sum(p.numel() for p in model.parameters())
                save_checkpoint({"config": arm_config, "epoch": 0, "model_state_dict": model.cpu().state_dict(),
                                 "parameter_count": counts[arm], "initialization_protocol": protocol,
                                 "initialization": {"kind": "scratch", "latent_reservoir_vectors": vectors,
                                                    "pretrained_tensors_loaded": False}}, sources[arm])
                del model
                torch.cuda.empty_cache()
            if len(set(counts.values())) != 1:
                raise ValueError("Architectures must have exactly matched parameter counts")
            ledger.write_text(json.dumps({arm: {"checkpoint": str(path), "sha256": digest(path), "parameters": counts[arm]}
                                          for arm, path in sources.items()}, indent=2) + "\n")
        for arm in ARMS:
            target = output / arm
            epochs, last = remaining_epochs(target, protocol, "temporal-4x") if args.resume else (protocol["epochs"], None)
            if not epochs:
                continue
            command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root,
                       "--checkpoint", str(sources[arm]), "--samples-dir", str(samples), "--output-dir", str(target),
                       "--epochs", str(epochs), "--steps-per-epoch", "4096", "--learning-rate", "1e-4",
                       "--weight-decay", "1e-5", "--seed", "2026", "--device", "cuda", "--sampling", "window",
                       "--normalization", "escape-v1", "--latent-geometry", "temporal-4x", "--physical-validation",
                       "--validation-details", "--evaluate-initial", "--val-samples-per-file", "2",
                       "--codebook-diagnostics", "--channel-diagnostics", "--early-stopping-metric", "macro-rmse",
                       "--early-stopping-patience", "17", "--early-stopping-min-delta", "1e-4"]
            if last:
                command += ["--resume", str(last)]
            run(command, output / f"{arm}.log", resume=args.resume, quiet=True)
        # Selection remains on the pilot; full val measures selected candidates, not new selection.
        audit = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--output-dir", str(audit)]
        for arm in ARMS:
            command += ["--candidate", f"{arm}=escape-v1={output / arm / 'best_macro.pt'}"]
        if audit.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("Architecture/initialization screening and full selected validation complete; no test access or promotion.\n")
        run(["finetuning/summarize_numeric_adaptation.py", "--experiment-dir", str(output)], output / "comparison.log", quiet=True)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
