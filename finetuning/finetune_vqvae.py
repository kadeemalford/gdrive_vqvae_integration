#!/usr/bin/env python3
"""Train versioned VQ-VAE candidates on native FEBUS windows, with safe resume."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from functools import partial
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .domain_routing import family_paths, FAMILIES
    from .codebook_refresh import CodebookRefresh
    from .model_factory import build_model, architecture
    from .latent_geometry import apply_geometry, checkpoint_geometry, POLICIES
    from .channel_normalization import encode_channel
    from .physical_validation import restore_for_validation
    from .reversible_normalization import encode
    from .robust_escape import encode_robust
    from .channel_objective import bounded_channel_mse, CHANNEL_WEIGHT_CAP
    from .sampling import epoch_paths, sample_groups
    from .reconstruction_objective import (append_metrics, evaluate_detailed, multiresolution_spectral_loss,
                                           path_fingerprint, stratified_validation, validation_metadata)
except ImportError:
    from domain_routing import family_paths, FAMILIES
    from codebook_refresh import CodebookRefresh
    from model_factory import build_model, architecture
    from latent_geometry import apply_geometry, checkpoint_geometry, POLICIES
    from channel_normalization import encode_channel
    from physical_validation import restore_for_validation
    from reversible_normalization import encode
    from robust_escape import encode_robust
    from channel_objective import bounded_channel_mse, CHANNEL_WEIGHT_CAP
    from sampling import epoch_paths, sample_groups
    from reconstruction_objective import (append_metrics, evaluate_detailed, multiresolution_spectral_loss,
                                          path_fingerprint, stratified_validation, validation_metadata)


def load_sample(path: Path, normalization="legacy") -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        data = np.asarray(payload["data"], dtype=np.float32)
    if normalization == "escape-v1":
        return encode(data).values
    if normalization == "channel-escape-v1":
        return encode_channel(data).values
    if normalization == "robust-escape-v1":
        return encode_robust(data).values
    if normalization != "legacy":
        raise ValueError("Unknown normalization policy")
    # FEBUS spikes can overflow float32 reductions before normalization.
    mean = float(np.mean(data, dtype=np.float64))
    std = float(np.std(data, dtype=np.float64))
    return ((data.astype(np.float64) - mean) / (std + 1e-6)).astype(np.float32)


def load_valid_mask(path, normalization="escape-v1"):
    with np.load(path, allow_pickle=False) as payload:
        data = np.asarray(payload["data"], dtype=np.float32)
    if normalization == "robust-escape-v1":
        return encode_robust(data).valid
    if normalization not in ("escape-v1", "channel-escape-v1"):
        raise ValueError("Unknown mask normalization policy")
    return encode(data).valid


def save_checkpoint(payload, path):
    """Keep the previous complete checkpoint if serialization is interrupted."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def configure_trainable(model, policy):
    if policy not in ("full", "decoder"):
        raise ValueError("Unknown trainable policy")
    if policy == "decoder" and not all(hasattr(model, key) for key in ("encoder", "quantizer", "decoder")):
        raise ValueError("Decoder-only policy requires encoder, quantizer and decoder modules")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(policy == "full" or name.startswith("decoder."))
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    return parameters


def evaluate(model, paths: list[Path], device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correlations: list[float] = []
    with torch.no_grad():
        for path in paths:
            values = load_sample(path)
            tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device)
            recon, vq_loss, _, _ = model(tensor)
            recon_loss = 0.5 * F.mse_loss(recon, tensor) + 0.5 * F.l1_loss(recon, tensor)
            losses.append(float((recon_loss + vq_loss).item()))
            decoded = recon[0, 0].cpu().numpy().ravel()
            correlations.append(float(np.corrcoef(values.ravel(), decoded)[0, 1]) if np.std(decoded) > 0 else 0.0)
    return float(np.mean(losses)), float(np.mean(correlations))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional deterministic training subset; 0 uses all samples.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional deterministic validation subset; 0 uses all samples.")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=5e-4)
    parser.add_argument("--resume", type=Path, help="Continue from a checkpoint saved by this script.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampling", choices=["window", "group", "family-group", "family-window"], default="window")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 uses the number of training windows; other values are update budgets.")
    parser.add_argument("--spectral-weight", type=float, default=0.0, help="Weight of optional multi-resolution STFT objective; 0 preserves temporal training loss.")
    parser.add_argument("--channel-mse-weight", type=float, default=0.0, help="Optional bounded inverse-variance channel MSE weight; 0 leaves the historical training objective unchanged.")
    parser.add_argument("--validation-details", action="store_true", help="Record per-group/macro fidelity and save best_macro.pt; best.pt stays selected by temporal+VQ val_loss.")
    parser.add_argument("--val-samples-per-file", type=int, default=0, help="Validation-only stratified pilot cap; 0 evaluates all validation windows.")
    parser.add_argument("--evaluate-initial", action="store_true", help="With validation-details, include initialization as epoch 0 checkpoint candidate in a new branch.")
    parser.add_argument("--normalization", choices=["legacy", "escape-v1", "channel-escape-v1", "robust-escape-v1"], default="legacy",
                        help="escape-v1 excludes finite abs>1e20 from scaling/loss; restore via exact side stream at deployment.")
    parser.add_argument("--trainable", choices=["full", "decoder"], default="full")
    parser.add_argument("--early-stopping-metric", choices=["val-loss", "macro-rmse"], default="val-loss")
    parser.add_argument("--physical-validation", action="store_true", help="Restore raw output then score both policies on the SAME global-normalized ordinary target; charge side parameters.")
    parser.add_argument("--latent-geometry", choices=POLICIES, help="Versioned stride/rate allocation; inherits source/resume metadata by default. Spatial policies add decoder-cropped bottom padding.")
    parser.add_argument("--codebook-diagnostics", action="store_true", help="Record epoch code usage/perplexity and train reconstruction/VQ losses.")
    parser.add_argument("--channel-diagnostics", action="store_true", help="Include robust channel statistics in physical validation.")
    parser.add_argument("--codebook-refresh", choices=("none", "dead-v1"), default="none",
                        help="Optional train-only bounded refresh after an epoch; never during validation.")
    parser.add_argument("--domain-family", choices=FAMILIES,
                        help="Train/select on one manifest family only; same original configuration split.")
    args = parser.parse_args()
    explicit_geometry = args.latent_geometry is not None
    if not math.isfinite(args.channel_mse_weight) or args.channel_mse_weight < 0:
        raise ValueError("Channel MSE weight must be finite/nonnegative")
    if args.channel_mse_weight and (not args.physical_validation or args.normalization == "legacy"):
        raise ValueError("Channel MSE objective requires physical validation and an ordinary escape mask")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1")
    if not math.isfinite(args.spectral_weight) or args.spectral_weight < 0 or args.val_samples_per_file < 0:
        raise ValueError("Spectral weight and per-file validation cap must be finite/nonnegative")
    if args.val_samples_per_file and args.max_val_samples:
        raise ValueError("Use either per-file validation cap or max-val-samples, not both")
    if args.evaluate_initial and not args.validation_details:
        raise ValueError("--evaluate-initial requires --validation-details")
    if args.early_stopping_metric == "macro-rmse" and not args.validation_details:
        raise ValueError("Macro early stopping requires validation-details")
    if args.normalization != "legacy" and (not args.validation_details or args.spectral_weight):
        raise ValueError("escape-v1 requires validation-details and spectral-weight=0; spectra with zero-filled escape cells are diagnostic only")
    if args.normalization == "channel-escape-v1" and not args.physical_validation:
        raise ValueError("Channel normalization requires physical-validation to avoid incomparable selectors")
    if args.normalization == "robust-escape-v1" and not args.physical_validation:
        raise ValueError("Robust normalization requires physical-validation and a common robust score reference")
    if args.physical_validation and (args.normalization == "legacy" or not args.validation_details):
        raise ValueError("Physical validation requires an escape policy and validation-details")
    if args.channel_diagnostics and not args.physical_validation:
        raise ValueError("Channel diagnostics require physical-validation")
    if args.codebook_refresh != "none" and not args.codebook_diagnostics:
        raise ValueError("Codebook refresh requires codebook-diagnostics")
    sample_loader = partial(load_sample, normalization=args.normalization)
    mask_loader = partial(load_valid_mask, normalization=args.normalization) if args.normalization != "legacy" else None
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.repo_root))
    from vqvae.evaluate_vqvae import load_checkpoint_config

    marker = args.samples_dir / "PRUNED_ARRAYS.json"
    if marker.is_file() and json.loads(marker.read_text()).get("status") != "restored":
        raise ValueError("Prepared dataset was pruned/incomplete; restore_pruned_samples.py is required before training")
    train_paths = sorted((args.samples_dir / "train").glob("*.npz"))
    val_paths = sorted((args.samples_dir / "val").glob("*.npz"))
    if args.domain_family:
        manifest = args.samples_dir / "samples_manifest.csv"
        train_paths = family_paths(train_paths, manifest, args.domain_family, "train")
        val_paths = family_paths(val_paths, manifest, args.domain_family, "val")
    selection_rng = random.Random(args.seed)
    if args.max_train_samples and args.max_train_samples < len(train_paths):
        train_paths = sorted(selection_rng.sample(train_paths, args.max_train_samples))
    if args.max_val_samples and args.max_val_samples < len(val_paths):
        val_paths = sorted(selection_rng.sample(val_paths, args.max_val_samples))
    if not train_paths or not val_paths:
        raise ValueError("Both train and val sample directories must be non-empty")
    metadata = None
    if args.validation_details or args.val_samples_per_file:
        metadata = validation_metadata(val_paths, args.samples_dir / "samples_manifest.csv")
        val_paths = stratified_validation(val_paths, metadata, args.val_samples_per_file, args.seed)
    validation_hash = path_fingerprint(val_paths)
    if args.steps_per_epoch < 0 or args.epochs < 1:
        raise ValueError("Epochs must be positive and steps-per-epoch nonnegative")
    groups = sample_groups(train_paths, args.samples_dir / "samples_manifest.csv") if args.sampling != "window" else {}
    device = torch.device(args.device)
    checkpoint, config = load_checkpoint_config(args.checkpoint, device)
    model = build_model(config, checkpoint, device)
    args.latent_geometry = args.latent_geometry or checkpoint_geometry(checkpoint)
    apply_geometry(model, args.latent_geometry)
    if args.latent_geometry != "standard" and not args.physical_validation:
        raise ValueError("Temporal latent geometry requires physical-validation/rate accounting")
    transform = partial(restore_for_validation, policy=args.normalization, model=model,
                        detailed_channels=args.channel_diagnostics) if args.physical_validation else None
    detailed_evaluator = partial(evaluate_detailed, load_valid_mask=mask_loader, fidelity_transform=transform)
    trainable_parameters = configure_trainable(model, args.trainable)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    refresh = CodebookRefresh(model, optimizer, args.seed) if args.codebook_refresh == "dead-v1" else None
    captured_latent = {}
    if refresh:
        # Hook records only TRAIN forwards and stores a detached tensor, not a graph.
        def capture_latent(module, inputs, output):
            if model.training:
                captured_latent["value"] = output.detach()
        model.encoder.register_forward_hook(capture_latent)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    best_loss = float("inf")
    best_epoch = 0
    best_macro_rmse = float("inf")
    best_macro_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        saved_state = resumed.get("finetune", {})
        saved_geometry = checkpoint_geometry(resumed)
        if saved_state.get("domain_family") != args.domain_family:
            raise ValueError("Resume must retain domain family")
        if saved_state.get("codebook_refresh", "none") != args.codebook_refresh:
            raise ValueError("Resume must retain codebook refresh policy")
        if architecture(resumed.get("config", {})) != architecture(config):
            raise ValueError("Resume must retain architecture")
        for key in ("codebook_diagnostics", "channel_diagnostics"):
            if saved_state.get(key, False) != getattr(args, key):
                raise ValueError(f"Resume must retain {key}")
        if explicit_geometry and args.latent_geometry != saved_geometry:
            raise ValueError("Resume must retain latent geometry")
        args.latent_geometry = saved_geometry
        apply_geometry(model, args.latent_geometry)
        if saved_state.get("trainable", "full") != args.trainable:
            raise ValueError("Resume must retain trainable policy")
        if saved_state.get("early_stopping_metric", "val-loss") != args.early_stopping_metric:
            raise ValueError("Resume must retain early-stopping metric")
        if saved_state.get("physical_validation", False) != args.physical_validation:
            raise ValueError("Resume must retain physical-validation")
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if refresh:
            refresh.load_state_dict(saved_state["codebook_refresh_state"])
        saved_state = resumed.get("finetune", {})
        if saved_state.get("normalization", "legacy") != args.normalization:
            raise ValueError("Resume must retain normalization; start a fresh branch to change it")
        if saved_state.get("sampling", "window") != args.sampling:
            raise ValueError("Resume must keep the checkpoint's sampling policy")
        if saved_state.get("spectral_weight", 0.0) != args.spectral_weight:
            raise ValueError("Resume must keep spectral weight; use a new experiment for a different objective")
        if (saved_state.get("channel_mse_weight", 0.0) != args.channel_mse_weight
                or (args.channel_mse_weight and saved_state.get("channel_weight_cap", CHANNEL_WEIGHT_CAP) != CHANNEL_WEIGHT_CAP)):
            raise ValueError("Resume must keep channel MSE objective policy")
        if "validation_details" in saved_state and saved_state["validation_details"] != args.validation_details:
            raise ValueError("Resume must keep validation-details setting")
        if saved_state.get("evaluate_initial", False) != args.evaluate_initial:
            raise ValueError("Resume must keep evaluate-initial setting")
        if saved_state.get("validation_hash", validation_hash) != validation_hash:
            raise ValueError("Resume validation window set differs from checkpoint")
        start_epoch = int(resumed.get("epoch", 0)) + 1
        best_loss = float(saved_state.get("best_val_loss", best_loss))
        best_epoch = int(saved_state.get("best_epoch", best_epoch))
        best_macro_rmse = float(saved_state.get("best_macro_rmse", best_macro_rmse))
        best_macro_epoch = int(saved_state.get("best_macro_epoch", best_macro_epoch))
        epochs_without_improvement = int(saved_state.get("epochs_without_improvement", 0))
        rng_state = saved_state.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(torch.as_tensor(rng_state["torch"], dtype=torch.uint8, device="cpu"))
            if device.type == "cuda" and rng_state.get("cuda"):
                torch.cuda.set_rng_state_all([torch.as_tensor(state, dtype=torch.uint8, device="cpu") for state in rng_state["cuda"]])
        print(f"[finetune] resuming: {args.resume} from epoch {start_epoch}", flush=True)
    end_epoch = start_epoch + args.epochs - 1
    def checkpoint_payload(epoch, metrics, steps):
        return {
            **checkpoint, "config": {**config, "latent_geometry": args.latent_geometry},
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
            "finetune": {
                "base_checkpoint": str(args.checkpoint), "samples_dir": str(args.samples_dir),
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "sampling": args.sampling, "steps_per_epoch": steps,
                "spectral_weight": args.spectral_weight, "validation_details": args.validation_details,
                "channel_mse_weight": args.channel_mse_weight, "channel_weight_cap": CHANNEL_WEIGHT_CAP,
                "evaluate_initial": args.evaluate_initial,
                "normalization": args.normalization,
                "trainable": args.trainable, "early_stopping_metric": args.early_stopping_metric,
                "physical_validation": args.physical_validation,
                "latent_geometry": args.latent_geometry,
                "codebook_diagnostics": args.codebook_diagnostics, "channel_diagnostics": args.channel_diagnostics,
                "codebook_refresh": args.codebook_refresh,
                "domain_family": args.domain_family,
                "codebook_refresh_state": refresh.state_dict() if refresh else None,
                "trainable_parameters": sum(p.numel() for p in trainable_parameters),
                "validation_hash": validation_hash, "val_samples_per_file": args.val_samples_per_file,
                "best_macro_rmse": best_macro_rmse, "best_macro_epoch": best_macro_epoch,
                "best_val_loss": best_loss, "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement, "metrics": metrics,
                "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
            },
        }
    print(f"[finetune] starting: train={len(train_paths)} val={len(val_paths)} device={device} epochs={start_epoch}-{end_epoch}", flush=True)
    with metrics_path.open("a" if args.resume else "w", newline="") as stream:
        fields = ["epoch", "elapsed_sec", "train_loss", "val_loss", "val_correlation"]
        if args.codebook_diagnostics:
            fields += ["train_temporal_loss", "train_vq_loss", "train_mean_perplexity", "train_unique_codes", "train_code_usage"]
        if args.channel_mse_weight:
            fields += ["train_channel_mse"]
        if refresh:
            fields += ["refreshed_codes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        if args.evaluate_initial and not args.resume:
            initial, initial_groups = detailed_evaluator(model, val_paths, metadata, device, sample_loader, args.spectral_weight)
            best_loss = initial["val_loss"]
            best_macro_rmse = initial["macro_relative_rmse"]
            append_metrics(args.output_dir / "validation_metrics.csv", [{"epoch": 0, **initial}])
            append_metrics(args.output_dir / "validation_groups.csv", [{"epoch": 0, **group} for group in initial_groups])
            initial_checkpoint = checkpoint_payload(0, initial, args.steps_per_epoch or len(train_paths))
            save_checkpoint(initial_checkpoint, args.output_dir / "best.pt")
            save_checkpoint(initial_checkpoint, args.output_dir / "best_macro.pt")
            save_checkpoint(initial_checkpoint, args.output_dir / "last.pt")
            print(f"[validation] epoch=0 macro_rRMSE={best_macro_rmse:.5f} val_loss={best_loss:.5f}; initialization retained as candidate", flush=True)
        for epoch in range(start_epoch, end_epoch + 1):
            started = time.perf_counter()
            model.train()
            if args.trainable == "decoder":
                model.encoder.eval()
                model.quantizer.eval()
            training_epoch = epoch_paths(train_paths, groups, args.sampling, args.steps_per_epoch)
            train_losses: list[float] = []
            channel_losses = []
            if args.codebook_diagnostics:
                used_codes = torch.zeros(model.num_embeddings, dtype=torch.bool, device=device)
                temporal_losses, vq_losses, perplexities = [], [], []
            if refresh:
                refresh.begin_epoch(epoch)
            progress = tqdm(training_epoch, desc=f"[finetune] epoch {epoch}/{end_epoch}", unit="window")
            for path in progress:
                tensor = torch.from_numpy(sample_loader(path)).unsqueeze(0).unsqueeze(0).to(device)
                optimizer.zero_grad(set_to_none=True)
                recon, vq_loss, indices, perplexity = model(tensor)
                if refresh:
                    refresh.observe(captured_latent.pop("value"), indices)
                if mask_loader:
                    mask = torch.from_numpy(mask_loader(path)).to(device)[None, None]
                    loss = 0.5 * F.mse_loss(recon[mask], tensor[mask]) + 0.5 * F.l1_loss(recon[mask], tensor[mask]) + vq_loss
                else:
                    loss = 0.5 * F.mse_loss(recon, tensor) + 0.5 * F.l1_loss(recon, tensor) + vq_loss
                if args.codebook_diagnostics:
                    used_codes[indices.reshape(-1)] = True
                    temporal_losses.append(float((loss - vq_loss).detach().item()))
                    vq_losses.append(float(vq_loss.detach().item()))
                    perplexities.append(float(perplexity.detach().item()))
                if args.spectral_weight:
                    loss = loss + args.spectral_weight * multiresolution_spectral_loss(recon, tensor)
                if args.channel_mse_weight:
                    channel_loss = bounded_channel_mse(recon, tensor, mask)
                    loss = loss + args.channel_mse_weight * channel_loss
                    channel_losses.append(float(channel_loss.detach().item()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
                progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{np.mean(train_losses):.4f}")
            # Refresh only after training, before validation/checkpoint selection.
            # Thus saved scores measure the exact saved, possibly refreshed codebook.
            refreshed_codes = refresh.refresh() if refresh else 0
            detailed = None
            if args.validation_details:
                detailed, val_groups = detailed_evaluator(model, val_paths, metadata, device, sample_loader, args.spectral_weight)
                val_loss, val_corr = detailed["val_loss"], detailed["correlation"]
                append_metrics(args.output_dir / "validation_metrics.csv", [{"epoch": epoch, **detailed}])
                append_metrics(args.output_dir / "validation_groups.csv", [{"epoch": epoch, **group} for group in val_groups])
                print(f"[validation] macro_rRMSE={detailed['macro_relative_rmse']:.5f} macro_corr={detailed['macro_correlation']:.4f} worst_family_rRMSE={detailed['worst_family_relative_rmse']:.5f} fft_error={detailed['macro_fft_magnitude_relative_error']:.4f}", flush=True)
            else:
                val_loss, val_corr = evaluate(model, val_paths, device)
            row = {"epoch": epoch, "elapsed_sec": time.perf_counter() - started, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "val_correlation": val_corr}
            if args.channel_mse_weight:
                row["train_channel_mse"] = float(np.mean(channel_losses))
            if args.codebook_diagnostics:
                unique = int(used_codes.sum().item())
                row.update(train_temporal_loss=float(np.mean(temporal_losses)), train_vq_loss=float(np.mean(vq_losses)),
                           train_mean_perplexity=float(np.mean(perplexities)), train_unique_codes=unique,
                           train_code_usage=unique / model.num_embeddings)
            if refresh:
                row["refreshed_codes"] = refreshed_codes
            writer.writerow(row)
            stream.flush()
            print(f"[finetune] epoch={epoch} elapsed={row['elapsed_sec']:.1f}s train_loss={row['train_loss']:.5f} val_loss={val_loss:.5f} val_corr={val_corr:.4f}", flush=True)
            is_new_best = val_loss < best_loss
            significant_improvement = val_loss < best_loss - args.early_stopping_min_delta
            if args.early_stopping_metric == "macro-rmse":
                significant_improvement = detailed["macro_relative_rmse"] < best_macro_rmse - args.early_stopping_min_delta
            is_new_macro = detailed is not None and detailed["macro_relative_rmse"] < best_macro_rmse
            if is_new_macro:
                best_macro_rmse = detailed["macro_relative_rmse"]
                best_macro_epoch = epoch
            if is_new_best:
                best_loss = val_loss
                best_epoch = epoch
            if significant_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            saved = checkpoint_payload(epoch, row, len(training_epoch))
            save_checkpoint(saved, args.output_dir / "last.pt")
            if is_new_macro:
                save_checkpoint(saved, args.output_dir / "best_macro.pt")
            if is_new_best:
                save_checkpoint(saved, args.output_dir / "best.pt")
            else:
                if args.early_stopping_metric == "val-loss" and epochs_without_improvement >= args.early_stopping_patience:
                    print(f"[finetune] early stopping: no validation-loss improvement of {args.early_stopping_min_delta:g} for {args.early_stopping_patience} epoch(s)", flush=True)
                    break
            if args.early_stopping_metric == "macro-rmse" and epochs_without_improvement >= args.early_stopping_patience:
                print(f"[finetune] early stopping: no macro-RMSE improvement of {args.early_stopping_min_delta:g} for {args.early_stopping_patience} epoch(s)", flush=True)
                break
    summary = {"base_checkpoint": str(args.checkpoint), "resume": str(args.resume) if args.resume else None, "best_val_loss": best_loss, "best_epoch": best_epoch, "epochs_requested_this_run": args.epochs, "epochs_completed": epoch, "early_stopping_patience": args.early_stopping_patience, "early_stopping_min_delta": args.early_stopping_min_delta, "train_samples": len(train_paths), "val_samples": len(val_paths), "sampling": args.sampling, "steps_per_epoch": len(training_epoch), "spectral_weight": args.spectral_weight, "validation_details": args.validation_details, "validation_hash": validation_hash, "val_samples_per_file": args.val_samples_per_file}
    if args.validation_details:
        summary.update(best_macro_rmse=best_macro_rmse, best_macro_epoch=best_macro_epoch)
    summary["evaluate_initial"] = args.evaluate_initial
    summary["normalization"] = args.normalization
    summary["trainable"] = args.trainable
    summary["early_stopping_metric"] = args.early_stopping_metric
    summary["physical_validation"] = args.physical_validation
    summary["latent_geometry"] = args.latent_geometry
    summary["architecture"] = architecture(config)
    summary["codebook_diagnostics"] = args.codebook_diagnostics
    summary["channel_diagnostics"] = args.channel_diagnostics
    summary["codebook_refresh"] = args.codebook_refresh
    summary["domain_family"] = args.domain_family
    summary["channel_mse_weight"] = args.channel_mse_weight
    summary["channel_weight_cap"] = CHANNEL_WEIGHT_CAP
    summary["validation_objective_excludes_channel_aux"] = True
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[finetune] success: best checkpoint -> {args.output_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[finetune] interrupted: the last fully completed epoch remains in last.pt", flush=True)
        raise SystemExit(130)
