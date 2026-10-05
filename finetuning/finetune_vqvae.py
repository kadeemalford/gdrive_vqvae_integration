#!/usr/bin/env python3
"""Fine-tune the existing VQ-VAE on native FEBUS windows without changing architecture."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .sampling import epoch_paths, sample_groups
except ImportError:
    from sampling import epoch_paths, sample_groups


def load_sample(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        data = np.asarray(payload["data"], dtype=np.float32)
    # FEBUS spikes can overflow float32 reductions before normalization.
    mean = float(np.mean(data, dtype=np.float64))
    std = float(np.std(data, dtype=np.float64))
    return ((data.astype(np.float64) - mean) / (std + 1e-6)).astype(np.float32)


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
    parser.add_argument("--sampling", choices=["window", "group", "family-group"], default="window")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 uses the number of training windows; other values are update budgets.")
    args = parser.parse_args()
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.repo_root))
    from vqvae.evaluate_vqvae import build_model, load_checkpoint_config

    train_paths = sorted((args.samples_dir / "train").glob("*.npz"))
    val_paths = sorted((args.samples_dir / "val").glob("*.npz"))
    selection_rng = random.Random(args.seed)
    if args.max_train_samples and args.max_train_samples < len(train_paths):
        train_paths = sorted(selection_rng.sample(train_paths, args.max_train_samples))
    if args.max_val_samples and args.max_val_samples < len(val_paths):
        val_paths = sorted(selection_rng.sample(val_paths, args.max_val_samples))
    if not train_paths or not val_paths:
        raise ValueError("Both train and val sample directories must be non-empty")
    if args.steps_per_epoch < 0 or args.epochs < 1:
        raise ValueError("Epochs must be positive and steps-per-epoch nonnegative")
    groups = sample_groups(train_paths, args.samples_dir / "samples_manifest.csv") if args.sampling != "window" else {}
    device = torch.device(args.device)
    checkpoint, config = load_checkpoint_config(args.checkpoint, device)
    model = build_model(config, checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        saved_state = resumed.get("finetune", {})
        if saved_state.get("sampling", "window") != args.sampling:
            raise ValueError("Resume must keep the checkpoint's sampling policy")
        start_epoch = int(resumed.get("epoch", 0)) + 1
        best_loss = float(saved_state.get("best_val_loss", best_loss))
        best_epoch = int(saved_state.get("best_epoch", best_epoch))
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
    print(f"[finetune] starting: train={len(train_paths)} val={len(val_paths)} device={device} epochs={start_epoch}-{end_epoch}", flush=True)
    with metrics_path.open("a" if args.resume else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "elapsed_sec", "train_loss", "val_loss", "val_correlation"])
        if not args.resume:
            writer.writeheader()
        for epoch in range(start_epoch, end_epoch + 1):
            started = time.perf_counter()
            model.train()
            training_epoch = epoch_paths(train_paths, groups, args.sampling, args.steps_per_epoch)
            train_losses: list[float] = []
            progress = tqdm(training_epoch, desc=f"[finetune] epoch {epoch}/{end_epoch}", unit="window")
            for path in progress:
                tensor = torch.from_numpy(load_sample(path)).unsqueeze(0).unsqueeze(0).to(device)
                optimizer.zero_grad(set_to_none=True)
                recon, vq_loss, _, _ = model(tensor)
                loss = 0.5 * F.mse_loss(recon, tensor) + 0.5 * F.l1_loss(recon, tensor) + vq_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
                progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{np.mean(train_losses):.4f}")
            val_loss, val_corr = evaluate(model, val_paths, device)
            row = {"epoch": epoch, "elapsed_sec": time.perf_counter() - started, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "val_correlation": val_corr}
            writer.writerow(row)
            stream.flush()
            print(f"[finetune] epoch={epoch} elapsed={row['elapsed_sec']:.1f}s train_loss={row['train_loss']:.5f} val_loss={val_loss:.5f} val_corr={val_corr:.4f}", flush=True)
            is_new_best = val_loss < best_loss
            significant_improvement = val_loss < best_loss - args.early_stopping_min_delta
            if is_new_best:
                best_loss = val_loss
                best_epoch = epoch
            if significant_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            saved = {
                **checkpoint,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "finetune": {
                    "base_checkpoint": str(args.checkpoint), "samples_dir": str(args.samples_dir),
                    "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                    "sampling": args.sampling, "steps_per_epoch": len(training_epoch),
                    "best_val_loss": best_loss, "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement, "metrics": row,
                    "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                },
            }
            torch.save(saved, args.output_dir / "last.pt")
            if is_new_best:
                torch.save(saved, args.output_dir / "best.pt")
            else:
                if epochs_without_improvement >= args.early_stopping_patience:
                    print(f"[finetune] early stopping: no validation-loss improvement of {args.early_stopping_min_delta:g} for {args.early_stopping_patience} epoch(s)", flush=True)
                    break
    (args.output_dir / "summary.json").write_text(json.dumps({"base_checkpoint": str(args.checkpoint), "resume": str(args.resume) if args.resume else None, "best_val_loss": best_loss, "best_epoch": best_epoch, "epochs_requested_this_run": args.epochs, "epochs_completed": epoch, "early_stopping_patience": args.early_stopping_patience, "early_stopping_min_delta": args.early_stopping_min_delta, "train_samples": len(train_paths), "val_samples": len(val_paths), "sampling": args.sampling, "steps_per_epoch": len(training_epoch)}, indent=2) + "\n")
    print(f"[finetune] success: best checkpoint -> {args.output_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[finetune] interrupted: the last fully completed epoch remains in last.pt", flush=True)
        raise SystemExit(130)
