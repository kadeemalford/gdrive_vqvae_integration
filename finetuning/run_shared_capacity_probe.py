#!/usr/bin/env python3
"""Small shared-model capacity/transfer probe; not a final model or CV estimate."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .latent_geometry import require_standard_geometry
    from .analyze_sampling_results import write_csv
    from .diagnose_reconstruction import select_validation
    from .finetune_vqvae import load_sample
    from .reconstruction_objective import aggregate_validation
    from .run_capacity_probe import ROOT, atomic_save, measure, probe_forward, select_cases
except ImportError:
    from latent_geometry import require_standard_geometry
    from analyze_sampling_results import write_csv
    from diagnose_reconstruction import select_validation
    from finetune_vqvae import load_sample
    from reconstruction_objective import aggregate_validation
    from run_capacity_probe import ROOT, atomic_save, measure, probe_forward, select_cases


def select_pool_and_controls(manifest, seed):
    """Keep the four original probe anchors; add another train file per family."""
    with manifest.open() as stream:
        available = [r for r in csv.DictReader(stream) if r["split"] == "train"]
    pool = select_cases(manifest, seed)
    rng = random.Random(seed + 1)
    for anchor in list(pool):
        candidates = [r for r in available if r["family"] == anchor["family"] and r["drive_id"] != anchor["drive_id"]]
        if not candidates:
            raise ValueError("Need two independent train files per family")
        different_config = [r for r in candidates if r["configuration"] != anchor["configuration"]]
        candidates = different_config or candidates
        configs = sorted({r["configuration"] for r in candidates})
        config = rng.choice(configs)
        candidates = [r for r in candidates if r["configuration"] == config]
        file_id = rng.choice(sorted({r["drive_id"] for r in candidates}))
        pool.append(rng.choice(sorted([r for r in candidates if r["drive_id"] == file_id], key=lambda r: r["sample_path"])))
    pool_paths = {r["sample_path"] for r in pool}
    controls = []
    for target in pool:
        candidates = [r for r in available if r["drive_id"] == target["drive_id"] and r["sample_path"] not in pool_paths]
        different_block = [r for r in candidates if r["block_index"] != target["block_index"]]
        candidates = different_block or candidates
        if not candidates:
            raise ValueError("No non-pool window available in training file")
        controls.append(rng.choice(sorted(candidates, key=lambda r: r["sample_path"])))
    return sorted(pool, key=lambda r: r["sample_path"]), sorted(controls, key=lambda r: r["sample_path"])


def check_partitions(pool, controls, validation):
    if any(r["split"] != "train" for r in pool + controls) or any(r["split"] != "val" for r in validation):
        raise ValueError("Wrong role in selected sample pool")
    pool_paths = {r["sample_path"] for r in pool}
    if pool_paths & {r["sample_path"] for r in controls}:
        raise ValueError("A control window is in the training pool")
    train_groups = {(r["family"], r["configuration"]) for r in pool + controls}
    val_groups = {(r["family"], r["configuration"]) for r in validation}
    if train_groups & val_groups:
        raise ValueError("Configuration leakage into validation")
    if {r["drive_id"] for r in pool + controls} & {r["drive_id"] for r in validation}:
        raise ValueError("File leakage into validation")


def update_schedule(count, steps, seed):
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < steps:
        cycle = list(range(count))
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return schedule[:steps]


def evaluate_split(model, cases, arrays, mode, device):
    rows = []
    samples = []
    for meta in cases:
        values = arrays[meta["sample_path"]]
        tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device)
        result, _ = measure(model, tensor, values, mode)
        numeric = {k: v for k, v in result.items() if k not in ("code_usage", "perplexity")}
        rows.append({"family": meta["family"], "configuration": meta["configuration"], **numeric})
        samples.append({"sample_path": meta["sample_path"], "drive_id": meta["drive_id"],
                        "family": meta["family"], "configuration": meta["configuration"], **numeric})
    macro, groups = aggregate_validation(rows)
    families = defaultdict(list)
    for group in groups:
        families[group["family"]].append(group)
    family_metrics = {family: {metric: float(np.mean([r[metric] for r in values]))
                              for metric in ("relative_rmse", "correlation", "fft_magnitude_relative_error", "fft_phase_alignment", "reconstruction_rms_ratio")}
                      for family, values in families.items()}
    return {"macro": macro, "groups": groups, "families": family_metrics, "samples": samples}


def run_arm(mode, output, protocol, cases, arrays, device, build_model, load_checkpoint_config):
    folder = output / mode
    folder.mkdir(exist_ok=True)
    if (folder / "COMPLETE").exists():
        return json.loads((folder / "summary.json").read_text())
    source, config = load_checkpoint_config(Path(protocol["source_checkpoint"]), device)
    require_standard_geometry(source)
    torch.manual_seed(protocol["seed"])
    model = build_model(config, source, device)
    del source
    if mode == "continuous":
        for p in model.quantizer.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=protocol["learning_rate"], weight_decay=0)
    start = 0
    best_train, best_val = float("inf"), float("inf")
    best_train_step, best_val_step = 0, 0
    saved_paths = [p for p in folder.glob("probe_*.pt") if p.name in ("probe_last.pt", "probe_best_train.pt", "probe_best_val.pt")]
    if saved_paths:
        # Best may have committed just before an interruption writing last.
        # Use the most advanced complete state, without keeping every full snapshot.
        saved = None
        for path in saved_paths:
            candidate = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if saved is None or candidate["step"] > saved["step"]:
                saved = candidate
        if saved["protocol"] != protocol or saved["probe_mode"] != mode:
            raise ValueError("Resume checkpoint protocol mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start = saved["step"]
        best_train, best_val = saved["best_train"], saved["best_val"]
        best_train_step, best_val_step = saved["best_train_step"], saved["best_val_step"]
        # Each checkpoint carries its own metrics to recover after interrupted JSON writes.
        (folder / f"metrics_{start:06d}.json").write_text(json.dumps(saved["record"], indent=2) + "\n")
        del saved
    elif any(folder.iterdir()):
        raise ValueError("Nonempty arm without checkpoint; refusing an ambiguous restart")

    def record(step, elapsed):
        nonlocal best_train, best_val, best_train_step, best_val_step
        results = {role: evaluate_split(model, selected, arrays, mode, device) for role, selected in cases.items()}
        train_rmse = results["pool"]["macro"]["macro_relative_rmse"]
        val_rmse = results["validation"]["macro"]["macro_relative_rmse"]
        if not math.isfinite(train_rmse) or not math.isfinite(val_rmse):
            raise ValueError("Non-finite probe metrics")
        is_best_train = train_rmse < best_train
        is_best_val = val_rmse < best_val
        if is_best_train:
            best_train, best_train_step = train_rmse, step
        if is_best_val:
            best_val, best_val_step = val_rmse, step
        measured = {"step": step, "elapsed_sec_this_invocation": elapsed, "splits": results}
        payload = {"probe_only": True, "probe_mode": mode, "protocol": protocol, "config": config,
                   "step": step, "best_train": best_train, "best_val": best_val,
                   "best_train_step": best_train_step, "best_val_step": best_val_step,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "record": measured}
        if is_best_train:
            atomic_save(payload, folder / "probe_best_train.pt")
        if is_best_val:
            atomic_save(payload, folder / "probe_best_val.pt")
        atomic_save(payload, folder / "probe_last.pt")
        (folder / f"metrics_{step:06d}.json").write_text(json.dumps(measured, indent=2) + "\n")
        print(f"[shared] {mode} step={step} pool_rRMSE={train_rmse:.4f} control_rRMSE={results['controls']['macro']['macro_relative_rmse']:.4f} val_rRMSE={val_rmse:.4f}", flush=True)

    if not saved_paths:
        record(0, 0)
    schedule = update_schedule(len(cases["pool"]), protocol["steps"], protocol["seed"])
    started = time.perf_counter()
    for step in tqdm(range(start + 1, protocol["steps"] + 1), desc=f"[shared] {mode}", unit="update"):
        meta = cases["pool"][schedule[step - 1]]
        values = arrays[meta["sample_path"]]
        tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        recon, vq_loss, _, _ = probe_forward(model, tensor, mode)
        loss = 0.5 * F.mse_loss(recon, tensor) + 0.5 * F.l1_loss(recon, tensor) + vq_loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % protocol["checkpoint_every"] == 0 or step == protocol["steps"]:
            record(step, time.perf_counter() - started)
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("metrics_*.json"))]
    by_step = {r["step"]: r for r in records}
    summary = {"mode": mode, "best_train_step": best_train_step, "best_val_step": best_val_step,
               "initial": by_step[0], "final": by_step[protocol["steps"]],
               "train_selected": by_step[best_train_step], "val_selected": by_step[best_val_step],
               "production_model": False}
    history = [{"step": r["step"], **{f"{role}_{metric}": values["macro"][f"macro_{metric}"]
               for role, values in r["splits"].items() for metric in ("relative_rmse", "correlation", "fft_magnitude_relative_error")}}
               for r in records]
    write_csv(folder / "history.csv", history)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (folder / "COMPLETE").write_text("Shared diagnostic probe complete; not a production model.\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, default=Path("results/experiments/sampling_full_20261005"))
    parser.add_argument("--fold", type=int, default=1, choices=range(4))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path("/home/murilo-barroso/source/smartgrids"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Steps and learning rate must be positive/finite")
    output = args.output_dir.resolve()
    source = args.source_experiment.resolve() / f"fold_{args.fold}/window/best.pt"
    if not source.exists():
        raise FileNotFoundError(source)
    manifest = ROOT / f"data/finetune/fold_{args.fold}/samples_manifest.csv"
    pool, controls = select_pool_and_controls(manifest, args.seed)
    validation = select_validation(manifest, 2, args.seed)
    check_partitions(pool, controls, validation)
    cases = {"pool": pool, "controls": controls, "validation": validation}
    protocol = {"kind": "shared_capacity_probe", "fold": args.fold, "steps": args.steps,
                "learning_rate": args.learning_rate, "weight_decay": 0, "seed": args.seed,
                "source_checkpoint": str(source), "arms": ["vq", "continuous"], "checkpoint_every": 500,
                "checkpoint_retention": "last, best_train, best_val; metric histories retained",
                "cases": cases, "pool_windows": len(pool), "validation_windows": len(validation),
                "objective": "0.5 MSE + 0.5 L1; VQ loss added only in vq arm", "production_model": False}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol differs; retain original flags")
        if (output / "COMPLETE").exists():
            print(f"[shared] already complete: {output / 'REPORT.md'}")
            return
    elif output.exists():
        raise FileExistsError(f"{output} exists; use --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 4 * 1024 ** 3:
        raise RuntimeError("Need at least 4 GiB free for retained checkpoints and atomic writes; no files were deleted")
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another probe is running this experiment") from None
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
            for role, selected in cases.items():
                write_csv(output / f"{role}_selection.csv", selected)
        arrays = {r["sample_path"]: load_sample(ROOT / r["sample_path"]) for selected in cases.values() for r in selected}
        sys.path.insert(0, str(args.repo_root))
        from vqvae.evaluate_vqvae import build_model, load_checkpoint_config
        results = {}
        for mode in protocol["arms"]:
            results[mode] = run_arm(mode, output, protocol, cases, arrays, torch.device(args.device), build_model, load_checkpoint_config)
        (output / "comparison.json").write_text(json.dumps({"protocol": protocol, "results": results}, indent=2) + "\n")
        lines = ["# Capacidade compartilhada e transferência", "",
                 "Um modelo por braço, oito janelas das quatro famílias. Diagnóstico em um fold; não é modelo final nem CV completa.",
                 "Controles = janelas não usadas na ramificação, dos mesmos arquivos de treino; podem ter sido usadas no fine-tuning de origem.", "",
                 "| Modelo | Seleção | Passo | RMSE pool | Controles | Validação |",
                 "|---|---|---:|---:|---:|---:|"]
        for mode, summary in results.items():
            for selector in ("initial", "train_selected", "val_selected", "final"):
                row = summary[selector]
                values = [row["splits"][role]["macro"]["macro_relative_rmse"] for role in cases]
                lines.append(f"| {mode} | {selector} | {row['step']} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
        lines += ["", "Seleção de produção deve usar validação, não o pool. Resultados de validação são exploratórios; os testes não foram avaliados.",
                  "AE contínuo tem outra taxa de informação e não inclui VQ na perda. Não inferir falta de dados ou incapacidade definitiva a partir deste orçamento."]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")
        plot_histories(output)
        (output / "COMPLETE").write_text("Shared probe finished. No production-model claim.\n")
    print(f"[shared] complete: {output / 'REPORT.md'}", flush=True)


def plot_histories(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, mode in zip(axes, ("vq", "continuous")):
        with (output / mode / "history.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for role in ("pool", "controls", "validation"):
            ax.plot([int(r["step"]) for r in rows], [float(r[f"{role}_relative_rmse"]) for r in rows], marker="o", label=role)
        ax.set_title(mode)
        ax.set_xlabel("Atualizações totais no modelo compartilhado")
        ax.set_ylabel("RMSE relativo macro")
        ax.legend()
    fig.savefig(output / "learning_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
