#!/usr/bin/env python3
"""Train-only memorization probe: VQ-VAE versus a genuinely trained continuous AE.

Each family gets its own one-window model. This is NOT cross-validation, a production
model, or a codec benchmark; model weights can memorize information from the target.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import random
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
    from .diagnose_reconstruction import spectral_metrics
    from .evaluate_checkpoints import metrics
    from .finetune_vqvae import load_sample
except ImportError:
    from latent_geometry import require_standard_geometry
    from analyze_sampling_results import write_csv
    from diagnose_reconstruction import spectral_metrics
    from evaluate_checkpoints import metrics
    from finetune_vqvae import load_sample

ROOT = Path(__file__).resolve().parents[1]


def select_cases(manifest: Path, seed: int) -> list[dict]:
    """One train window/family: uniform configuration, file, then window, using metadata only."""
    with manifest.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    families = defaultdict(lambda: defaultdict(list))
    for row in rows:
        families[row["family"]][row["configuration"]].append(row)
    if set(families) != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Probe requires all four families in the training split")
    rng = random.Random(seed)
    selected = []
    for family in sorted(families):
        configuration = rng.choice(sorted(families[family]))
        files = defaultdict(list)
        for row in families[family][configuration]:
            files[row["drive_id"]].append(row)
        file_id = rng.choice(sorted(files))
        selected.append(rng.choice(sorted(files[file_id], key=lambda row: row["sample_path"])))
    return selected


def probe_forward(model, tensor, mode):
    if mode == "vq":
        return model(tensor)
    if mode != "continuous":
        raise ValueError(f"Unknown probe mode: {mode}")
    height, width = tensor.shape[-2:]
    multiple = 2 ** model.downsample_levels
    padded = F.pad(tensor, (0, (-width) % multiple, 0, (-height) % multiple))
    latent = model.encoder(padded)
    recon = model.decoder(latent)[..., :height, :width]
    return recon, tensor.new_zeros(()), None, None


def nominal_rates(channels, time_samples, levels, embeddings, embedding_dim):
    multiple = 2 ** levels
    positions = math.ceil(channels / multiple) * math.ceil(time_samples / multiple)
    bits_per_index = math.ceil(math.log2(embeddings))
    input_bits = channels * time_samples * 32
    return {"latent_positions": positions, "index_bits": bits_per_index,
            "vq_index_only_ratio": input_bits / (positions * bits_per_index),
            "continuous_float32_latent_ratio": input_bits / (positions * embedding_dim * 32)}


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def measure(model, tensor, values, mode):
    model.eval()
    with torch.no_grad():
        recon, vq_loss, indices, perplexity = probe_forward(model, tensor, mode)
        decoded = recon[0, 0].cpu().numpy()
        temporal = 0.5 * F.mse_loss(recon, tensor) + 0.5 * F.l1_loss(recon, tensor)
        row = {**metrics(values, decoded), **spectral_metrics(values, decoded),
               "mse": float(F.mse_loss(recon, tensor).item()), "temporal_loss": float(temporal.item()),
               "vq_loss": float(vq_loss.item()), "objective": float((temporal + vq_loss).item()),
               "code_usage": float(indices.unique().numel() / model.num_embeddings) if indices is not None else None,
               "perplexity": float(perplexity.item()) if perplexity is not None else None}
    return row, decoded


def run_arm(case, mode, source, output, protocol, device, build_model, load_checkpoint_config):
    folder = output / case["family"] / mode
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "COMPLETE").exists():
        return json.loads((folder / "summary.json").read_text())
    checkpoint, config = load_checkpoint_config(source, device)
    require_standard_geometry(checkpoint)
    torch.manual_seed(protocol["seed"])
    model = build_model(config, checkpoint, device)
    del checkpoint
    if mode == "continuous":
        for parameter in model.quantizer.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=protocol["learning_rate"], weight_decay=0)
    path = ROOT / case["sample_path"]
    values = load_sample(path)
    tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0).to(device)
    history_path = folder / "history.csv"
    last_path = folder / "probe_last.pt"
    start = 0
    best_rmse, best_step = float("inf"), 0
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["probe_mode"] != mode or saved["protocol"] != protocol or saved["case"] != case:
            raise ValueError(f"Resume state mismatch: {last_path}")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start = saved["step"]
        best_rmse, best_step = saved["best_rmse"], saved["best_step"]
        if (not history_path.exists() and start != 0) or not (folder / "probe_best.pt").exists():
            raise ValueError(f"Incomplete probe artifacts: {folder}")
    elif any(folder.iterdir()):
        raise ValueError(f"Nonempty arm without checkpoint: {folder}")
    rates = nominal_rates(values.shape[0], values.shape[1], model.downsample_levels,
                          model.num_embeddings, model.embedding_dim)

    def record(step, elapsed):
        nonlocal best_rmse, best_step
        measured, _ = measure(model, tensor, values, mode)
        if not math.isfinite(measured["relative_rmse"]):
            raise ValueError("Non-finite reconstruction metrics")
        is_best = measured["relative_rmse"] < best_rmse
        if is_best:
            best_rmse, best_step = measured["relative_rmse"], step
        payload = {"probe_only": True, "probe_mode": mode, "case": case, "protocol": protocol,
                   "config": config, "step": step, "best_rmse": best_rmse, "best_step": best_step,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
        # Write best before last so resumed state never refers to a missing best file.
        if is_best:
            atomic_save(payload, folder / "probe_best.pt")
        atomic_save(payload, last_path)
        row = {"step": step, "elapsed_sec": elapsed, **measured}
        with history_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"[capacity] {case['family']} {mode} step={step} rRMSE={measured['relative_rmse']:.4f} corr={measured['correlation']:.4f} temporal={measured['temporal_loss']:.4f} VQ={measured['vq_loss']:.4f}", flush=True)

    # Repair an interruption between last checkpoint and history append, deterministically.
    if start or last_path.exists():
        history = []
        if history_path.exists():
            with history_path.open() as stream:
                history = list(csv.DictReader(stream))
        if history and int(history[-1]["step"]) > start:
            raise ValueError("History is ahead of saved state; refusing ambiguous resume")
        if not history or int(history[-1]["step"]) < start:
            record(start, 0)
    else:
        record(0, 0)
    started = time.perf_counter()
    progress = tqdm(range(start + 1, protocol["steps"] + 1), desc=f"[capacity] {case['family']} {mode}", unit="update")
    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        recon, vq_loss, _, _ = probe_forward(model, tensor, mode)
        loss = 0.5 * F.mse_loss(recon, tensor) + 0.5 * F.l1_loss(recon, tensor) + vq_loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite probe objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % protocol["checkpoint_every"] == 0 or step == protocol["steps"]:
            record(step, time.perf_counter() - started)
    best = torch.load(folder / "probe_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    best_metrics, decoded = measure(model, tensor, values, mode)
    np.savez_compressed(folder / "reconstruction.npz", original=values, reconstruction=decoded)
    summary = {"case": case, "mode": mode, "steps": protocol["steps"], "best_step": best_step,
               "best_metrics": best_metrics, "nominal_rates": rates, "production_model": False}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (folder / "COMPLETE").write_text("Train-only one-window memorization complete. Not a production model.\n")
    return summary


def plot_case(case, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    family = case["family"]
    arrays = {}
    for mode in ("vq", "continuous"):
        with np.load(output / family / mode / "reconstruction.npz") as payload:
            arrays[mode] = payload["reconstruction"]
            original = payload["original"]
    limit = max(float(np.percentile(np.abs(original), 99)), 1e-6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, array, title in zip(axes[0], (original, arrays["vq"], arrays["continuous"]),
                                ("Original normalizado", "VQ memorizado", "AE contínuo treinado")):
        ax.imshow(array, aspect="auto", cmap="seismic", vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xlabel("Amostra temporal")
    for ax, mode in zip(axes[1, :2], ("vq", "continuous")):
        ax.imshow(original - arrays[mode], aspect="auto", cmap="seismic", vmin=-limit, vmax=limit)
        ax.set_title(f"Erro {mode}")
    for label, array in (("original", original), *arrays.items()):
        power = np.mean(np.abs(np.fft.rfft(array, axis=-1)) ** 2, axis=0)
        axes[1, 2].semilogy(np.fft.rfftfreq(array.shape[-1], float(case["dt_s"])), power + 1e-12, label=label)
    axes[1, 2].legend()
    axes[1, 2].set_xlabel("Hz")
    axes[1, 2].set_title("Potência FFT média dos canais")
    fig.savefig(output / family / "comparison.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-experiment", type=Path, default=Path("results/experiments/sampling_full_20261005"))
    parser.add_argument("--fold", type=int, default=1, choices=range(4))
    parser.add_argument("--steps", type=int, default=500)
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
    selected = select_cases(ROOT / f"data/finetune/fold_{args.fold}/samples_manifest.csv", args.seed)
    protocol = {"kind": "capacity_probe", "fold": args.fold, "steps": args.steps, "seed": args.seed,
                "learning_rate": args.learning_rate, "weight_decay": 0, "checkpoint_every": 100,
                "source_checkpoint": str(source), "arms": ["vq", "continuous"], "split": "train",
                "reconstruction_objective": "0.5 MSE + 0.5 L1; VQ loss added only in vq arm",
                "selected_cases": selected, "generalization_claim": False}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/cases differ; preserve original flags")
        if (output / "COMPLETE").exists():
            print(f"[capacity] already complete: {output / 'REPORT.md'}")
            return
    elif output.exists():
        raise FileExistsError(f"{output} exists; use --resume")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another probe is running this experiment") from None
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
            write_csv(output / "cases.csv", selected)
        sys.path.insert(0, str(args.repo_root))
        from vqvae.evaluate_vqvae import build_model, load_checkpoint_config
        results = []
        for index, case in enumerate(selected):
            for arm_index, mode in enumerate(protocol["arms"]):
                print(f"\n[capacity {index * 2 + arm_index + 1}/8] {case['family']}/{case['configuration']} {mode}", flush=True)
                results.append(run_arm(case, mode, source, output, protocol, torch.device(args.device), build_model, load_checkpoint_config))
            plot_case(case, output)
        (output / "comparison.json").write_text(json.dumps({"protocol": protocol, "results": results}, indent=2) + "\n")
        lines = ["# Capacidade: memorização de uma janela de treino por família", "",
                 "Não mede generalização. AE contínuo não é um codec com a mesma taxa de bits. Modelos separados por janela, não especialistas de produção.", "",
                 "| Família | Modelo | Melhor passo | RMSE relativo | Corr. | Erro FFT | RMS ratio |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for result in results:
            m = result["best_metrics"]
            lines.append(f"| {result['case']['family']} | {result['mode']} | {result['best_step']} | {m['relative_rmse']:.4f} | {m['correlation']:.4f} | {m['fft_magnitude_relative_error']:.4f} | {m['reconstruction_rms_ratio']:.4f} |")
        lines += ["", "Inicialização é candidata no passo 0. Melhor selecionado por RMSE no próprio alvo visto.",
                  "Fórmulas de taxa são nominais, sem pesos/metadados. Pesos podem memorizar o alvo; esses números não são rate-distortion de um codec generalista.",
                  "Limites: uma janela/família, uma seed, orçamento finito, LR única; falha no teste não prova incapacidade matemática da arquitetura."]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")
        (output / "COMPLETE").write_text("Capacity probe complete. No generalization or production-model claim.\n")
    print(f"[capacity] complete: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
