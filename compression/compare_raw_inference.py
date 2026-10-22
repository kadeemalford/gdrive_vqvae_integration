#!/usr/bin/env python3
"""Run paired raw-file inference with immutable outputs and shared diagnostics."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.physical_validation import ordinary_fidelity


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def compare_arrays(paths, output, name):
    arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as packet:
            arrays.append((packet["raw"][0], packet["recon"][0],
                           json.loads(str(packet["metadata_json"].item()))))
    raw, baseline, metadata = arrays[0]
    np.testing.assert_array_equal(raw, arrays[1][0])
    dense = arrays[1][1]
    dt, dx = metadata["dt_s"], metadata["dx_m"]
    extent = [0, (raw.shape[1]-1)*dt, metadata["distance_start_m"],
              metadata["distance_start_m"]+(raw.shape[0]-1)*dx]
    limit = max(float(np.quantile(np.abs(raw), .99)), 1e-12)
    # Display decimation only; scores and saved reconstructions use all samples.
    step = max(1, raw.shape[1] // 1800)
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for ax, values, title in zip(axes[0], (raw, baseline, dense),
                                 ("Original", "Offshore sem fine-tuning", "FEBUS 4× · epoch 40")):
        image = ax.imshow(values[:, ::step], origin="lower", aspect="auto", extent=extent,
                          cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(title=title, xlabel="Tempo concatenado (s)", ylabel="Distância (m)")
        figure.colorbar(image, ax=ax, label="Strain rate (nStrain/s)")
    # Choose channel and waveform interval using the original alone, never model error.
    channel = int(np.argmax(np.std(raw.astype(np.float64), axis=1)))
    center = int(np.argmax(np.abs(raw[channel])))
    half = max(1, round(.005 / dt))
    start, end = max(0, center-half), min(raw.shape[1], center+half+1)
    for values, label in ((raw, "Original"), (baseline, "Offshore"), (dense, "FEBUS 4×")):
        axes[1, 0].plot(np.arange(start, end)*dt, values[channel, start:end], label=label, alpha=.85)
    axes[1, 0].set(title=f"Canal {channel}: maior desvio padrão do original", xlabel="Tempo (s)", ylabel="nStrain/s")
    axes[1, 0].legend()
    for values, label in ((raw, "Original"), (baseline, "Offshore"), (dense, "FEBUS 4×")):
        rms = np.sqrt(np.mean(values.astype(np.float64)**2, axis=1))
        axes[1, 1].plot(metadata["distance_start_m"]+np.arange(raw.shape[0])*dx, rms, label=label)
    axes[1, 1].set(title="Amplitude RMS por posição", xlabel="Distância (m)", ylabel="RMS (nStrain/s)")
    axes[1, 1].legend()
    for values, label in ((raw, "Original"), (baseline, "Offshore"), (dense, "FEBUS 4×")):
        trace = values[channel].astype(np.float64)
        spectrum = np.abs(np.fft.rfft((trace-trace.mean())*np.hanning(trace.size))) / trace.size
        axes[1, 2].semilogy(np.fft.rfftfreq(trace.size, dt), np.maximum(spectrum, 1e-12), label=label)
    axes[1, 2].set(title=f"Espectro · mesmo canal {channel}", xlabel="Frequência (Hz)", ylabel="Magnitude")
    axes[1, 2].legend()
    for ax in axes[1]:
        ax.grid(alpha=.25)
    figure.suptitle(f"{name} · mesma entrada e escala · não é comparação à mesma taxa")
    figure.savefig(output / f"{name}_paired_comparison.png", dpi=160)
    plt.close(figure)
    return [ordinary_fidelity(raw, pred, np.isfinite(raw)) for pred in (baseline, dense)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/offshore_vs_4x_raw.json"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2)+"\n")
    checkpoints = {
        "offshore": ROOT / "models/febus_finetune_base/best.pt",
        "febus_4x": ROOT / "results/experiments/robust_rate_continuation_20261018/rate_4x/best_macro.pt",
    }
    sources = {str(ROOT / "data/raw" / item["local_filename"]):
               sha256(ROOT / "data/raw" / item["local_filename"]) for item in config["files"]}
    extraction = {path: json.loads((ROOT / path).read_text()) for path in config["extraction_configs"]}
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_updates": 0, "input_sha256": sources,
        "checkpoints": {label: {"path": str(path), "sha256": sha256(path)} for label, path in checkpoints.items()},
        "extraction_configs": extraction,
        "selection": "Retained best macro-fidelity fold-0 4x checkpoint, epoch 40; not per-file selection.",
        "scope": "Two requested single-mode files; known historical held-out configurations of fold 0, not new independent acquisitions or a generalist acceptance test.",
        "comparison": "Same extracted float32 data and 1000-sample chunk boundaries; each model uses checkpoint-native normalization and geometry. Not rate matched.",
        "nonfinite_policy": "Remove entirely nonfinite channels identically before both models, preserving source channel indices. Reject partial missingness or noncontiguous retained maps.",
        "rate": "Nominal fixed-width indices + saved normalization/escape side only. NPZ uint16 storage and whole archive sizes are separate; no entropy codec/model/container overhead included.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    for i, source_config in enumerate(config["extraction_configs"]):
        run("acquisition/extract_febus_windows.py", "--config", source_config,
            "--output-dir", output / "samples", "--manifest-output", output / f"extraction_{i}.json",
            "--drop-nonfinite-channels")
    summaries = {}
    for label, checkpoint in checkpoints.items():
        folder = output / label
        run("compression/encode_decode_samples.py", "--config", config_path, "--checkpoint", checkpoint,
            "--device", config["device"], "--samples-dir", output / "samples",
            "--output-dir", folder / "reconstructions", "--metrics-dir", folder / "metrics",
            "--summary-output", folder / "metrics/reconstruction_summary.csv")
        run("compression/compute_fidelity_metrics.py", "--recon-dir", folder / "reconstructions",
            "--output", folder / "metrics/fidelity_metrics.csv")
        run("presentation/plot_results.py", "--config", config_path,
            "--recon-dir", folder / "reconstructions", "--output-dir", folder / "figures",
            "--summary", folder / "metrics/reconstruction_summary.csv")
        with (folder / "metrics/per_sample_metrics.csv").open() as stream:
            summaries[label] = {row["file_name"]: row for row in csv.DictReader(stream)}
    figures = output / "comparison_figures"
    figures.mkdir()
    rows = []
    for item in config["files"]:
        stem = Path(item["local_filename"]).stem
        physical = compare_arrays([output / label / "reconstructions" / f"{stem}.npz" for label in checkpoints], figures, stem)
        for (label, checkpoint), metrics in zip(checkpoints.items(), physical):
            record = summaries[label][item["local_filename"]]
            rows.append({"file": item["local_filename"], "model": label,
                         **{key: record[key] for key in ("correlation", "rmse_relative", "rmse", "num_indices",
                            "fixed_index_bits", "side_bytes", "index_plus_side_nominal_ratio", "normalization", "latent_geometry")},
                         **metrics})
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2)+"\n")
    report = ["# Inferência pareada: offshore original × FEBUS 4×", "",
              "Sem treinamento. Mesmos recortes e chunks de 1.000 amostras; normalização/geometria próprias de cada checkpoint.",
              "200 Hz: 545 canais finitos, 3 s, 20–128,8 m; 3.000 Hz: 111 canais, 2 s, 20–130 m. Sample rate: 20 kHz.",
              "Seis canais inteiramente não finitos (índices 545–550 do arquivo de 200 Hz) são removidos igualmente; mapa original preservado nos metadados.", "",
              "| Arquivo | Modelo | Correlação | RMSE relativo (RMS bruto) | RMSE relativo centrado | p90 por canal | Compressão nominal |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['file']} | {row['model']} | {float(row['correlation']):.6f} | {float(row['rmse_relative']):.6f} | {row['ordinary_global_centered_relative_rmse']:.6f} | {row['ordinary_channel_p90_relative_rmse']:.6f} | {float(row['index_plus_side_nominal_ratio']):.3f}× |")
    report += ["", "## Interpretação e limites", "",
               "- A comparação mede o sistema original versus o sistema adaptado mais denso; não isola fine-tuning de geometria, normalização e bitrate.",
               "- Métricas calculadas em todas as amostras dos recortes; não são as médias macro por família dos estudos de desenvolvimento.",
               "- O RMSE relativo da pipeline usa RMS bruto como denominador; o RMSE centrado e o p90 por canal estão em comparison.csv.",
               "- Taxa nominal: índices a 11 bits + side. Arquivos NPZ guardam índices uint16 e outros dados de diagnóstico; seu tamanho não é a taxa nominal.",
               "- As duas configurações pertencem ao fold 0 reservado; já houve inspeções históricas. Esta comparação não é validação externa independente nem promoção generalista.",
               "- Extração mantém [500:20500] de cada bloco e concatena os trechos. FFT de todo o recorte pode refletir emendas; não implica continuidade física auditada nas junções.",
               "- Figuras pareadas usam a mesma escala e canal escolhido só pelo original. Decimação é apenas visual; não altera inferência ou métricas.",
               "", "Veja protocol.json para hashes e critérios; comparison_figures/ para painéis e offshore/figures e febus_4x/figures para diagnósticos individuais."]
    (output / "REPORT.md").write_text("\n".join(report)+"\n")
    assert all(sha256(Path(path)) == digest for path, digest in sources.items())
    assert all(sha256(Path(spec["path"])) == spec["sha256"] for spec in protocol["checkpoints"].values())
    (output / "COMPLETE").write_text("Paired inference and diagnostics complete; sources and checkpoints unchanged.\n")
    print(json.dumps(rows, indent=2), flush=True)
    print(f"[compare] complete: {output}", flush=True)


if __name__ == "__main__":
    main()
