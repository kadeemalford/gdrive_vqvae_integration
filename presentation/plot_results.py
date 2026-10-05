#!/usr/bin/env python3
"""Generate original-vs-reconstructed DAS diagnostics and FFT comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def matrices(values: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values[np.newaxis, ...]
    if values.ndim == 3:
        return values
    raise ValueError(f"Expected [channel,time] or [block,channel,time], got {values.shape}")


def finite_values(array: np.ndarray) -> np.ndarray:
    return array[np.isfinite(array)]


def robust_scale(array: np.ndarray, percentile: float = 99.0) -> float:
    values = finite_values(array)
    scale = float(np.percentile(np.abs(values), percentile)) if values.size else 1.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def axes_for(metadata: dict[str, object], channels: int, time_samples: int, default_dt: float) -> tuple[np.ndarray, np.ndarray]:
    dt = float(metadata.get("dt_s", default_dt) or default_dt)
    dx = float(metadata.get("dx_m", 1.0) or 1.0)
    distance_start = float(metadata.get("distance_start_m", 0.0) or 0.0)
    time = np.arange(time_samples) * dt if dt > 0 else np.arange(time_samples)
    distance = distance_start + np.arange(channels) * dx
    return time, distance


def save_image_comparison(raw: np.ndarray, recon: np.ndarray, error: np.ndarray, metadata: dict[str, object], output: Path, name: str, default_dt: float, display_min: float, display_max: float) -> None:
    time, distance = axes_for(metadata, raw.shape[0], raw.shape[1], default_dt)
    extent = [time[0], time[-1], distance[0], distance[-1]]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, values, title, cmap in zip(
        axes, (raw, recon, error), ("Original", "Reconstruído", "Erro"), ("gray", "gray", "coolwarm")
    ):
        image = axis.imshow(values, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap, vmin=display_min, vmax=display_max, extent=extent)
        axis.set_title(title); axis.set_xlabel("Tempo (s)"); axis.set_ylabel("Distância (m)")
        figure.colorbar(image, ax=axis, label="Erro (nStrain/s)" if title == "Erro" else str(metadata.get("variable", "DAS")))
    figure.suptitle(f"{name} | {metadata.get('variable', '')} | {metadata.get('frequency_hz', '')} Hz")
    figure.savefig(output / f"{name}_waterfall_comparison.png", dpi=180)
    plt.close(figure)


def channel_statistics(matrix: np.ndarray, quiet_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rms = np.sqrt(np.nanmean(np.square(matrix, dtype=np.float64), axis=1))
    noise = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for channel, values in enumerate(matrix):
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        count = max(1, int(quiet_fraction * finite.size))
        quiet = finite[np.argpartition(np.abs(finite), count - 1)[:count]]
        noise[channel] = np.std(quiet)
    snr_db = 20.0 * np.log10((rms + 1e-12) / (noise + 1e-12))
    return rms, noise, snr_db


def save_spatial_diagnostics(raw: np.ndarray, recon: np.ndarray, metadata: dict[str, object], output: Path, name: str, default_dt: float) -> None:
    raw_rms, raw_noise, raw_snr = channel_statistics(raw)
    recon_rms, recon_noise, recon_snr = channel_statistics(recon)
    _, distance = axes_for(metadata, raw.shape[0], raw.shape[1], default_dt)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, original, decoded, title, ylabel in zip(
        axes, (raw_rms, raw_noise, raw_snr), (recon_rms, recon_noise, recon_snr),
        ("RMS vs distância", "Ruído vs distância", "SNR vs distância"), ("RMS", "Desvio padrão dos 20% instantes mais calmos", "SNR (dB)"),
    ):
        axis.plot(distance, original, label="Original", linewidth=1.5)
        axis.plot(distance, decoded, label="Reconstruído", linewidth=1.5, linestyle="--")
        axis.set_title(title); axis.set_xlabel("Distância (m)"); axis.set_ylabel(ylabel); axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle(name)
    figure.savefig(output / f"{name}_spatial_diagnostics.png", dpi=180)
    plt.close(figure)


def save_waveforms(raw: np.ndarray, recon: np.ndarray, metadata: dict[str, object], output: Path, name: str, default_dt: float) -> None:
    time, distance = axes_for(metadata, raw.shape[0], raw.shape[1], default_dt)
    peak_index = int(np.nanargmax(np.abs(raw)))
    channel, time_index = divmod(peak_index, raw.shape[1])
    dt = float(metadata.get("dt_s", default_dt) or default_dt)
    half_window = max(1, int(0.02 / dt))
    start = max(0, time_index - half_window)
    end = min(raw.shape[1], time_index + half_window + 1)
    relative_time = (np.arange(start, end) - time_index) * dt
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.plot(relative_time, raw[channel, start:end], label="Original")
    axis.plot(relative_time, recon[channel, start:end], label="Reconstruído", linestyle="--")
    axis.axvline(0.0, color="red", linestyle="--", label="Pico do evento")
    axis.set_xlabel("Tempo relativo ao pico (s)")
    axis.set_ylabel(str(metadata.get("variable", "DAS")))
    axis.set_title(f"Evento impulsivo | canal {channel} / {distance[channel]:.2f} m")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.suptitle(f"Waveform comparativo | {name}")
    figure.savefig(output / f"{name}_event_waveform_comparison.png", dpi=180)
    plt.close(figure)


def fft_frequency_distance(matrix: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    time_samples = matrix.shape[1]
    window = np.hanning(time_samples)[np.newaxis, :]
    prepared = np.nan_to_num(matrix - np.nanmean(matrix, axis=1, keepdims=True)) * window
    magnitude = np.abs(np.fft.rfft(prepared, axis=1)) / time_samples
    return np.fft.rfftfreq(time_samples, d=1.0 / sample_rate), 20.0 * np.log10(magnitude.T + 1e-12)


def noise_spectrum(matrix: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    energy = np.nansum(np.square(matrix, dtype=np.float64), axis=0)
    count = max(1, int(0.2 * matrix.shape[1]))
    quiet_indices = np.sort(np.argpartition(energy, count - 1)[:count])[:4096]
    signal = matrix[matrix.shape[0] // 2, quiet_indices]
    signal = signal[np.isfinite(signal)]
    signal = signal - np.mean(signal)
    windowed = signal * np.hanning(signal.size)
    nfft = 1 << int(np.ceil(np.log2(max(1, windowed.size))))
    magnitude = np.abs(np.fft.rfft(windowed, n=nfft)) / max(1, windowed.size)
    return np.fft.rfftfreq(nfft, d=1.0 / sample_rate), magnitude


def save_fft(raw: np.ndarray, recon: np.ndarray, metadata: dict[str, object], output: Path, name: str, default_dt: float, fmin_hz: float, fmax_hz: float) -> None:
    _, distance = axes_for(metadata, raw.shape[0], raw.shape[1], default_dt)
    sample_rate = 1.0 / float(metadata.get("dt_s", default_dt) or default_dt)
    frequency_axis, raw_fft = fft_frequency_distance(raw, sample_rate)
    _, recon_fft = fft_frequency_distance(recon, sample_rate)
    _, error_fft = fft_frequency_distance(raw - recon, sample_rate)
    mask = (frequency_axis >= fmin_hz) & (frequency_axis <= min(fmax_hz, sample_rate / 2.0))
    frequency_axis = frequency_axis[mask]
    spectra = (raw_fft[mask], recon_fft[mask], error_fft[mask])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, spectrum, title, cmap in zip(axes, spectra, ("FFT original", "FFT reconstruído", "FFT do erro"), ("magma", "magma", "coolwarm")):
        image = axis.imshow(spectrum, aspect="auto", origin="lower", interpolation="nearest", cmap=cmap, extent=[distance[0], distance[-1], frequency_axis[0], frequency_axis[-1]])
        axis.set_title(title); axis.set_xlabel("Distância (m)"); axis.set_ylabel("Frequência (Hz)")
        figure.colorbar(image, ax=axis, label="Magnitude (dB)")
    figure.suptitle(f"FFT frequência-distância | {name} | janela de 3 s")
    figure.savefig(output / f"{name}_fft_frequency_distance_comparison.png", dpi=180)
    plt.close(figure)

    frequency, raw_psd = noise_spectrum(raw, sample_rate)
    _, recon_psd = noise_spectrum(recon, sample_rate)
    figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    axis.semilogy(frequency, raw_psd + 1e-12, label="Original")
    axis.semilogy(frequency, recon_psd + 1e-12, label="Reconstruído", linestyle="--")
    axis.set_xlabel("Frequência (Hz)"); axis.set_ylabel("Magnitude"); axis.set_title(f"Espectro de ruído (segmento calmo) | {name}")
    axis.grid(alpha=0.25); axis.legend()
    figure.savefig(output / f"{name}_noise_spectrum_comparison.png", dpi=180)
    plt.close(figure)


def save_metrics_table(rows: list[list[str]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, max(2.4, 0.75 * (len(rows) + 1))))
    axis.axis("off")
    headers = ["Arquivo", "Faixa espacial", "Canais válidos", "Correlação", "RMSE relativo", "MAE (nStrain/s)", "Compressão"]
    table = axis.table(cellText=rows, colLabels=headers, colWidths=[0.20, 0.14, 0.13, 0.12, 0.14, 0.16, 0.11], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for column in range(len(headers)):
        table[(0, column)].set_facecolor("#1f4e79")
        table[(0, column)].set_text_props(color="white", weight="bold")
    axis.set_title("Resumo das métricas de reconstrução", pad=16, fontsize=14, weight="bold")
    figure.savefig(output / "reconstruction_metrics_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recon-dir", type=Path, default=Path("results/reconstructions"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--summary", type=Path, default=Path("results/reconstruction_summary.csv"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    default_dt = float(config.get("default_dt_s", 5e-5))
    display_min = float(config.get("display_vmin", -robust_scale(np.array([1.0]))))
    display_max = float(config.get("display_vmax", -display_min))
    if display_min >= display_max:
        raise ValueError("display_vmin must be lower than display_vmax")
    fft_fmin = float(config.get("fft_fmin_hz", 0.0))
    fft_fmax = float(config.get("fft_fmax_hz", float("inf")))
    paths = sorted(args.recon_dir.glob("*.npz"))
    if not paths:
        raise SystemExit("Nenhuma reconstrução encontrada; execute encode_decode_samples.py primeiro.")
    print(f"[plot] starting: {len(paths)} reconstruction(s) -> {args.output_dir}")
    summary_metrics: dict[str, dict[str, str]] = {}
    summary_path = args.summary
    if summary_path.exists():
        with summary_path.open(newline="") as stream:
            summary_metrics = {row["file_name"]: row for row in csv.DictReader(stream)}
    table_rows: list[list[str]] = []
    for path in paths:
        print(f"[plot] rendering: {path}")
        with np.load(path, allow_pickle=False) as payload:
            raw = matrices(payload["raw"])
            recon = matrices(payload["recon"])
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        # Blocks are consecutive in time; preserve the spatial channel axis.
        raw, recon = np.concatenate(list(raw), axis=1), np.concatenate(list(recon), axis=1)
        valid_channels = np.isfinite(raw).any(axis=1) & np.isfinite(recon).any(axis=1)
        raw, recon = raw[valid_channels], recon[valid_channels]
        error = raw - recon
        name = Path(metadata.get("local_filename", path.name)).stem
        file_output = args.output_dir / name
        file_output.mkdir(parents=True, exist_ok=True)
        save_image_comparison(raw, recon, error, metadata, file_output, name, default_dt, display_min, display_max)
        save_spatial_diagnostics(raw, recon, metadata, file_output, name, default_dt)
        save_waveforms(raw, recon, metadata, file_output, name, default_dt)
        save_fft(raw, recon, metadata, file_output, name, default_dt, fft_fmin, fft_fmax)
        valid = np.isfinite(raw) & np.isfinite(recon)
        metrics = summary_metrics.get(metadata.get("local_filename", path.name), {})
        correlation = float(np.corrcoef(raw[valid], recon[valid])[0, 1]) if valid.any() else float("nan")
        table_rows.append([
            name,
            f"{metadata.get('distance_start_m', metadata.get('spatial_start_m', 0)):.1f}-{metadata.get('distance_end_m', metadata.get('spatial_end_m', 0)):.1f} m",
            f"{raw.shape[0]} / {metadata.get('channels', raw.shape[0])}",
            f"{correlation:.4f}",
            f"{float(metrics.get('rmse_relative', float('nan'))):.4f}",
            f"{float(metrics.get('mae', float('nan'))):,.0f}",
            f"{float(metrics.get('theoretical_ratio', float('nan'))):.1f}x",
        ])
        print(f"[plot] complete: {file_output}")
    save_metrics_table(table_rows, args.output_dir)
    print(f"[plot] success: wrote diagnostics for {len(paths)} file(s) and {args.output_dir / 'reconstruction_metrics_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
