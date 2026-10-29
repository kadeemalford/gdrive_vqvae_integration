#!/usr/bin/env python3
"""Download a seeded, stratified Drive sample and audit spatial energy concentration.

Read-only signal analysis: no neural inference, filtering, or modified raw files.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acquisition.extract_febus_windows import choose_dataset
from finetuning.download_febus_raw import load_credentials


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            value.update(chunk)
    return value.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_files(rows, seed):
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    previous = {("single_mode", "Configuration 1", "Copy of 200Hz.h5"),
                ("single_mode", "Configuration 5", "Copy of 3000Hz.h5")}
    for row in rows:
        if (row["family"], row["configuration"], row["title"]) not in previous:
            groups[row["family"]][row["configuration"]].append(row)
    selected = []
    for family in sorted(groups):
        configs = rng.sample(sorted(groups[family]), 3)
        for config in sorted(configs):
            candidates = sorted(groups[family][config], key=lambda r: r["drive_id"])
            selected.extend(rng.sample(candidates, 2))
    return selected


def concentration(energy, valid):
    """Energy is temporal centered sum of squares per original spatial channel."""
    energy = np.asarray(energy, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    active = energy[valid]
    total = active.sum()
    if total <= 0 or not valid.any():
        raise ValueError("No variable finite channels")
    k = max(1, math.ceil(.1*valid.sum()))
    weights = energy / total
    # Shortest contiguous span containing 80% of energy; holes retain their positions.
    left = 0
    accum = 0.
    best = (0, len(energy)-1)
    for right, value in enumerate(weights):
        accum += value
        while left < right and accum-weights[left] >= .8:
            accum -= weights[left]
            left += 1
        if accum >= .8 and right-left < best[1]-best[0]:
            best = (left, right)
    tail = max(1, math.ceil(.1*len(energy)))
    return {
        "top10pct_channel_energy_share": float(np.sort(active)[-k:].sum()/total),
        "effective_channel_fraction": float(1/(np.sum(weights**2)*valid.sum())),
        "first10pct_position_energy_share": float(weights[:tail].sum()),
        "last10pct_position_energy_share": float(weights[-tail:].sum()),
        "span80_start_index": best[0], "span80_end_index": best[1],
        "span80_fraction_original_channels": (best[1]-best[0]+1)/len(energy),
    }


def analyze_file(path, row, figures):
    with h5py.File(path, "r") as handle:
        dataset_name, dataset, variable = choose_dataset(handle, "auto")
        if dataset.ndim != 3:
            raise ValueError("Expected native FEBUS [block,time,channel] layout")
        blocks, times, channels = dataset.shape
        spacing = np.asarray(dataset.parent.attrs["Spacing"]).reshape(-1)
        origin = np.asarray(dataset.parent.attrs["Origin"]).reshape(-1)
        extent = np.asarray(dataset.parent.attrs["Extent"]).reshape(-1)
        dx, dt = float(spacing[0]), float(spacing[1])*.001
        distance = float(origin[0]+extent[0]*dx)+np.arange(channels)*dx
        total_energy = np.zeros(channels)
        all_counts = np.zeros(channels, dtype=np.int64)
        block_rows, maps = [], []
        extreme_count = 0
        ordinary_energy = np.zeros(channels)
        full_finite = np.ones(channels, dtype=bool)
        max_abs = 0.
        preview = None
        for block in range(blocks):
            # Same central one-second interval as earlier preparation, all spatial rows.
            start, end = (500, 20500) if times >= 20500 else (0, times)
            values = np.asarray(dataset[block, start:end, :], dtype=np.float64)
            finite = np.isfinite(values)
            counts = finite.sum(axis=0)
            full_finite &= finite.all(axis=0)
            safe = np.where(finite, values, 0.)
            mean = safe.sum(axis=0)/np.maximum(counts, 1)
            centered = np.where(finite, values-mean, 0.)
            energy = np.sum(centered**2, axis=0)
            total_energy += energy
            all_counts += counts
            extreme_count += int(np.sum(finite & (np.abs(values)>1e20)))
            if finite.any():
                max_abs = max(max_abs, float(np.max(np.abs(values[finite]))))
            # Sensitivity diagnostic only: do huge finite values explain concentration?
            ordinary = finite & (np.abs(values)<=1e20)
            count_o = ordinary.sum(axis=0)
            mean_o = np.where(ordinary, values, 0.).sum(axis=0)/np.maximum(count_o, 1)
            ordinary_energy += np.sum(np.where(ordinary, values-mean_o, 0.)**2, axis=0)
            metrics = concentration(energy, counts>0)
            block_rows.append({"drive_id": row["drive_id"], "family": row["family"],
                               "configuration": row["configuration"], "block": block, **metrics})
            bins = []
            for index in range(0, len(values), 500):
                piece = centered[index:index+500]
                count = finite[index:index+500].sum(axis=0)
                rms = np.sqrt(np.sum(piece**2, axis=0)/np.maximum(count, 1))
                rms[count==0] = np.nan
                bins.append(rms)
            maps.append(np.asarray(bins).T)
            if preview is None:
                preview = values[:1000].T.copy()
        valid = all_counts>0
        metrics = concentration(total_energy, valid)
        ordinary_metrics = concentration(ordinary_energy, valid) if ordinary_energy.sum()>0 else {}
        channel_std = np.sqrt(total_energy/np.maximum(all_counts, 1))
        channel_std[~valid] = np.nan
        profile = [{"source_channel": i, "distance_m": float(distance[i]),
                    "finite_samples": int(all_counts[i]), "finite_in_all_selected_samples": bool(full_finite[i]),
                    "temporal_std": float(channel_std[i]), "energy_share": float(total_energy[i]/total_energy.sum())}
                   for i in range(channels)]
        write_csv(figures / f"{row['drive_id']}.channels.csv", profile)
        figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
        finite_preview = preview[np.isfinite(preview)]
        limit = max(float(np.quantile(np.abs(finite_preview), .99)), 1e-12)
        image = axes[0].imshow(preview, origin="lower", aspect="auto", cmap="RdBu_r",
                               vmin=-limit, vmax=limit,
                               extent=[0, preview.shape[1]*dt, distance[0], distance[-1]])
        axes[0].set(title="Primeiros 50 ms · escala linear p99", xlabel="Tempo relativo (s)", ylabel="Distância (m)")
        figure.colorbar(image, ax=axes[0], label=variable)
        rms_map = np.concatenate(maps, axis=1)
        image = axes[1].imshow(np.log10(np.maximum(rms_map, 1e-30)), origin="lower", aspect="auto", cmap="viridis",
                               extent=[0, blocks*(end-start)*dt, distance[0], distance[-1]])
        axes[1].set(title="Persistência · RMS temporal em log10", xlabel="Tempo concatenado (s)", ylabel="Distância (m)")
        figure.colorbar(image, ax=axes[1], label="log10 RMS centrado por bloco")
        axes[2].semilogy(distance, np.maximum(channel_std, 1e-30))
        a,b = metrics["span80_start_index"], metrics["span80_end_index"]
        axes[2].axvspan(distance[a], distance[b], color="orange", alpha=.25, label="Menor faixa com 80% da energia")
        axes[2].set(title=f"Top 10% canais: {metrics['top10pct_channel_energy_share']:.1%} da energia", xlabel="Distância (m)", ylabel="Desvio padrão temporal")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=.25)
        figure.suptitle(f"{row['family']} / {row['configuration']} / {row['title']}\nSem filtragem; canais originais preservados; cores entre arquivos não são diretamente comparáveis")
        figure.savefig(figures/f"{row['drive_id']}.png", dpi=140)
        plt.close(figure)
        return {"drive_id": row["drive_id"], "family": row["family"], "configuration": row["configuration"],
                "title": row["title"], "dataset": dataset_name, "variable": variable,
                "blocks": blocks, "channels": channels, "channels_with_finite_samples": int(valid.sum()),
                "channels_finite_in_all_selected_samples": int(full_finite.sum()), "dt_s": dt, "dx_m": dx,
                "distance_start_m": float(distance[0]), "distance_end_m": float(distance[-1]),
                "max_abs": max_abs, "finite_cells_above_1e20": extreme_count,
                **metrics, "span80_start_m": float(distance[metrics['span80_start_index']]),
                "span80_end_m": float(distance[metrics['span80_end_index']]),
                "ordinary_top10pct_energy_share": ordinary_metrics.get("top10pct_channel_energy_share", float("nan")),
                "block_top10_share_min": min(r["top10pct_channel_energy_share"] for r in block_rows),
                "block_top10_share_max": max(r["top10pct_channel_energy_share"] for r in block_rows)}, block_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with (ROOT/"data/manifests/febus_folds.csv").open() as stream:
        selected = select_files(list(csv.DictReader(stream)), args.seed)
    output, raw = args.output_dir.resolve(), args.raw_dir.resolve()
    protocol = {"seed": args.seed, "selection": selected,
                "sampling": "Three uniformly sampled configurations per family, two uniformly sampled files per configuration; exclude two earlier raw examples. No replacement after viewing data.",
                "analysis": "All original spatial channels; [500:20500] of each block when available; temporal energy centered separately per channel and block. Nonfinite cells excluded from sums, spatial positions retained.",
                "definitions": "Top10 uses ceil(0.1 * finite channel count), sorted by energy. Span80 is shortest contiguous original-index span containing 80% of energy. Strong concentration flag: top10 share >= 0.5, descriptive only, not physical invalidity.",
                "limitations": "Existing 207-file Drive corpus, not new acquisitions or independent generalization evidence. Geometry/descriptors inspected without neural inference. Ordinary <=1e20 sensitivity is separate from untouched raw results.",
                "neural_inference": False, "filtering": False}
    if args.resume:
        if json.loads((output/"protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
        write_csv(output/"selection.csv", selected)
    raw.mkdir(parents=True, exist_ok=True)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    service = build("drive", "v3", credentials=load_credentials(Path.home()/".config/gdrive-vqvae/token.json", None))
    all_rows, block_rows, downloads = [], [], []
    for i, row in enumerate(selected):
        file_id = row["drive_id"]
        meta = service.files().get(fileId=file_id, fields="id,name,size,md5Checksum,modifiedTime", supportsAllDrives=True).execute()
        path = raw/f"{file_id}.h5"
        if not path.exists():
            partial = path.with_suffix(".h5.part")
            with partial.open("wb") as stream:
                downloader = MediaIoBaseDownload(stream, service.files().get_media(fileId=file_id, supportsAllDrives=True), chunksize=8*1024*1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=3)
            if partial.stat().st_size != int(meta["size"]) or digest(partial, "md5") != meta["md5Checksum"]:
                raise ValueError("Drive download integrity check failed")
            partial.rename(path)
        if path.stat().st_size != int(meta["size"]) or digest(path,"md5") != meta["md5Checksum"]:
            raise ValueError("Local file differs from Drive")
        fingerprint = digest(path)
        downloads.append({**meta, "local_path": str(path), "sha256": fingerprint})
        (output/"downloads.json").write_text(json.dumps(downloads, indent=2)+"\n")
        metrics, blocks = analyze_file(path, row, figures)
        assert digest(path)==fingerprint, "Raw file changed during analysis"
        all_rows.append(metrics)
        block_rows.extend(blocks)
        write_csv(output/"files.csv", all_rows)
        write_csv(output/"blocks.csv", block_rows)
        print(f"[{i+1}/{len(selected)}] {row['family']}/{row['configuration']} {row['title']}: top10={metrics['top10pct_channel_energy_share']:.1%}, span80={metrics['span80_fraction_original_channels']:.1%}", flush=True)
    family_rows = []
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, family in zip(axes.ravel(), sorted({r["family"] for r in all_rows})):
        rows = [r for r in all_rows if r["family"]==family]
        shares = np.array([r["top10pct_channel_energy_share"] for r in rows])
        family_rows.append({"family": family, "files": len(rows), "configurations": len({r['configuration'] for r in rows}),
                            "strong_concentration_files": int(np.sum(shares>=.5)), "top10_share_min": float(shares.min()),
                            "top10_share_median": float(np.median(shares)), "top10_share_max": float(shares.max()),
                            "last10_share_median": float(np.median([r['last10pct_position_energy_share'] for r in rows])),
                            "span80_fraction_median": float(np.median([r['span80_fraction_original_channels'] for r in rows]))})
        ax.barh(np.arange(len(rows)), shares*100, color="#007f86")
        ax.set_yticks(np.arange(len(rows)), [r['configuration']+' / '+r['title'].replace('Copy of ','') for r in rows], fontsize=8)
        ax.axvline(50, color="orange", linestyle="--", label="Indicador descritivo de concentração")
        ax.set(xlim=(0,100), title=family, xlabel="Energia nos 10% canais mais energéticos (%)")
        ax.grid(axis="x", alpha=.25)
    figure.suptitle("Amostra aleatória estratificada do Drive · 24 arquivos / 12 configurações\nEnergia temporal centrada; sem inferência neural ou filtragem")
    figure.savefig(output/"family_summary.png", dpi=150)
    plt.close(figure)
    write_csv(output/"families.csv", family_rows)
    index = ["# Figuras por arquivo", "", "| Família | Configuração | Arquivo | Diagnóstico |", "|---|---|---|---|"]
    for row in all_rows:
        index.append(f"| {row['family']} | {row['configuration']} | {row['title']} | [Abrir figura](figures/{row['drive_id']}.png) |")
    (output/"FIGURES.md").write_text("\n".join(index)+"\n")
    report = ["# Faixas espaciais de alta energia: amostra aleatória do Drive", "",
              "24 downloads verificados pelo MD5 do Drive e SHA-256 local; três configurações por família, dois arquivos por configuração. Seed 20261022. Nenhuma inferência neural, filtragem ou alteração dos HDF5.", "",
              "Energia = soma dos desvios temporais quadráticos, centrados por canal e por bloco. Todos os canais espaciais são mantidos nas coordenadas originais; células não finitas não contribuem para as somas. Análise em [500:20500] por bloco, como nos recortes anteriores.", "",
              "| Família | Arquivos com top 10% ≥ 50% da energia | Top 10%: mínimo / mediana / máximo | Mediana da fração espacial com 80% da energia |",
              "|---|---:|---:|---:|"]
    for r in family_rows:
        report.append(f"| {r['family']} | {r['strong_concentration_files']}/{r['files']} | {r['top10_share_min']:.1%} / {r['top10_share_median']:.1%} / {r['top10_share_max']:.1%} | {r['span80_fraction_median']:.1%} |")
    report += ["", "## Limites de interpretação", "",
               "- Concentração em poucos canais não implica faixa contígua, mesma posição na fibra ou sinal inválido. Os mapas e span80 diferenciam essas situações.",
               "- O corte top10 ≥ 50% é um indicador descritivo previamente fixado, não um limiar de validade física ou tolerância do codec.",
               "- Os descritores por bloco em blocks.csv mostram persistência, sem transformar blocos correlacionados em réplicas independentes.",
               "- families.csv contém medianas por arquivo desta amostra, não estimativa balanceada por tempo/canal de todo o corpus.",
               "- files.csv separa contagem de valores >1e20 e concentração após excluí-los apenas do diagnóstico de sensibilidade. Os mapas e métricas principais usam os dados sem filtragem.",
               "- O sorteio cobre o corpus já inventariado; não constitui aquisição nova. Nenhum arquivo foi substituído em função do resultado.",
               "- Esta auditoria identifica distribuição espacial de energia, mas não prova sua origem física nem o ganho causal de remover faixas durante treinamento.",
               "", "## Evidência", "", "- family_summary.png: comparação de concentração por arquivo.",
               "- figures/: mapa linear do primeiro trecho, mapa logarítmico de persistência e perfil espacial por arquivo; CSV por canal com coordenadas.",
               "- FIGURES.md: índice das figuras com nomes de famílias/configurações/arquivos.",
               "- selection.csv / protocol.json / downloads.json: amostragem e integridade reproduzíveis."]
    (output/"REPORT.md").write_text("\n".join(report)+"\n")
    (output/"COMPLETE").write_text("24 Drive files downloaded and audited; no neural inference or filtering.\n")
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
