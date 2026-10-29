#!/usr/bin/env python3
"""Audit the fixed spatial filter on downloaded files without neural inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.prepare_finetune_samples import blocks, choose_dataset
from finetuning.spatial_band_filter import POLICY, DEFAULTS, detect_spatial_bands, retained_regions, contiguous_regions


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--include-raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with args.manifest.open() as stream:
        selected = [(row, args.raw_dir/f"{row['drive_id']}.h5") for row in csv.DictReader(stream)]
    if args.include_raw_dir:
        selected += [({"drive_id": path.stem, "family": "single_mode", "configuration": path.stem,
                       "title": path.name}, path) for path in sorted(args.include_raw_dir.glob("*.h5"))]
    protocol = {"policy": POLICY, "parameters": DEFAULTS, "neural_inference": False,
                "training": False, "raw_mutation": False,
                "coverage_note": "Thresholds are conservative heuristics, not validated physical labels; do not tune by reconstruction error."}
    (args.output_dir/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    rows = []
    for row, path in selected:
        before = file_hash(path)
        with h5py.File(path, "r") as handle:
            name, dataset = choose_dataset(handle)
            source = blocks(dataset)
            decision = detect_spatial_bands(source)
            removed = np.asarray(decision["removed_source_channel_indices"], dtype=int)
            spacing = np.asarray(dataset.parent.attrs["Spacing"]).reshape(-1)
            origin = np.asarray(dataset.parent.attrs["Origin"]).reshape(-1)
            extent = np.asarray(dataset.parent.attrs["Extent"]).reshape(-1)
            positions = origin[0]+(extent[0]+np.arange(source.shape[1]))*spacing[0]
        (args.output_dir/f"{row['drive_id']}.decision.json").write_text(json.dumps(decision, indent=2, allow_nan=False)+"\n")
        energy = np.zeros(source.shape[1])
        counts = np.zeros(source.shape[1])
        total_finite = 0
        kept_finite = 0
        region_count = []
        for block in source:
            finite = np.isfinite(block)
            count = finite.sum(axis=1)
            safe = np.where(finite, block.astype(np.float64), 0.)
            mean = safe.sum(axis=1)/np.maximum(count,1)
            energy += np.sum(np.where(finite, safe-mean[:,None], 0.)**2, axis=1)
            counts += count
            regions = retained_regions(block, decision)
            region_count.append(len(regions))
            total_finite += int(finite.sum())
            kept_finite += sum(int(block[index].size) for index in regions)
            assert all(np.all(np.diff(index)==1) for index in regions)
        dropped = np.zeros(source.shape[1], dtype=bool)
        dropped[removed] = True
        ranges = [{"start_channel": int(indices[0]), "end_channel_inclusive": int(indices[-1]),
                   "start_m": float(positions[indices[0]]), "end_m": float(positions[indices[-1]])}
                  for indices in contiguous_regions(dropped)]
        std = np.sqrt(energy/np.maximum(counts,1))
        std[counts==0] = np.nan
        figure, ax = plt.subplots(figsize=(11,4), constrained_layout=True)
        ax.semilogy(positions, np.maximum(std,1e-30), label="Original · desvio padrão temporal")
        for i, region in enumerate(ranges):
            ax.axvspan(region["start_m"]-spacing[0]/2, region["end_m"]+spacing[0]/2,
                       color="red", alpha=.25, label="Faixa excluída" if i==0 else None)
        ax.set(title=f"{row['family']} / {row['configuration']} / {row['title']}\n{decision['status']} · {len(removed)} canais excluídos por amplitude persistente",
               xlabel="Distância (m)", ylabel="Desvio padrão · escala logarítmica")
        ax.grid(alpha=.25)
        ax.legend()
        figure.savefig(args.output_dir/f"{row['drive_id']}.png", dpi=140)
        plt.close(figure)
        rows.append({"drive_id": row["drive_id"], "family": row["family"], "configuration": row["configuration"],
                     "title": row["title"], "sha256": before, "dataset": name,
                     "source_channels": source.shape[1], "status": decision["status"],
                     "removed_channels": len(removed), "removed_ranges_json": json.dumps(ranges),
                     "retained_finite_cell_fraction": kept_finite/max(total_finite,1),
                     "removed_temporal_energy_fraction": float(energy[removed].sum()/max(energy.sum(),1e-300)),
                     "minimum_retained_regions": min(region_count), "maximum_retained_regions": max(region_count)})
        assert file_hash(path)==before
        print(row["family"], row["configuration"], row["title"], decision["status"], "removed", len(removed), flush=True)
    with (args.output_dir/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    report = ["# Validação operacional da seleção espacial", "",
              f"{len(rows)} arquivos do manifesto e diretório local opcional. Sem inferência neural ou treino. HDF5 inalterados e hashes conferidos.",
              "Valores dos canais retidos não são filtrados, interpolados ou normalizados nesta etapa; faixas separadas permanecem separadas.", "",
              "| Família | Configuração | Arquivo | Canais excluídos | Estado | Figura |", "|---|---|---|---:|---|---|"]
    for row in rows:
        report.append(f"| {row['family']} | {row['configuration']} | {row['title']} | {row['removed_channels']}/{row['source_channels']} | {row['status']} | [Ver]({row['drive_id']}.png) |")
    report += ["", "Remoção de energia não é métrica de sucesso: não há rótulos físicos para medir falsos positivos ou falsos negativos. A heurística não substitui calibração e pode preservar faixas moderadamente maiores ou rejeitar eventos espaciais persistentes de grande amplitude.",
               "Mapas por arquivo e decisões detalhadas estão nos JSON; falhas de evidência mantêm candidatos por amplitude e registram abstenção."]
    (args.output_dir/"REPORT.md").write_text("\n".join(report)+"\n")
    (args.output_dir/"COMPLETE").write_text("Spatial filter audit complete; no inference, training or raw modifications.\n")


if __name__ == "__main__":
    main()
