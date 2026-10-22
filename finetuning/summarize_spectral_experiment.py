#!/usr/bin/env python3
"""Compare objective and checkpoint selection separately, using validation histories only."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("relative_rmse", "correlation", "fft_magnitude_relative_error", "fft_phase_alignment",
           "reconstruction_rms_ratio", "spectral_loss", "temporal_loss", "vq_loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir
    protocol = json.loads((root / "protocol.json").read_text())
    fold_results, selected_groups, family_groups = {}, [], defaultdict(list)
    lines = ["# Temporal + espectral: comparação pareada na validação", "",
             "Piloto curto: não demonstra convergência/generalização." if protocol["pilot"] else "Validação completa, sem consultar testes para seleção.", "",
             "Ambos os braços partem do mesmo best uniforme de cada fold, com AdamW novo e LR 3e-6.",
             "Seleção legacy = temporal+VQ val_loss; macro = menor rRMSE macro por família/configuração.", "",
             "| Fold | Braço | Época legacy / macro | RMSE macro legacy / macro | Corr. macro legacy / macro |",
             "|---|---|---:|---:|---:|"]
    for fold in range(4):
        fold_results[fold] = {}
        hashes = set()
        for arm in protocol["arms"]:
            folder = root / f"fold_{fold}/{arm}"
            summary = json.loads((folder / "summary.json").read_text())
            hashes.add(summary["validation_hash"])
            with (folder / "validation_metrics.csv").open() as stream:
                history = {int(r["epoch"]): r for r in csv.DictReader(stream)}
            with (folder / "validation_groups.csv").open() as stream:
                grouped = list(csv.DictReader(stream))
            selectors = {}
            for selector, epoch in (("legacy", summary["best_epoch"]), ("macro", summary["best_macro_epoch"])):
                row = history[epoch]
                selectors[selector] = {"epoch": epoch, **{f"macro_{key}": float(row[f"macro_{key}"]) for key in METRICS},
                                       "worst_family_relative_rmse": float(row["worst_family_relative_rmse"])}
                for group in grouped:
                    if int(group["epoch"]) == epoch:
                        values = {metric: float(group[metric]) for metric in METRICS}
                        family_groups[(arm, selector, group["family"])].append(values)
                        selected_groups.append({"fold": fold, "arm": arm, "selector": selector,
                                                "family": group["family"], "configuration": group["configuration"],
                                                "epoch": epoch, **values})
            fold_results[fold][arm] = selectors
            a, b = selectors["legacy"], selectors["macro"]
            lines.append(f"| {fold} | {arm} | {a['epoch']} / {b['epoch']} | {a['macro_relative_rmse']:.5f} / {b['macro_relative_rmse']:.5f} | {a['macro_correlation']:.4f} / {b['macro_correlation']:.4f} |")
        if len(hashes) != 1:
            raise ValueError(f"Branches evaluated different validation windows in fold {fold}")
    families = {"|".join(key): {metric: mean(r[metric] for r in rows) for metric in METRICS}
                for key, rows in family_groups.items()}
    lines += ["", "## Famílias — checkpoint macro em ambos os braços", "",
              "| Família | rRMSE temporal / espectral | Corr. temporal / espectral | Erro FFT temporal / espectral | Fase temporal / espectral | RMS ratio temporal / espectral |",
              "|---|---:|---:|---:|---:|---:|"]
    for family in sorted({key[2] for key in family_groups}):
        a, b = families[f"temporal|macro|{family}"], families[f"spectral|macro|{family}"]
        lines.append(f"| {family} | {a['relative_rmse']:.5f} / {b['relative_rmse']:.5f} | {a['correlation']:.4f} / {b['correlation']:.4f} | {a['fft_magnitude_relative_error']:.4f} / {b['fft_magnitude_relative_error']:.4f} | {a['fft_phase_alignment']:.4f} / {b['fft_phase_alignment']:.4f} | {a['reconstruction_rms_ratio']:.4f} / {b['reconstruction_rms_ratio']:.4f} |")
    wins = sum(v["spectral"]["macro"]["macro_relative_rmse"] < v["temporal"]["macro"]["macro_relative_rmse"] for v in fold_results.values())
    lines += ["", f"Espectral melhora RMSE macro em {wins}/4 folds de validação.",
              "Não promover automaticamente. Avaliar regressões por família, fase, amplitudes e pior família; seleção e avaliação aqui usam a mesma validação.",
              "Os testes reservados não foram lidos por este runner. Perdas totais com pesos espectrais distintos não são diretamente comparáveis."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    (root / "comparison.json").write_text(json.dumps({"protocol": protocol, "folds": fold_results,
        "families": families, "macro_rmse_fold_wins": wins, "selected_groups": selected_groups}, indent=2) + "\n")


if __name__ == "__main__":
    main()
