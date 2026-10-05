#!/usr/bin/env python3
"""Configuration- and family-macro paired metrics, without treating windows as independent."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("correlation", "relative_rmse", "snr_db")


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["family"], row["configuration"])].append(row)
    configurations = {key: {m: mean(float(r[m]) for r in rs) for m in METRICS} for key, rs in groups.items()}
    families = defaultdict(list)
    for (mode, family, _), values in configurations.items():
        families[(mode, family)].append(values)
    return configurations, {key: {m: mean(v[m] for v in vs) for m in METRICS} for key, vs in families.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.experiment_dir / "protocol.json").read_text())
    all_rows = []
    fold_results = {}
    lines = ["# Balanceamento FEBUS: comparação pareada", "",
             "PILOTO curto, na validação: não comprova generalização nem decide o protocolo final." if protocol["pilot"] else
             "Comparação nos testes isolados. Cada configuração pertence a uma única partição por fold.", "",
             "Pesos iguais por configuração dentro da família; pesos iguais por família. Δ = balanceado − uniforme.", "",
             "| Fold | Correlação uniforme | Balanceado | Δ | RMSE relativo uniforme | Balanceado |",
             "|---|---:|---:|---:|---:|---:|"]
    for fold in range(4):
        with (args.experiment_dir / f"fold_{fold}/evaluation/per_sample_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        all_rows.extend(rows)
        _, families = aggregate(rows)
        values = {mode: {m: mean(v[m] for (label, _), v in families.items() if label == mode) for m in METRICS}
                  for mode in protocol["sampling"]}
        fold_results[fold] = values
        a, b = values["window"], values["family-group"]
        lines.append(f"| {fold} | {a['correlation']:.4f} | {b['correlation']:.4f} | {b['correlation']-a['correlation']:+.4f} | {a['relative_rmse']:.4f} | {b['relative_rmse']:.4f} |")
    _, families = aggregate(all_rows)
    lines += ["", "## Famílias (média por configuração)", "", "| Família | Corr. uniforme | Balanceado | RMSE uniforme | Balanceado | SNR uniforme | Balanceado |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for family in sorted({family for _, family in families}):
        a, b = families[("window", family)], families[("family-group", family)]
        lines.append(f"| {family} | {a['correlation']:.4f} | {b['correlation']:.4f} | {a['relative_rmse']:.4f} | {b['relative_rmse']:.4f} | {a['snr_db']:.3f} | {b['snr_db']:.3f} |")
    wins = sum(v["family-group"]["correlation"] > v["window"]["correlation"] for v in fold_results.values())
    regressions = [family for mode, family in families if mode == "window" and
                   (families[("family-group", family)]["correlation"] < families[(mode, family)]["correlation"] or
                    families[("family-group", family)]["relative_rmse"] > families[(mode, family)]["relative_rmse"])]
    lines += ["", f"Ganho de correlação macro em {wins}/4 folds. Famílias com regressão em correlação ou RMSE: {', '.join(regressions) or 'nenhuma'}.",
              "", "Não adotar automaticamente: revisar dispersão por configuração, perdas por família, espectro e reconstruções visuais. Uma única semente não demonstra robustez estatística.",
              "No piloto, as configurações de validação são avaliadas por modelos diferentes dos do teste completo; não comparar estes números diretamente ao baseline histórico."]
    (args.experiment_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    payload = {"protocol": protocol, "folds": fold_results, "families": {f"{m}:{f}": v for (m, f), v in families.items()},
               "correlation_macro_wins": wins, "family_regressions": regressions}
    (args.experiment_dir / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
