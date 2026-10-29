#!/usr/bin/env python3
"""Plot completed, full-validation A/B/C evidence without model inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir",type=Path,required=True)
    args = parser.parse_args()
    root = args.study_dir
    if not (root/"COMPLETE").is_file():
        raise ValueError("Completed study required")
    results = json.loads((root/"results.json").read_text())
    colors = {"A":"#2864b4","B":"#d57722","C":"#007f86"}
    labels = {"A":"A · transferência","B":"B · nativa do zero","C":"C · anisotrópica do zero"}
    figure,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for arm in ("A","B","C"):
        rows = [results[f"{rate}_{arm}_phase24"] for rate in ("compact","high")]
        x = [32/r["pooled_index_plus_side_ratio"] for r in rows]
        for ax,key in zip(axes,("macro_relative_rmse","macro_correlation")):
            ax.plot(x,[r[key] for r in rows],"o-",color=colors[arm],label=labels[arm])
    for ax,title,ylabel in zip(axes,("Erro de reconstrução · menor é melhor","Correlação · maior é melhor"),("RMSE relativo macro","Correlação macro")):
        ax.set(title=title,xlabel="Bits nominais por célula (índices + side)",ylabel=ylabel)
        ax.grid(alpha=.25);ax.legend(fontsize=9)
    figure.suptitle("A/B/C · selecionados após 24 épocas · validação completa\nMesma coorte espacial; duas taxas medidas, linhas apenas conectam os pontos")
    figure.savefig(root/"rate_fidelity_phase24.png",dpi=160)
    plt.close(figure)
    def groups(path):
        with path.open() as stream:
            return {(r["family"],r["configuration"]):r for r in csv.DictReader(stream)}
    reference = groups(root/"baselines/historical_4x.groups.csv")
    keys = sorted(reference)
    figure,ax = plt.subplots(figsize=(12,5),constrained_layout=True)
    for i,arm in enumerate(("A","B","C")):
        candidate = groups(root/f"full_phase24_high_{arm}"/f"high_{arm}.groups.csv")
        delta = [float(candidate[k]["relative_rmse"])-float(reference[k]["relative_rmse"]) for k in keys]
        ax.bar(np.arange(len(keys))+(i-1)*.24,delta,width=.24,color=colors[arm],label=labels[arm])
    ax.axhline(0,color="black",linewidth=1)
    ax.set_xticks(np.arange(len(keys)),[f"{a}\n{b}" for a,b in keys],fontsize=8)
    ax.set(title="Taxa alta · mudança de RMSE por configuração contra o 4× histórico\nNegativo melhora; positivo regride. Mesma validação e pré-processamento.",ylabel="RMSE candidato − RMSE histórico")
    ax.legend();ax.grid(axis="y",alpha=.25)
    figure.savefig(root/"high_rate_configuration_deltas.png",dpi=160)
    plt.close(figure)


if __name__=="__main__":
    main()
