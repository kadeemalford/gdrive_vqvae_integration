#!/usr/bin/env python3
"""Validation-only accounting of initial, selected and final numeric-pilot states."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_history(folder):
    with (folder / "validation_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    history = {int(row["epoch"]): row for row in rows}
    if len(history) != len(rows) or 0 not in history:
        raise ValueError("History must have unique epochs and initialization")
    with (folder / "validation_groups.csv").open() as stream:
        groups = list(csv.DictReader(stream))
    summary = json.loads((folder / "summary.json").read_text())
    return history, groups, summary


def family_metrics(groups, epoch):
    families = defaultdict(list)
    seen = set()
    for row in groups:
        if int(row["epoch"]) != epoch:
            continue
        key = row["family"], row["configuration"]
        if key in seen:
            raise ValueError("Repeated configuration in selected epoch")
        seen.add(key)
        families[row["family"]].append(row)
    if not families:
        raise ValueError("Missing group metrics for selected epoch")
    keys = ("relative_rmse", "correlation")
    return {family: {key: mean(float(row[key]) for row in rows) for key in keys}
            for family, rows in sorted(families.items())}


def compare(root):
    protocol = json.loads((root / "protocol.json").read_text())
    results, hashes, initial_metrics = {}, set(), []
    for arm in protocol["arms"]:
        history, groups, summary = read_history(root / arm)
        hashes.add(summary["validation_hash"])
        selected_epoch = int(summary["best_macro_epoch"])
        final_epoch = int(summary["epochs_completed"])
        states = {}
        initial_epoch = int(protocol.get("initial_epoch", 0))
        for label, epoch in (("initial", initial_epoch), ("selected", selected_epoch), ("final", final_epoch)):
            row = history[epoch]
            states[label] = {"epoch": epoch, "macro_relative_rmse": float(row["macro_relative_rmse"]),
                             "macro_correlation": float(row["macro_correlation"]),
                             "worst_family_relative_rmse": float(row["worst_family_relative_rmse"]),
                             "families": family_metrics(groups, epoch)}
            for key in ("macro_ordinary_channel_macro_relative_rmse", "macro_ordinary_channel_macro_correlation",
                        "macro_side_bits_per_cell", "macro_index_plus_side_nominal_ratio",
                        "macro_reconstruction_rms_ratio", "macro_fft_phase_alignment", "macro_fft_magnitude_relative_error"):
                if key in row:
                    states[label][key] = float(row[key])
        initial_metrics.append(states["initial"])
        initial, selected = states["initial"], states["selected"]
        if initial["families"].keys() != selected["families"].keys():
            raise ValueError("Family coverage changed between states")
        regressions = [family for family in initial["families"]
                       if selected["families"][family]["relative_rmse"] > initial["families"][family]["relative_rmse"] + 1e-6]
        states["selected_family_rmse_regressions"] = regressions
        initial_groups = {(row["family"], row["configuration"]): row for row in groups if int(row["epoch"]) == initial_epoch}
        selected_groups = {(row["family"], row["configuration"]): row for row in groups if int(row["epoch"]) == selected_epoch}
        if initial_groups.keys() != selected_groups.keys():
            raise ValueError("Configuration coverage changed between states")
        states["selected_configuration_rmse_regressions"] = [
            {"family": family, "configuration": configuration,
             "delta_rmse": float(selected_groups[(family, configuration)]["relative_rmse"]) - float(row["relative_rmse"])}
            for (family, configuration), row in initial_groups.items()
            if float(selected_groups[(family, configuration)]["relative_rmse"]) > float(row["relative_rmse"]) + 1e-6]
        states["selected_macro_rmse_delta"] = selected["macro_relative_rmse"] - initial["macro_relative_rmse"]
        states["selected_worst_family_rmse_delta"] = selected["worst_family_relative_rmse"] - initial["worst_family_relative_rmse"]
        results[arm] = states
    if len(hashes) != 1:
        raise ValueError("Branches used different validation sets")
    # This pilot's validation has no escaped cells, so initial macro scores should match.
    if protocol.get("initial_validation_expected_equal", True) and any(abs(row["macro_relative_rmse"] - initial_metrics[0]["macro_relative_rmse"]) > 1e-6 for row in initial_metrics):
        raise ValueError("Initial validation mismatch: inspect preprocessing/checkpoint parity")
    return {"protocol": protocol, "arms": results,
            "warning": "Selection and measurement share exploratory validation; not independent generalization evidence. No automatic promotion."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir
    if not (root / "COMPLETE").is_file():
        raise ValueError("Wait for both training branches to complete")
    comparison = compare(root)
    titles = {"preservation_full_corpus_v1": "Adaptação completa versus decoder-only — validação completa",
              "architecture_initialization_screening_v1": "Triagem de arquitetura/inicialização — validação exploratória",
              "scratch_budget_continuation_v1": "Continuação de orçamento scratch — optimizer/RNG preservados",
              "paired_codebook_refresh_v1": "Reativação train-only de códigos — comparação pareada",
              "spatiotemporal_rate_allocation_v1": "Alocação espaço-temporal — mesma taxa ou quase igual",
              "spatiotemporal_allocation_fold_confirmation_v1": "Confirmação de alocação — fold/configurações e seed adicionais",
              "neural_rate_distortion_sweep_v1": "Curva neural taxa-fidelidade — taxas diferentes explicitamente cobradas",
              "high_rate_budget_continuation_v1": "Continuação do candidato de maior taxa — mesmo optimizer/RNG",
              "high_rate_fold_confirmation_robust_v1": "Confirmação de taxa no fold 0 — referência robusta comum",
              "temporal_density_full_corpus_continuation_v1": "Continuação longa 2x/4x — validação completa, fidelidade e taxa",
              "temporal_latent_density_pilot_v1": "Densidade temporal dos latentes — fidelidade e taxa na validação",
              "channel_representation_pilot_v1": "Normalização global versus por canal — validação no mesmo alvo restaurado"}
    title = titles.get(comparison["protocol"]["kind"], "Piloto numérico/conservador — validação")
    lines = [f"# {title}", "", comparison["warning"], "",
             "| Braço | Estado | Epoch | RMSE macro | Correlação macro | Pior família RMSE |",
             "|---|---|---:|---:|---:|---:|"]
    for arm, result in comparison["arms"].items():
        for label in ("initial", "selected", "final"):
            state = result[label]
            lines.append(f"| {arm} | {label} | {state['epoch']} | {state['macro_relative_rmse']:.6f} | {state['macro_correlation']:.6f} | {state['worst_family_relative_rmse']:.6f} |")
    lines += ["", "## Famílias: inicial → selecionado → final", "",
              "| Braço | Família | RMSE inicial | Selecionado | Final |", "|---|---|---:|---:|---:|"]
    for arm, result in comparison["arms"].items():
        for family in result["initial"]["families"]:
            values = [result[label]["families"][family]["relative_rmse"] for label in ("initial", "selected", "final")]
            lines.append(f"| {arm} | {family} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
        lines += ["", f"{arm}: famílias com regressão de RMSE no selecionado: {result['selected_family_rmse_regressions']}.", ""]
        lines += [f"Configurações com regressão: {result['selected_configuration_rmse_regressions']}.", ""]
    if all("macro_fft_phase_alignment" in result["selected"] for result in comparison["arms"].values()):
        lines += ["## Amplitude e fase: inicial → selecionado", "",
                  "| Braço | RMS inicial | RMS selecionado | Fase inicial | Fase selecionada |",
                  "|---|---:|---:|---:|---:|"]
        for arm, result in comparison["arms"].items():
            a, b = result["initial"], result["selected"]
            lines.append(f"| {arm} | {a['macro_reconstruction_rms_ratio']:.6f} | {b['macro_reconstruction_rms_ratio']:.6f} | {a['macro_fft_phase_alignment']:.6f} | {b['macro_fft_phase_alignment']:.6f} |")
    if comparison["protocol"].get("physical_validation"):
        lines += ["## Custos e erro por canal no selecionado", "",
                  "| Braço | Side bits/célula | Razão nominal índices+side | RMSE relativo médio por canal |",
                  "|---|---:|---:|---:|"]
        for arm, result in comparison["arms"].items():
            state = result["selected"]
            lines.append(f"| {arm} | {state['macro_side_bits_per_cell']:.6f} | {state['macro_index_plus_side_nominal_ratio']:.2f} | {state['macro_ordinary_channel_macro_relative_rmse']:.6f} |")
        lines += ["", "Taxa nominal parcial: índices fixos + parâmetros/escapes, sem container, mapa espacial ou codificação entrópica.",
                  "Erro relativo por canal é sensível a canais quase constantes; não substitui erro raw, distribuição por canal e critérios físicos."]
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
