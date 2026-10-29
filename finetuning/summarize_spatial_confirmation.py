"""Summarize paired validation diagnostics without new inference or selection."""
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("spatial_abc_20261024_v2", "spatial_abc_fold1_20261026")
METRICS = ("relative_rmse", "correlation", "ordinary_channel_p90_relative_rmse",
           "ordinary_raw_rmse", "reconstruction_rms_ratio", "fft_phase_alignment",
           "fft_magnitude_relative_error", "escaped_cell_fraction")


def main():
    output = ROOT / "results/experiments/spatial_confirmation_diagnostics_20261028"
    output.mkdir(exist_ok=True)
    files = []
    for study in STUDIES:
        root = ROOT / "results/experiments" / study
        if not (root / "COMPLETE").is_file():
            raise ValueError(f"Incomplete study: {study}")
        def load(path):
            with path.open() as stream:
                return {row["path"]: row for row in csv.DictReader(stream)}
        candidate = load(root / "full_phase24_high_A/high_A.windows.csv")
        reference = load(root / "baselines/historical_4x.windows.csv")
        if candidate.keys() != reference.keys():
            raise ValueError("Unpaired validation coverage")
        grouped = defaultdict(list)
        for path, row in candidate.items():
            other = reference[path]
            for key in ("drive_id", "family", "configuration", "input_cells"):
                if row[key] != other[key]:
                    raise ValueError(f"Mismatched {key}: {path}")
            grouped[(row["family"], row["configuration"], row["drive_id"])].append((row, other))
        for (family, configuration, drive_id), pairs in sorted(grouped.items()):
            cells = sum(int(a["input_cells"]) for a, _ in pairs)
            result = dict(study=study, family=family, configuration=configuration,
                          drive_id=drive_id, windows=len(pairs), input_cells=cells)
            for metric in METRICS:
                for index, label in enumerate(("A", "historical")):
                    result[f"{label}_{metric}"] = sum(
                        float(pair[index][metric]) * int(pair[0]["input_cells"])
                        for pair in pairs) / cells
                result[f"delta_{metric}"] = result[f"A_{metric}"] - result[f"historical_{metric}"]
            files.append(result)
    with (output / "paired_file_diagnostics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(files[0]))
        writer.writeheader()
        writer.writerows(files)
    summary = {"kind": "retrospective_diagnostic_not_acceptance_or_selection",
               "aggregation": "input-cell-weighted window metrics within source file; not pooled channel p90",
               "files": len(files), "studies": {}}
    for study in STUDIES:
        rows = [r for r in files if r["study"] == study]
        summary["studies"][study] = {
            "files": len(rows),
            "files_with_higher_rmse": sum(r["delta_relative_rmse"] > 0 for r in rows),
            "files_with_lower_correlation": sum(r["delta_correlation"] < 0 for r in rows),
            "largest_rmse_regressions": sorted(rows, key=lambda r: r["delta_relative_rmse"], reverse=True)[:5]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: {x: y for x, y in v.items() if x != "largest_rmse_regressions"}
                      for k, v in summary["studies"].items()}, indent=2))


if __name__ == "__main__":
    main()
