"""Paired descriptive physical guards from existing expanded records."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from finetuning.audit_prospective_preparation import write_json
from finetuning.prospective_spectral import merge_spectral, spectral_result


def compare_records(reference, candidate):
    left = {(row["group_id"], row["drive_id"]): row for row in reference["records"]}
    right = {(row["group_id"], row["drive_id"]): row for row in candidate["records"]}
    if left.keys() != right.keys() or reference["cells"] != candidate["cells"]:
        raise ValueError("Physical comparison requires paired files/cells")
    spectra = {"reference": defaultdict(dict), "candidate": defaultdict(dict)}
    files = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["cells"] != b["cells"] or len(a["channels"]) != len(b["channels"]):
            raise ValueError("Physical comparison coverage mismatch")
        files.append({"group_id": key[0], "drive_id": key[1],
                      "nrmse_star_increased": b["nrmse_star_macro_channels"] > a["nrmse_star_macro_channels"],
                      "p90_channel_nrmse_star_increased": b["p90_channel_nrmse_star"] > a["p90_channel_nrmse_star"],
                      "correlation_defined_reference": a["correlation_defined_channels"],
                      "correlation_defined_candidate": b["correlation_defined_channels"]})
        for label, row in (("reference", a), ("candidate", b)):
            # Only sufficient statistics may be added, never ratios themselves.
            merge_spectral(spectra[label][key[0]], {
                band: {name: values[name] for name in (
                    "target_energy", "error_energy", "prediction_energy", "phase_weight",
                    "phase_error_weighted_sum", "phase_defined_bins", "total_bins")}
                for band, values in row["spectral"].items()})
    group_spectra = {label: {group: spectral_result(values) for group, values in groups.items()}
                     for label, groups in spectra.items()}
    regressions = []
    for group, bands in group_spectra["reference"].items():
        for band, a in bands.items():
            b = group_spectra["candidate"][group][band]
            regressions.append({
                "group_id": group, "band": band,
                "band_relative_rmse_increased": b["relative_rmse"] > a["relative_rmse"]
                    if a["relative_rmse"] is not None and b["relative_rmse"] is not None else None,
                "rms_distance_to_one_increased": abs(b["rms_ratio"] - 1) > abs(a["rms_ratio"] - 1)
                    if a["rms_ratio"] is not None and b["rms_ratio"] is not None else None,
                "phase_error_increased": b["phase_error_radians"] > a["phase_error_radians"]
                    if a["phase_error_radians"] is not None and b["phase_error_radians"] is not None else None,
                "phase_defined_reference": a["phase_defined_bins"], "phase_defined_candidate": b["phase_defined_bins"]})
    return {"files": files, "spectra": group_spectra, "band_regressions": regressions,
            "summary": {"paired_files": len(files),
                        "file_nrmse_regressions": sum(r["nrmse_star_increased"] for r in files),
                        "file_p90_regressions": sum(r["p90_channel_nrmse_star_increased"] for r in files),
                        "band_rmse_regressions": sum(r["band_relative_rmse_increased"] is True for r in regressions),
                        "band_rms_amplitude_regressions": sum(r["rms_distance_to_one_increased"] is True for r in regressions),
                        "phase_descriptive_regressions": sum(r["phase_error_increased"] is True for r in regressions)},
            "limits": "Descriptive pooled band statistics; phase supports may differ, coverage explicit; no final guard acceptance"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_records(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    write_json(args.output, result)
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
