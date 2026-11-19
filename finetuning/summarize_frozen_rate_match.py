"""Measured-point dominance only: no interpolated codec or checkpoint reselection."""
import argparse
import json
from pathlib import Path

from finetuning.audit_prospective_preparation import write_json
from finetuning.compare_physical_probe import compare_records


def dominators(reference, candidates):
    result = []
    for candidate in candidates:
        if candidate["cells"] != reference["cells"] or candidate["groups"].keys() != reference["groups"].keys():
            raise ValueError("Rate comparison requires identical acquired population")
        if (candidate["stream_bytes"] <= reference["stream_bytes"]
                and candidate["macro_families"] < reference["macro_families"]):
            result.append(candidate)
    return sorted(result, key=lambda row: (row["macro_families"], row["stream_bytes"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not (args.root / "COMPLETE").exists():
        raise ValueError("Both frozen curves must finish")
    rows = {}
    for variant in ("flat", "product"):
        if not (args.root / variant / "COMPLETE").exists():
            raise ValueError("Partial frozen curve")
        rows[variant] = [json.loads(path.read_text()) for path in
                         sorted((args.root / variant).glob("multiplier_*/results.json"))]
        if len(rows[variant]) != 6:
            raise ValueError("Expected all six registered rate points")
    comparisons = []
    for variant in rows:
        other = "product" if variant == "flat" else "flat"
        for reference in rows[variant]:
            choices = dominators(reference, rows[other])
            entry = {"reference_variant": variant, "reference_multiplier": reference["multiplier"],
                     "reference_error": reference["macro_families"], "reference_bytes": reference["stream_bytes"],
                     "other_variant": other,
                     "measured_dominators": [row["multiplier"] for row in choices]}
            if choices:
                candidate = choices[0]
                physical = compare_records(reference, candidate)
                entry.update(best_measured_multiplier=candidate["multiplier"],
                    relative_error_reduction=1 - candidate["macro_families"] / reference["macro_families"],
                    relative_byte_saving=1 - candidate["stream_bytes"] / reference["stream_bytes"],
                    regressing_groups=[group for group in reference["groups"]
                                       if candidate["groups"][group] > reference["groups"][group]],
                    physical=physical["summary"])
                name = f"{variant}_{reference['multiplier']:g}_vs_{other}_{candidate['multiplier']:g}"
                write_json(args.root / f"PHYSICAL_{name}.json", physical)
            comparisons.append(entry)
    result = {"measured_points": {variant: [{key: row[key] for key in (
        "multiplier", "macro_families", "stream_bpc", "stream_bytes")} for row in values]
        for variant, values in rows.items()}, "comparisons": comparisons,
        "parameter_updates": 0, "interpolated_points": False, "final_acceptance": False,
        "limits": "Finite measured grid; scalar dominance alone does not certify physical guards"}
    write_json(args.root / "MEASURED_FRONTIER.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
