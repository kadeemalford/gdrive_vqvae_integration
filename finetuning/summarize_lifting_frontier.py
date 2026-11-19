"""Measured discrete RD comparisons; never certify a frontier by interpolation."""
import argparse
import json
from pathlib import Path

from finetuning.audit_prospective_preparation import write_json


def compare(reference, candidates, include_weights=False):
    def size(row):
        return row["stream_bytes"] + (row.get("shared_bytes", 0) if include_weights else 0)
    eligible = [row for row in candidates if size(row) <= size(reference)]
    if not eligible:
        return {"reference": reference["label"], "candidate": None,
                "includes_shared_weights": include_weights}
    best = min(eligible, key=lambda row: row["macro_families"])
    improvement = 1 - best["macro_families"] / reference["macro_families"]
    regressions = [group for group, value in best["groups"].items()
                   if value > reference["groups"][group]]
    equivalent = [row for row in eligible if row["macro_families"] <= reference["macro_families"]
                  and all(row["groups"][group] <= value for group, value in reference["groups"].items())]
    cheapest = min(equivalent, key=size) if equivalent else None
    saving = 1 - size(cheapest) / size(reference) if cheapest is not None else None
    return {"reference": reference["label"], "candidate": best["label"],
            "includes_shared_weights": include_weights,
            "reference_bytes": size(reference), "candidate_bytes": size(best),
            "reference_error": reference["macro_families"], "candidate_error": best["macro_families"],
            "relative_error_reduction": improvement, "regressing_groups": regressions,
            "exploratory_five_percent_error_gate": improvement >= .05 and not regressions,
            "equivalent_error_candidate": cheapest["label"] if cheapest else None,
            "equivalent_error_relative_byte_saving": saving,
            "exploratory_ten_percent_byte_gate": saving is not None and saving >= .1,
            "final_acceptance": False,
            "limits": "Development checks, not confirmation; all physical guards still required"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--vq", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for result in json.loads(args.curve.read_text()):
        study = Path(result["study"])
        arm = study.name
        rows.append({**result, "label": f"{study.parent.name}/{arm}/x{result['multiplier']:g}",
                     "method": arm, "shared_bytes": result["receiver_bytes"]})
    for path in args.vq:
        result = json.loads(path.read_text())
        rows.append({**{k: v for k, v in result.items() if k != "records"},
                     "label": path.parent.name, "method": "vq",
                     "shared_bytes": result["shared_full_model_bytes"]})
    learned = [r for r in rows if r["method"] == "learned"]
    references = [r for r in rows if r["method"] in ("fixed", "vq")]
    comparisons = [compare(reference, learned, include) for include in (False, True)
                   for reference in references]
    write_json(args.output, {"points": rows, "discrete_comparisons": comparisons,
                            "weights_note": "VQ export includes encoder; not a minimal decoder amortization certificate",
                            "interpretation": "No interpolation, new training, independent validation or promotion"})
    print(json.dumps({"points": len(rows), "comparisons": len(comparisons),
                      "final_acceptance": False}), flush=True)


if __name__ == "__main__":
    main()
