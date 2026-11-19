"""Summarize frozen F1 records without reading waveform or updating models."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(folder):
    folder = Path(folder)
    if not (folder / "COMPLETE").exists():
        raise ValueError("F1 must finish before interpretation")
    result = json.loads((folder / "results.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    records = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
    scores = json.loads((folder / "context_scores.json").read_text())
    if (result["forward_passes"] != 414 or len(records) != 414
            or result["parameter_updates"] != 0
            or result["reserved_signal_reads"] != 0
            or not result["weights_unchanged"]
            or len(result["full_replay_identity"]) != 60
            or not all(row["full_native_fixed_bit_identity"] for row in result["full_replay_identity"])):
        raise ValueError("Frozen F1 completion invariants failed")
    sources_verified = []
    for original, expected in protocol["source_sha256"].items():
        parts = Path(original).parts
        if "gdrive_vqvae_integration" in parts:
            relative = Path(*parts[parts.index("gdrive_vqvae_integration") + 1:])
        elif "vqvae" in parts:
            relative = Path("vendor/vqvae").joinpath(*parts[parts.index("vqvae") + 1:])
        else:
            raise ValueError(f"Cannot locate snapshot for source: {original}")
        if sha256(folder / "code_snapshot" / relative) != expected:
            raise ValueError(f"F1 source snapshot differs: {original}")
        sources_verified.append(str(relative))
    grouped = defaultdict(list)
    for row in records:
        grouped[row["model"], row["normalization"], row["sample_path"]].append(row)
    ratios = defaultdict(list)
    no_contrast = []
    for (model, norm, sample), values in sorted(grouped.items()):
        values.sort(key=lambda value: value["context_channels"])
        narrow, full = values[0], values[-1]
        if len(values) == 1:
            no_contrast.append({"model": model, "normalization": norm, "sample_path": sample})
            continue
        left, right = narrow["metrics"]["rmse"], full["metrics"]["rmse"]
        ratios[model, norm].append({
            "sample_path": sample, "drive_id": narrow["drive_id"], "group_id": narrow["group_id"],
            "narrow_channels": narrow["context_channels"], "full_channels": full["context_channels"],
            "narrow_rmse": left, "full_rmse": right,
            "narrow_over_full_minus_one": left / right - 1 if right > 0 else None,
        })
    summaries = []
    for (model, norm), comparisons in sorted(ratios.items()):
        defined = [r["narrow_over_full_minus_one"] for r in comparisons
                   if r["narrow_over_full_minus_one"] is not None]
        summaries.append({
            "model": model, "normalization": norm, "files_with_contrast": len(comparisons),
            "files_with_defined_ratio": len(defined),
            "narrow_better": sum(value < 0 for value in defined),
            "narrow_worse": sum(value > 0 for value in defined),
            "narrow_equal": sum(value == 0 for value in defined),
            "median_narrow_over_full_minus_one": statistics.median(defined) if defined else None,
            "minimum": min(defined) if defined else None, "maximum": max(defined) if defined else None,
            "comparisons": comparisons,
            "passing_same_pair_gates": [gate for gate in scores["gates"]
                                       if gate["model"] == model and gate["normalization"] == norm
                                       and gate["descriptive_gate_met"]],
        })
    return {
        "kind": "feasibility-context-interpretation-v1",
        "inputs": {name: sha256(folder / name) for name in
                   ("protocol.json", "results.json", "records.jsonl", "context_scores.json")},
        "sources_verified": sources_verified,
        "historical_input_verification": protocol["historical_input_verification"],
        "summary": summaries, "no_context_contrast": no_contrast,
        "forward_passes": 414, "parameter_updates": 0, "reserved_signal_reads": 0,
        "new_waveform_reads": 0, "codec_claim": False, "model_promoted": False,
        "limits": [
            "Medians describe 19 fixed central cores, not the complete corpus.",
            "Narrowest widths and full widths vary by file; these ratios are descriptive.",
            "The predeclared sensitivity gate is computed for the same explicit width pair.",
            "No complete-codec rate, independent-session confirmation or training benefit measured.",
            "Normalization and internal context effects are not necessarily additive.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output.exists() and args.output.read_text() != text:
        raise ValueError("Preserve existing analysis; choose a new destination")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    for row in result["summary"]:
        print(row["model"], row["normalization"],
              row["median_narrow_over_full_minus_one"], row["narrow_better"], row["narrow_worse"])


if __name__ == "__main__":
    main()
