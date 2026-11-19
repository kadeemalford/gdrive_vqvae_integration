"""Describe configuration sensitivity using only the frozen transfer byte ledger."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(data):
    if data["kind"] != "feasibility-npz-transfer-results-v1" or data["fit_calls"] != 0:
        raise ValueError("Expected frozen transfer results")
    result = []
    for ratio in (.03, .1):
        groups = {}
        for row in data["curves"]:
            if row["ratio"] == ratio and row["horizon"] == 20:
                arms = groups.setdefault(row["group_id"], {})
                if row["mode"] in arms:
                    raise ValueError("Duplicate rate point")
                arms[row["mode"]] = row
        if len(groups) != 8:
            raise ValueError("Expected eight measured configurations")
        for mode in ("temporal", "spatiotemporal"):
            pairs = {}
            for group, arms in groups.items():
                b, c = arms["marginal"], arms[mode]
                if any(b[k] != c[k] for k in ("cells", "files", "windows", "available_windows_by_file")):
                    raise ValueError("Unpaired coverage")
                pairs[group] = b["total_bytes"], c["total_bytes"]
            base, candidate = map(sum, zip(*pairs.values()))
            gains = {g: 1 - c / b for g, (b, c) in sorted(pairs.items())}
            result.append({
                "ratio": ratio, "mode": mode, "aggregate_saving": 1 - candidate / base,
                "macro_configuration_saving": sum(gains.values()) / len(gains),
                "configurations_with_saving": sum(gain > 0 for gain in gains.values()),
                "configuration_saving": gains,
                "saving_without_each_configuration": {
                    g: 1 - (candidate - c) / (base - b) for g, (b, c) in sorted(pairs.items())
                },
            })
    return {"kind": "frozen-transfer-configuration-sensitivity-v1", "contrasts": result,
            "scope": "descriptive sensitivity, same frozen bytes; not resampling inference or confirmation",
            "new_waveform_reads": 0, "parameter_updates": 0, "historical_gates_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    result = summarize(json.loads(source))
    result["source"] = str(args.input)
    result["source_sha256"] = hashlib.sha256(source).hexdigest()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text() != text:
            raise ValueError("Refusing to overwrite different evidence")
        return
    with args.output.open("x") as stream:
        stream.write(text)


if __name__ == "__main__":
    main()
