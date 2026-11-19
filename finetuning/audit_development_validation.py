"""Audit frozen development results without reopening acquired waveforms."""
import argparse
import json
from pathlib import Path

from finetuning.audit_prospective_preparation import write_json
from finetuning.compare_physical_probe import compare_records
from finetuning.development_validation import candidates


def audit(root):
    exits = json.loads((root / "EXIT.json").read_text())
    if len(exits) != 5 or any(row["returncode"] for row in exits) or not (root / "COMPLETE").exists():
        raise ValueError("Incomplete frozen evaluation")
    results = {}
    paths = None
    physical_population = None
    frozen_scales = None
    for name in candidates():
        folder = root / name
        result = json.loads((folder / "results.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        manifests = [json.loads(line) for line in (folder / "frames_manifest.jsonl").read_text().splitlines()]
        if not (folder / "COMPLETE").exists():
            raise ValueError("Missing candidate completion marker")
        current_paths = [row["sample_path"] for row in manifests]
        if paths is None:
            paths = current_paths
        if (current_paths != paths or len(set(current_paths)) != 7220
                or current_paths != protocol["sample_paths"] or result["windows"] != 7220
                or result["cells"] != 635820000 or len(result["records"]) != 51
                or len(result["groups"]) != 6 or len(result["families"]) != 4):
            raise ValueError("Incomplete or mismatched validation population")
        if (result["parameter_updates"] or result["final_test_access"] or result["checkpoint_reselection"]
                or result["frozen_model_sha256"] != protocol["model_sha256"]):
            raise ValueError("Frozen evaluation contract violated")
        coverage = result["coverage_classes"]
        population = {
            (row["group_id"], row["drive_id"]): (
                tuple(row["source_channel_indices"]), tuple(channel["cells"] for channel in row["channels"])
            ) for row in result["records"]
        }
        if physical_population is None:
            physical_population, frozen_scales = population, protocol["scales"]
        if population != physical_population or protocol["scales"] != frozen_scales:
            raise ValueError("Physical channels, pooling or frozen scales differ across candidates")
        if sum(coverage[key] for key in ("literal_cells", "neural_ordinary_cells", "exact_escape_cells")) != result["cells"]:
            raise ValueError("Physical coverage classes do not sum to acquired cells")
        if (sum(row["bytes"] for row in manifests) != result["frame_bytes"]
                or sum(row["cells"] for row in result["records"]) != result["cells"]
                or sum(row["windows"] for row in result["records"]) != result["windows"]
                or result["stream_bytes"] != result["frame_bytes"] + result["setup_bytes"]):
            raise ValueError("Coverage or byte accounting mismatch")
        results[name] = result
    comparisons = {}
    for reference, candidate in [
        ("independent", "h2_joint"), ("encoder_local", "encoder_full"),
        ("mature_reference", "h2_joint"), ("mature_reference", "encoder_full"),
    ]:
        left, right = results[reference], results[candidate]
        physical = compare_records(left, right)
        label = f"{candidate}_vs_{reference}"
        write_json(root / f"PHYSICAL_{label}.json", physical)
        comparisons[label] = {
            "relative_error_reduction": 1 - right["macro_families"] / left["macro_families"],
            "relative_stream_saving": 1 - right["stream_bytes"] / left["stream_bytes"],
            "regressing_groups": [group for group in left["groups"] if right["groups"][group] > left["groups"][group]],
            **physical["summary"],
        }
    return {
        "passed": True, "waveforms_reopened": False, "parameter_updates": 0,
        "population_per_candidate": {"windows": 7220, "files": 51, "cells": 635820000},
        "comparisons": comparisons,
        "scales": {name: json.loads((root / name / "protocol.json").read_text())["scales"] for name in results},
        "scope": "Historically consumed development validation; no independent confirmation or promotion",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    write_json(args.root / "POSTFLIGHT_AUDIT.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
