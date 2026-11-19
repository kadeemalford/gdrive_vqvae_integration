"""Train-only nominal clock pairing and bounded original-context opportunity."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.robust_escape import encode_robust

CORPUS = Path("data/finetune/spatial_band_v1_fold0_20261024")
RAW = Path("data/finetune/raw")
OUTPUT = Path("results/experiments/clock_aligned_multicore_probe_20261115")
AUDIT = Path("results/experiments/train_clock_copy_audit_fold0_20261115/results.json")
HEADERS = Path("results/experiments/physical_metadata_preflight_fold0_20261115/results.json")


def zone_attributes(header):
    return header["attributes"]["/" + str(Path(header["dataset"]).parent).lstrip("/")]


def nominal_pairs(files, headers):
    pairs = []
    for left in files:
        if not left["group_id"].startswith("multicore/") or not left["dataset"].endswith(" Ch1"):
            continue
        for right in files:
            if right["group_id"] != left["group_id"] or not right["dataset"].endswith(" Ch2"):
                continue
            if left["clock_status"] != "read" or right["clock_status"] != "read":
                continue
            common = sorted(set(left["clock_native"]) & set(right["clock_native"]))
            if not common:
                continue
            a, b = zone_attributes(headers[left["drive_id"]]), zone_attributes(headers[right["drive_id"]])
            if (a["Spacing"] != b["Spacing"] or a["Origin"][0] != b["Origin"][0]
                    or a["Extent"][:2] != b["Extent"][:2] or a["GaugeLength"] != b["GaugeLength"]):
                raise ValueError("Nominally simultaneous cores have incompatible spatial metadata")
            pairs.append({"group_id": left["group_id"], "left_id": left["drive_id"],
                          "right_id": right["drive_id"], "common_clocks": common})
    appearances = defaultdict(int)
    for pair in pairs:
        for key in ("left_id", "right_id"):
            appearances[pair[key]] += 1
    if any(count > 1 for count in appearances.values()):
        raise ValueError("Ambiguous clock pairing; do not split connected acquisitions silently")
    groups = defaultdict(list)
    for pair in sorted(pairs, key=lambda row: (row["group_id"], row["left_id"], row["right_id"])):
        groups[pair["group_id"]].append(pair)
    for values in groups.values():
        if len(values) < 2:
            raise ValueError("Disjoint pairs needed for fit/check")
        for index, pair in enumerate(values):
            pair["role"] = "fit" if index < (len(values) + 1) // 2 else "check"
    return [pair for values in groups.values() for pair in values]


def normalized_context(left, right):
    a, b = encode_robust(left), encode_robust(right)
    context = a.values.astype(np.float64).copy()
    context[~a.valid] = 0
    return context, b.values.astype(np.float64), b.valid


def coefficient(moments):
    aa, ab = moments
    return float(np.clip(ab / aa, -2, 2)) if aa > 0 else 0.


def main():
    audit = json.loads(AUDIT.read_text())
    headers = {row["drive_id"]: row for row in json.loads(HEADERS.read_text())["records"]}
    files = {row["drive_id"]: row for row in audit["files"]}
    pairs = nominal_pairs(list(files.values()), headers)
    if not pairs:
        raise ValueError("No exactly clock-aligned training pairs")
    maps = defaultdict(set)
    for line in (CORPUS / "channel_maps.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["drive_id"] in files:
            maps[(row["drive_id"], row["block_index"])].update(row["source_channel_indices"])
    windows = []
    for pair_index, pair in enumerate(pairs):
        clocks = pair["common_clocks"]
        for clock in sorted({clocks[0], clocks[len(clocks) // 2], clocks[-1]}):
            left = files[pair["left_id"]]
            right = files[pair["right_id"]]
            left_block, right_block = left["clock_native"].index(clock), right["clock_native"].index(clock)
            a = maps[(pair["left_id"], left_block)]
            b = maps[(pair["right_id"], right_block)]
            channels = sorted(a & b)
            if not channels:
                raise ValueError("No shared retained spatial rows")
            length_a, length_b = left["shape"][1], right["shape"][1]
            if length_a != length_b or length_a < 1000:
                raise ValueError("Do not realign or resample incompatible blocks")
            windows.append({**pair, "pair_index": pair_index, "clock_native": clock,
                "left_block": left_block, "right_block": right_block, "start": (length_a - 1000) // 2,
                "length": 1000, "channels": channels, "left_unmatched_rows": len(a - b),
                "right_unmatched_rows": len(b - a)})
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "protocol.json", {
        "kind": "clock-aligned-multicore-opportunity-v1", "pairs": pairs, "windows": windows,
        "clock_audit_sha256": sha256(AUDIT), "header_inventory_sha256": sha256(HEADERS),
        "script_sha256": sha256(Path(__file__)), "parameter_updates": 0,
        "scope": "Only train IDs from provenance audit; original Ch1 to Ch2, common retained rows",
        "gate": "at least10percent original-context fit error-energy reduction in each of two groups",
        "limits": "Nominal alignment only; normalized pooled energy is not full-codec rate or prospective physical macro"})
    (OUTPUT / "probes").mkdir()
    probes = []
    moments = defaultdict(lambda: np.zeros(2))
    for index, window in enumerate(windows):
        values = []
        for label in ("left", "right"):
            file_id = window[f"{label}_id"]
            with h5py.File(RAW / f"{file_id}.h5", "r") as source:
                dataset = source[headers[file_id]["dataset"]]
                raw = np.asarray(dataset[window[f"{label}_block"], window["start"]:window["start"] + 1000,
                                         window["channels"]]).T.copy()
            if raw.dtype != np.float32 or not np.isfinite(raw).all():
                raise ValueError("Native finite float32 paired data required")
            values.append(raw)
        name = f"pair{window['pair_index']:02d}_window{index:03d}.npz"
        np.savez_compressed(OUTPUT / "probes" / name, left=values[0], right=values[1],
                            metadata_json=json.dumps(window, sort_keys=True))
        probes.append({"file": name, "sha256": sha256(OUTPUT / "probes" / name), **window})
        if window["role"] == "fit":
            context, target, valid = normalized_context(*values)
            moments[window["group_id"]] += [float((context[valid] ** 2).sum()),
                                             float((context[valid] * target[valid]).sum())]
    coefficients = {group: coefficient(value) for group, value in moments.items()}
    write_json(OUTPUT / "FROZEN_FIT.json", {"coefficients": coefficients, "probes": probes})
    totals = defaultdict(lambda: np.zeros(3))
    for row in probes:
        with np.load(OUTPUT / "probes" / row["file"], allow_pickle=False) as data:
            context, target, valid = normalized_context(data["left"], data["right"])
        error = target - coefficients[row["group_id"]] * context
        totals[(row["group_id"], row["role"])] += [
            float((error[valid] ** 2).sum()), float((target[valid] ** 2).sum()), int(valid.sum())]
    scores = [{"group_id": group, "role": role, "error_energy": error, "target_energy": energy,
               "ordinary_cells": int(count), "error_energy_ratio": error / energy if energy > 0 else None}
              for (group, role), (error, energy, count) in sorted(totals.items())]
    fit = [row for row in scores if row["role"] == "fit"]
    eligible = len(fit) == 2 and all(row["error_energy_ratio"] is not None and row["error_energy_ratio"] <= .9 for row in fit)
    result = {"pairs": len(pairs), "windows": len(probes), "coefficients": coefficients,
              "scores": scores, "decoded_context_stage_eligible": eligible,
              "parameter_updates": 0, "neural_inference": False,
              "full_codec_measurement": False, "nominal_clock_alignment_only": True}
    write_json(OUTPUT / "ORIGINAL_CONTEXT_RESULTS.json", result)
    (OUTPUT / "ORIGINAL_CONTEXT_COMPLETE").touch()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
