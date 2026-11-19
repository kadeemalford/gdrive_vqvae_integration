"""Bounded train-only clocks/fingerprints and conditional full-block verification."""
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from finetuning.audit_physical_metadata import native
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.run_spatial_opportunity import select

CORPUS = Path("data/finetune/spatial_band_v1_fold0_20261024")
RAW = Path("data/finetune/raw")
OUTPUT = Path("results/experiments/train_clock_copy_audit_fold0_20261115")
HEADERS = Path("results/experiments/physical_metadata_preflight_fold0_20261115/results.json")
FULL_READ_BUDGET = 1024 ** 3


def sample_block(dataset, block, channels):
    if dataset.ndim != 3 or not channels or dataset.dtype.kind != "f":
        raise ValueError("Expected native floating [block,time,channel] dataset and retained channels")
    length = min(256, dataset.shape[1])
    starts = sorted({0, (dataset.shape[1] - length) // 2, dataset.shape[1] - length})
    indices = sorted({channels[index] for index in np.linspace(0, len(channels) - 1, min(8, len(channels)), dtype=int)})
    descriptor = {"shape": list(dataset.shape[1:]), "dtype": dataset.dtype.str,
                  "time_starts": starts, "length": length, "source_channel_indices": indices}
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode())
    values = []
    for start in starts:
        # Selection is explicit and ordered, never reads other datasets.
        array = np.asarray(dataset[block, start:start + length, indices])
        digest.update(np.ascontiguousarray(array).tobytes())
        values.append(array.ravel())
    values = np.concatenate(values)
    bits = np.ascontiguousarray(values).view(np.uint8).reshape(len(values), dataset.dtype.itemsize)
    return {**descriptor, "sample_sha256": digest.hexdigest(), "sampled_cells": len(values),
            "sample_constant_bits": bool(np.all(bits == bits[0])),
            "finite_cells": int(np.isfinite(values).sum()),
            "sample_all_zero": bool(np.all(values == 0))}


def full_block_hash(dataset, block):
    digest = hashlib.sha256(json.dumps({"shape": list(dataset.shape[1:]),
                                      "dtype": dataset.dtype.str}, sort_keys=True).encode())
    read_bytes = 0
    for start in range(0, dataset.shape[1], 1024):
        array = np.ascontiguousarray(dataset[block, start:start + 1024, :])
        digest.update(array.tobytes())
        read_bytes += array.nbytes
    return {"sha256": digest.hexdigest(), "value_bytes_read": read_bytes}


def candidate_pairs(records):
    buckets = defaultdict(list)
    for record in records:
        if not record["sample_constant_bits"]:
            buckets[record["sample_sha256"]].append(record)
    pairs = []
    for values in buckets.values():
        for left, right in itertools.combinations(sorted(values, key=lambda r: (r["drive_id"], r["block"])), 2):
            if left["drive_id"] != right["drive_id"]:
                pairs.append((left, right))
    return sorted(pairs, key=lambda pair: tuple((r["drive_id"], r["block"]) for r in pair))


def main():
    headers = json.loads(HEADERS.read_text())["records"]
    roles = {row["drive_id"]: row["role"] for row in select(CORPUS)}
    if {record["drive_id"] for record in headers} != set(roles):
        raise ValueError("Header inventory must match exactly the authorized training IDs")
    maps = [json.loads(line) for line in (CORPUS / "channel_maps.jsonl").read_text().splitlines()]
    channels = defaultdict(set)
    for mapping in maps:
        if mapping["drive_id"] in roles:
            channels[(mapping["drive_id"], int(mapping["block_index"]))].update(mapping["source_channel_indices"])
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "protocol.json", {
        "kind": "train-clock-copy-audit-v1", "header_inventory_sha256": sha256(HEADERS),
        "manifest_sha256": sha256(CORPUS / "samples_manifest.csv"),
        "maps_sha256": sha256(CORPUS / "channel_maps.jsonl"), "script_sha256": sha256(Path(__file__)),
        "authorized_train_ids": sorted(roles), "roles": roles,
        "sampling": "up to8 retained channels/block, first/center/last256 time samples",
        "full_confirmation_pair_limit": 8, "full_confirmation_value_byte_limit": FULL_READ_BUDGET,
        "parameter_updates": 0, "neural_inference": False, "reserved_signal_access": False})
    files, blocks = [], []
    lookup = {}
    for header in sorted(headers, key=lambda r: r["drive_id"]):
        file_id = header["drive_id"]
        path = RAW / f"{file_id}.h5"
        lookup[file_id] = header["dataset"]
        with h5py.File(path, "r") as source:
            dataset = source[header["dataset"]]
            if list(dataset.shape) != header["shape"] or str(dataset.dtype) != header["dtype"]:
                raise ValueError("Native source schema changed after header census")
            clock_path = dataset.parent.parent.name + "/time"
            record = {"drive_id": file_id, "group_id": header["group_id"], "role": roles[file_id],
                      "dataset": header["dataset"], "shape": list(dataset.shape),
                      "clock_path": clock_path, "clock_status": "missing"}
            if clock_path in source:
                clock = source[clock_path]
                if clock.shape == (dataset.shape[0],) and clock.dtype.kind in "fi":
                    ticks = np.asarray(clock[:])
                    record.update(clock_status="read", clock_native=native(ticks), clock_dtype=str(clock.dtype),
                        clock_sha256=hashlib.sha256(ticks.tobytes()).hexdigest(),
                        clock_strictly_increasing=bool(np.isfinite(ticks).all() and np.all(np.diff(ticks) > 0)))
                else:
                    record["clock_status"] = "unsupported_shape_or_dtype"
            for block in range(dataset.shape[0]):
                selected = sorted(channels[(file_id, block)])
                if not selected:
                    raise ValueError("Cannot silently omit source block without retained physical map")
                sample = sample_block(dataset, block, selected)
                blocks.append({"drive_id": file_id, "group_id": header["group_id"], "role": roles[file_id],
                               "block": block, **sample})
            files.append(record)
    pairs = candidate_pairs(blocks)
    confirmations = []
    cache = {}
    read_bytes = 0
    for left, right in pairs[:8]:
        needed = {(record["drive_id"], record["block"]): record for record in (left, right)}
        additional = sum(int(np.prod(record["shape"])) * np.dtype(record["dtype"]).itemsize
                         for key, record in needed.items() if key not in cache)
        if read_bytes + additional > FULL_READ_BUDGET:
            confirmations.append({"status": "budget_exhausted", "value_bytes_needed": additional})
            break
        for key, record in needed.items():
            if key not in cache:
                with h5py.File(RAW / f"{key[0]}.h5", "r") as source:
                    cache[key] = full_block_hash(source[lookup[key[0]]], key[1])
                read_bytes += cache[key]["value_bytes_read"]
        confirmations.append({
            "left": [left["drive_id"], left["block"]], "right": [right["drive_id"], right["block"]],
            "crosses_fit_check_files": left["role"] != right["role"],
            "complete_block_values_equal": cache[(left["drive_id"], left["block"])]["sha256"]
                                          == cache[(right["drive_id"], right["block"])]["sha256"]})
    clock_groups = defaultdict(list)
    for record in files:
        if record["clock_status"] == "read":
            clock_groups[(record["clock_dtype"], tuple(record["clock_native"]))].append({
                "drive_id": record["drive_id"], "group_id": record["group_id"], "role": record["role"]})
    result = {
        "files": files, "sampled_blocks": blocks,
        "matching_nonconstant_fingerprint_pairs": len(pairs),
        "cross_fit_check_fingerprint_pairs": sum(left["role"] != right["role"] for left, right in pairs),
        "complete_block_confirmations": confirmations, "full_confirmation_value_bytes_read": read_bytes,
        "full_confirmation_limited": len(confirmations) < len(pairs),
        "exact_clock_array_groups": [{"clock_dtype": dtype, "clock_native": list(ticks), "files": values}
                                     for (dtype, ticks), values in clock_groups.items()],
        "parameter_updates": 0, "neural_inference": False, "reserved_signal_access": False,
        "limits": "Sample differences do not certify independence. Full-block equality concerns dataset values, not entire HDF5 file provenance."}
    write_json(OUTPUT / "results.json", result)
    (OUTPUT / "COMPLETE").touch()
    print(json.dumps({"files": len(files), "sampled_blocks": len(blocks),
        "clock_arrays_read": sum(row["clock_status"] == "read" for row in files),
        "clock_array_groups": len(clock_groups), "nonconstant_fingerprint_pairs": len(pairs),
        "complete_block_confirmations": confirmations, "full_confirmation_value_bytes_read": read_bytes}), flush=True)


if __name__ == "__main__":
    main()
