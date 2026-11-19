"""Prepare a fail-closed, metadata-only D2a temporal preflight (report 107).

This module deliberately has no HDF5, NumPy, network, or waveform reader.
Historical clock conversions are hypotheses, not a certified clock contract.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tools.freeze_feasibility_metadata import (
    canonical, select_f2_pool, sha256_file, validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = "data/finetune/spatial_band_v1_fold0_20261024"
FREEZE = "results/experiments/feasibility_definition_20261117/metadata_freeze.json"
HEADERS = "results/experiments/physical_metadata_preflight_fold0_20261115"
PROVENANCE = "results/experiments/train_clock_copy_audit_fold0_20261115"
OUTPUT = "results/experiments/feasibility_temporal_preflight_20261117"
PREFIX = "feasibility-temporal-v1|"
SAMPLES = 5000
CELL_LIMIT = 64_000_000


def selection_key(value):
    return hashlib.sha256((PREFIX + value).encode("utf-8")).hexdigest()


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric coordinate")
    return value


def index(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Expected a nonnegative integer source index")
    return value


def contiguous_runs(channels):
    """Split sorted physical source indices; never bridge a discarded channel."""
    if channels != sorted(set(channels)):
        raise ValueError("Source indices must be unique and increasing")
    runs = []
    for channel in channels:
        index(channel)
        if not runs or channel != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(channel)
    return runs


def assign_components(files, components):
    """Last SHA-ranked component per group is check; conflicts block, never split."""
    by_id = {entry["drive_id"]: entry for entry in files}
    if len(by_id) != len(files):
        raise ValueError("Duplicate candidate ID")
    grouped = defaultdict(set)
    membership = {}
    for component in components:
        cid = component["component_id"]
        for fid in component["members"]:
            if fid in by_id:
                if fid in membership:
                    raise ValueError("File in multiple components")
                membership[fid] = cid
                grouped[by_id[fid]["group_id"]].add(cid)
    if set(membership) != set(by_id):
        raise ValueError("Incomplete candidate component membership")
    proposals = defaultdict(set)
    groups = {}
    for group, ids in sorted(grouped.items()):
        ordered = sorted(ids, key=lambda cid: (selection_key(cid), cid))
        groups[group] = {
            "ordered_components": ordered,
            "status": "candidate" if len(ids) > 1 else "blocked_single_component",
        }
        for cid in ordered:
            proposals[cid].add("blocked" if len(ids) < 2 else
                               "check" if cid == ordered[-1] else "fit")
    conflicts = sorted(cid for cid, roles in proposals.items() if len(roles) > 1)
    # A group touched by a conflict loses its contrast; preserve its other IDs.
    for group, entry in groups.items():
        if set(entry["ordered_components"]) & set(conflicts):
            entry["status"] = "blocked_multigroup_assignment_conflict"
    roles = {}
    for fid, cid in sorted(membership.items()):
        roles[fid] = ("blocked" if len(proposals[cid]) != 1 else next(iter(proposals[cid])))
    return {"groups": groups, "file_roles": roles,
            "multigroup_conflicting_components": conflicts,
            "component_roles": {cid: next(iter(values)) if len(values) == 1 else "blocked"
                                for cid, values in sorted(proposals.items())}}


def conditional_intervals(header, clock, dt):
    """Historical hypothesis ONLY: tick seconds + temporal Origin milliseconds."""
    shape = header["shape"]
    if len(shape) != 3 or shape[1] < SAMPLES or header["dtype"] != "float32":
        raise ValueError("Unsupported raw [block,time,channel] schema")
    zone = "/" + header["dataset"].rsplit("/", 1)[0].strip("/")
    attrs = header["attributes"][zone]
    spacing = attrs["Spacing"]
    origin = attrs["Origin"]
    extent = attrs["Extent"]
    if not math.isclose(finite_number(spacing[1]) * .001, dt, abs_tol=1e-12):
        raise ValueError("Prepared dt differs from historical spacing conversion")
    if extent[3] - extent[2] + 1 != shape[1]:
        raise ValueError("Temporal extent differs from raw shape")
    if clock["dataset"] != header["dataset"] or clock["shape"] != shape:
        raise ValueError("Clock/header schema mismatch")
    ticks = clock.get("clock_native", [])
    if clock.get("clock_status") != "read" or len(ticks) != shape[0]:
        raise ValueError("Missing or malformed block clocks")
    ticks = [finite_number(tick) for tick in ticks]
    if any(right <= left for left, right in zip(ticks, ticks[1:])):
        raise ValueError("Non-increasing block clocks")
    offset = (finite_number(origin[1]) + extent[2] * spacing[1]) * .001
    return {
        "clock_path": clock["clock_path"], "clock_dtype": clock["clock_dtype"],
        "clock_native": ticks, "clock_deltas_native": [b - a for a, b in zip(ticks, ticks[1:])],
        "origin_native": origin, "spacing_native": spacing, "extent_native": extent,
        "block_overlap_native": attrs.get("BlockOverlap"),
        "block_rate_native": attrs.get("BlockRate"),
        "assumed_temporal_offset_s": offset,
        "raw_block_duration_s": shape[1] * dt,
        "intervals": [{"block_index": block, "start": tick + offset,
                       "stop": tick + offset + shape[1] * dt}
                      for block, tick in enumerate(ticks)],
    }


def interval_edges(records):
    """All positive half-open intersections, including within-file block overlap."""
    edges = []
    for left, right in itertools.combinations(records, 2):
        # No assumption that another group/instrument is an independent session.
        for a in left["intervals"]:
            for b in right["intervals"]:
                start, stop = max(a["start"], b["start"]), min(a["stop"], b["stop"])
                if start < stop:
                    edges.append({"left": [left["drive_id"], a["block_index"]],
                                  "right": [right["drive_id"], b["block_index"]],
                                  "intersection_s": [start, stop],
                                  "kind": "conditional_physical_clock_overlap"})
    return edges


def augment_components(components, edges, historical_checks, identities):
    parent = {fid: fid for component in components for fid in component["members"]}

    def find(fid):
        while parent[fid] != fid:
            parent[fid] = parent[parent[fid]]
            fid = parent[fid]
        return fid

    def join(a, b):
        a, b = sorted((find(a), find(b)))
        parent[b] = a

    for component in components:
        for fid in component["members"][1:]:
            join(component["members"][0], fid)
    for edge in edges:
        join(edge["left"][0], edge["right"][0])
    grouped = defaultdict(list)
    for fid in sorted(parent):
        grouped[find(fid)].append(fid)
    result, files, excluded = [], [], []
    for members in grouped.values():
        cid = hashlib.sha256(canonical(members).encode()).hexdigest()
        checks = sorted(set(members) & set(historical_checks))
        result.append({"component_id": cid, "members": members, "check_ids": checks})
        for fid in members:
            entry = {"drive_id": fid, "group_id": identities[fid][1], "component_id": cid}
            (excluded if checks else files).append(entry)
    return sorted(result, key=lambda c: c["component_id"]), files, excluded


def plan_file(fid, rows, maps, header, timing):
    """Select one source block, then every disjoint retained contiguous region."""
    if not rows or any(row["split"] != "train" or row["drive_id"] != fid for row in rows):
        raise ValueError("Only authorized train rows may enter the plan")
    if {row["dataset"] for row in rows} != {header["dataset"]}:
        raise ValueError("Prepared/source dataset mismatch")
    blocks = sorted({int(row["block_index"]) for row in rows})
    block = min(blocks, key=lambda b: (selection_key(f"{fid}|block|{b}"), b))
    if not 0 <= block < header["shape"][0]:
        raise ValueError("Block outside raw shape")
    selected = [row for row in rows if int(row["block_index"]) == block]
    map_ids = sorted({row["channel_map_id"] for row in selected})
    coordinates, owners = {}, defaultdict(list)
    zone = header["attributes"]["/" + header["dataset"].rsplit("/", 1)[0].strip("/")]
    extent, origin, spacing = zone["Extent"], zone["Origin"], zone["Spacing"]
    for mid in map_ids:
        mapping = maps[mid]
        channels = mapping["source_channel_indices"]
        if (mapping["drive_id"] != fid or mapping["block_index"] != block
                or mapping["dataset"] != header["dataset"]
                or mapping["source_channel_count"] != header["shape"][2]
                or mapping["selected_channel_count"] != len(channels)):
            raise ValueError("Map/source identity or shape mismatch")
        runs = contiguous_runs(channels)
        if len(runs) != 1 or not mapping["selected_rows_contiguous"]:
            raise ValueError("Existing retained region is not contiguous")
        if not (len(channels) == len(mapping["metadata_positions_m"])
                == len(mapping["source_extent_channel_indices"])
                == len(mapping["dataset_relative_positions_m"])):
            raise ValueError("Incomplete map coordinates")
        matching_rows = [row for row in selected if row["channel_map_id"] == mid]
        if any(int(row["channels"]) != len(channels) or row["spatial_rows_contiguous"] != "True"
               for row in matching_rows):
            raise ValueError("Manifest/map channel mismatch")
        for local, channel in enumerate(channels):
            if channel >= header["shape"][2] or channel in coordinates:
                raise ValueError("Overlapping maps or channel outside source")
            position = finite_number(mapping["metadata_positions_m"][local])
            extent_index = mapping["source_extent_channel_indices"][local]
            relative = finite_number(mapping["dataset_relative_positions_m"][local])
            if (extent_index != extent[0] + channel
                    or not math.isclose(position, origin[0] + extent_index * spacing[0], abs_tol=1e-9)
                    or not math.isclose(relative, channel * spacing[0], abs_tol=1e-9)):
                raise ValueError("Source index/physical map formula mismatch")
            coordinates[channel] = (position, relative, extent_index)
            owners[channel].append(mid)
    interval = timing["intervals"][block]
    regions = []
    for number, channels in enumerate(contiguous_runs(sorted(coordinates))):
        regions.append({
            "region_index": number, "source_channel_indices": channels,
            "source_extent_channel_indices": [coordinates[c][2] for c in channels],
            "metadata_positions_m": [coordinates[c][0] for c in channels],
            "dataset_relative_positions_m": [coordinates[c][1] for c in channels],
            "channel_map_ids": sorted({mid for c in channels for mid in owners[c]}),
            "time_start_source_index": 0, "time_stop_source_index_exclusive": SAMPLES,
            "time_extent_indices_half_open": [extent[2], extent[2] + SAMPLES],
            "time_relative_to_block_first_sample_s": [0, SAMPLES * float(rows[0]["dt_s"])],
            "conditional_clock_interval_s": [
                interval["start"], interval["start"] + SAMPLES * float(rows[0]["dt_s"])],
            "cells": len(channels) * SAMPLES,
        })
    if not regions:
        raise ValueError("No retained regions")
    return {
        "drive_id": fid, "group_id": rows[0]["group_id"],
        "drive_path": rows[0]["drive_path"], "dataset": header["dataset"],
        "source_shape": header["shape"], "source_dtype": header["dtype"],
        "block_index": block, "dt_s_pipeline_assumption": float(rows[0]["dt_s"]),
        "unit_dataset_label_only": header["dataset"].split("/")[-1],
        "unit_calibration_certified": False, "spatial_coordinates_externally_calibrated": False,
        "raw_block_conditional_clock_interval_s": [interval["start"], interval["stop"]],
        "regions": regions, "cells": sum(region["cells"] for region in regions),
        "source_block_reads": 1,
    }


def build_preflight(root):
    root = Path(root)
    inputs = {}

    def record(path):
        path = Path(path)
        full = path if path.is_absolute() else root / path
        inputs[str(full.relative_to(root))] = sha256_file(full)
        return full

    def load(path):
        return json.loads(record(path).read_text(encoding="utf-8"))

    freeze = load(FREEZE)
    for evidence in freeze["input_sha256"].values():
        path = Path(evidence["path"])
        if sha256_file(record(path)) != evidence["sha256"]:
            raise ValueError("Frozen input hash changed")
    manifest_path = record(f"{CORPUS}/samples_manifest.csv")
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities, _ = validate_manifest(rows)
    train = {fid for fid, (split, _) in identities.items() if split == "train"}
    provenance = load(f"{PROVENANCE}/results.json")
    provenance_protocol = load(f"{PROVENANCE}/protocol.json")
    headers_path = record(f"{HEADERS}/results.json")
    headers = json.loads(headers_path.read_text(encoding="utf-8"))["records"]
    header_protocol = load(f"{HEADERS}/protocol.json")
    maps_path = record(f"{CORPUS}/channel_maps.jsonl")
    for actual, expected in [
        (inputs[str(maps_path.relative_to(root))], provenance_protocol["maps_sha256"]),
        (inputs[str(headers_path.relative_to(root))], provenance_protocol["header_inventory_sha256"]),
        (inputs[str(manifest_path.relative_to(root))], header_protocol["manifest_sha256"]),
        (inputs[str(manifest_path.relative_to(root))], provenance_protocol["manifest_sha256"]),
    ]:
        if actual != expected:
            raise ValueError("Historical metadata chain mismatch")
    if set(header_protocol["train_file_ids"]) != train or set(provenance_protocol["authorized_train_ids"]) != train:
        raise ValueError("Historical authorization differs from train manifest")
    checks = load(freeze["input_sha256"]["checks"]["path"])["checks"]
    pool = select_f2_pool(rows, checks, provenance)
    if pool != freeze["f2_candidate_pool"]:
        raise ValueError("Recomputed pool differs from metadata freeze")
    header_by_id = {r["drive_id"]: r for r in headers}
    clock_by_id = {r["drive_id"]: r for r in provenance["files"]}
    if set(header_by_id) != train or len(headers) != len(train) or set(clock_by_id) != train:
        raise ValueError("Header/clock inventory must cover train only")
    by_file = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_file[row["drive_id"]].append(row)
    timing = {}
    for fid in sorted(train):
        dts = {float(row["dt_s"]) for row in by_file[fid]}
        if len(dts) != 1 or dts != {0.00005}:
            raise ValueError("Unexpected or inconsistent prepared sample period")
        timing[fid] = {"drive_id": fid, **conditional_intervals(
            header_by_id[fid], clock_by_id[fid], next(iter(dts)))}
    # Historical maps include reserved metadata; only parse train records into the planner.
    maps = {}
    with maps_path.open(encoding="utf-8") as stream:
        for line in stream:
            mapping = json.loads(line)
            if mapping["drive_id"] in train:
                mid = mapping["channel_map_id"]
                if mid in maps and maps[mid] != mapping:
                    raise ValueError("Conflicting channel map ID")
                maps[mid] = mapping
    edges = interval_edges(list(timing.values()))
    components, conditional_files, excluded = augment_components(
        pool["components"], edges, pool["historical_check_ids"], identities)
    assignments = assign_components(pool["files"], pool["components"])
    conditional_assignment = assign_components(conditional_files, components)
    pool_ids = {entry["drive_id"] for entry in pool["files"]}
    check_ids = set(pool["historical_check_ids"])
    direct_pool_check_edges = [
        edge for edge in edges
        if (edge["left"][0] in pool_ids and edge["right"][0] in check_ids)
        or (edge["right"][0] in pool_ids and edge["left"][0] in check_ids)
    ]
    remaining_ids = {entry["drive_id"] for entry in conditional_files}
    multigroup_components = [
        {"component_id": component["component_id"], "members": component["members"],
         "groups": sorted({identities[fid][1] for fid in component["members"]})}
        for component in pool["components"]
        if set(component["members"]) & pool_ids
        and len({identities[fid][1] for fid in component["members"]}) > 1
    ]
    plans = []
    for entry in pool["files"]:
        fid = entry["drive_id"]
        plan = plan_file(fid, by_file[fid], maps, header_by_id[fid], timing[fid])
        plan.update(component_id=entry["component_id"],
                    candidate_role=assignments["file_roles"][fid],
                    executable=False)
        plans.append(plan)
    cells = sum(plan["cells"] for plan in plans)
    budget_ok = cells <= CELL_LIMIT
    within_file = []
    for fid, entry in timing.items():
        for a, b in itertools.combinations(entry["intervals"], 2):
            if max(a["start"], b["start"]) < min(a["stop"], b["stop"]):
                within_file.append({"drive_id": fid, "blocks": [a["block_index"], b["block_index"]],
                                    "intersection_s": [max(a["start"], b["start"]),
                                                       min(a["stop"], b["stop"])]})
    sources = [
        "tools/freeze_feasibility_metadata.py", "finetuning/audit_train_provenance.py",
        "finetuning/audit_physical_metadata.py", "finetuning/audit_febus_corpus.py",
        "acquisition/summarize_external_acquisitions.py",
        "progress_reports/106-FEBUS_FEASIBILITY_AND_RESEARCH_REDIRECTION.md",
        "progress_reports/107-FEASIBILITY_CONTRACT_AND_EXECUTION_HANDOFF.md",
        "tools/prepare_feasibility_temporal.py", "tests/test_feasibility_temporal.py",
    ]
    for source in sources:
        record(source)
    return {
        "kind": "feasibility-temporal-preflight-v1",
        "status": "blocked_clock_semantics_not_certified",
        "blockers": ["Source/time unit, epoch, reference point and relation to Zone Origin/Extent are not certified",
                     "BlockOverlap=5 and BlockRate=1000 units/placement are not documented by instrument evidence",
                     "multicore/C1 and C4 has only one frozen component; no component fit/check contrast"],
        "clock_contract": {
            "certified": False, "external_instrument_semantics_evidence": None,
            "historical_hypothesis": "Source/time in seconds; Zone temporal Origin/Spacing in milliseconds; sample k = tick + (Origin[1] + (Extent[2]+k)*Spacing[1])/1000",
            "hypothesis_sources": ["finetuning/audit_febus_corpus.py:audit_raw",
                                   "acquisition/summarize_external_acquisitions.py"],
            "warning": "Historical implementation and plausible 1s steps/1.05s blocks do not certify clock semantics. Never use AcqStartTime as exported block origin.",
            "physical_intervals_verified": False,
            "interval_boundary_convention": "half-open sample-support intervals; stop is one sample after last sample",
        },
        "selection": {"prefix": PREFIX, "block_key": "drive_id|block|decimal_zero_based_block_index",
                      "blocks_per_file": 1, "samples_per_region": SAMPLES,
                      "component_order": "ascending (SHA256(prefix+component_id), component_id); last check, others fit",
                      "signal_tuning": False, "npz_concatenation": False,
                      "indices": "zero-based source dataset [block,time,channel], stop exclusive; extent indices are separate metadata coordinates",
                      "retained_coverage": "union of existing selected-block maps split at every source-index gap; no new mask"},
        "candidate_plan": {"files": plans, "assignment": assignments,
                           "file_count": len(plans), "cell_count": cells,
                           "multigroup_components_kept_whole": multigroup_components,
                           "role_file_counts": dict(Counter(p["candidate_role"] for p in plans))},
        "conditional_overlap_audit": {
            "status": "hypothesis_only_not_physical_certification",
            "authorized_train_file_count": len(train),
            "historical_check_file_count": len(check_ids),
            "raw_block_interval_count": sum(len(t["intervals"]) for t in timing.values()),
            "all_raw_blocks_not_only_selected_blocks": True,
            "cross_file_intersection_count": len(edges),
            "direct_frozen_pool_to_historical_check_intersections": direct_pool_check_edges,
            "remaining_file_count": len(conditional_files),
            "frozen_pool_collapses_under_hypothesis": not bool(remaining_ids & pool_ids),
            "remaining_group_file_counts": dict(sorted(Counter(
                entry["group_id"] for entry in conditional_files).items())),
            "all_authorized_train_clocks": list(timing.values()),
            "cross_file_block_intersections": edges,
            "within_file_block_intersections": within_file,
            "augmented_components": components,
            "remaining_files": conditional_files,
            "excluded_connected_to_historical_checks": excluded,
            "newly_excluded_frozen_pool_ids": sorted(
                pool_ids - remaining_ids),
            "assignment": conditional_assignment,
            "must_reconcile_before_execution": True,
        },
        "executable_plan": {"status": "blocked", "files": [], "authorized_waveform_cells": 0},
        "budget": {"max_source_cells": CELL_LIMIT, "candidate_source_cells": cells,
                   "candidate_within_budget": budget_ok, "waveform_cells_read": 0,
                   "budget_exceeded_blocks_execution": not budget_ok,
                   "read_contract": "only selected region slices, one source block per file; no full raw-block read; reuse same cells for all nine precisions"},
        "estimand": "within-development diagnostic, not independent sessions or unseen mature-model training data",
        "all_ten_configuration_pass_available": False,
        "input_and_code_sha256": dict(sorted(inputs.items())),
        "access": {"raw_hdf5_opened": False, "waveforms_read": False, "remote": False,
                   "gpu": False, "reserved_signal_access": False, "edas_access": False},
        "next_required_evidence": [
            "Instrument/export documentation tying Source/time to raw block samples, temporal units and overlap placement",
            "Certify or revise the conditional interval graph (including historical checks) before authorizing any fit/check waveform read",
            "Return changed exclusions, multigroup conflicts or missing contrasts to scientific owner; do not replace IDs or split components",
        ],
    }


def write_preflight(output, metadata):
    """Idempotent exclusive creation; divergent evidence requires a new version."""
    output = Path(output)
    payload = (json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    if output.exists():
        if output.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite divergent evidence: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT / "temporal_preflight.json")
    args = parser.parse_args()
    result = build_preflight(ROOT)
    write_preflight(args.output, result)
    print(json.dumps({"status": result["status"], "candidate_files": result["candidate_plan"]["file_count"],
                      "candidate_cells": result["candidate_plan"]["cell_count"],
                      "executable_files": 0, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
