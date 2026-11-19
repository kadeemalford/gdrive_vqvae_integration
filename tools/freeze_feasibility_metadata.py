"""Freeze deterministic feasibility metadata without opening signal arrays."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from finetuning.feasibility_context import plan_context_windows


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/finetune/spatial_band_v1_fold0_20261024/samples_manifest.csv"
CHECKS = "results/experiments/physical_conditioning_20261115/scientific/constant/protocol.json"
PROVENANCE = "results/experiments/train_clock_copy_audit_fold0_20261115/results.json"
REFERENCE = "results/experiments/prospective_p0_v2_fold0_20261114/protocol.json"
CONFIG = "config/feasibility_protocol_v1.json"


def canonical(value):
    """Encode structured metadata deterministically."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def selection_key(value):
    return hashlib.sha256(("feasibility-v1|" + value).encode("utf-8")).hexdigest()


def validate_manifest(rows):
    """Reject ambiguous identities and any file crossing dataset splits."""
    files = {}
    samples = {}
    for row in rows:
        for field in ("drive_id", "sample_path", "group_id", "split"):
            if not row.get(field):
                raise ValueError(f"Missing manifest field: {field}")
        file_id = row["drive_id"]
        identity = (row["split"], row["group_id"])
        if file_id in files and files[file_id] != identity:
            raise ValueError(f"Cross-split or inconsistent group ID: {file_id}")
        files[file_id] = identity
        path = row["sample_path"]
        if path in samples and samples[path] != row:
            raise ValueError(f"Conflicting manifest sample: {path}")
        samples[path] = dict(row)
    return files, samples


def validate_checks(rows, checks):
    """Match every historical check against all common manifest fields."""
    files, samples = validate_manifest(rows)
    matched = {}
    for check in checks:
        path = check["sample_path"]
        if path not in samples:
            raise ValueError(f"Historical check missing from manifest: {path}")
        row = samples[path]
        for field in row.keys() & check.keys():
            if row[field] != check[field]:
                raise ValueError(f"Historical check mismatch: {path}: {field}")
        if row["split"] != "train":
            raise ValueError(f"Historical check is not train: {path}")
        matched[path] = row
    if not matched:
        raise ValueError("Historical checks are empty")
    return files, list(matched.values())


def select_f1_panel(rows, checks):
    """Select two historical check files per group and one historical row each."""
    _, matched = validate_checks(rows, checks)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in matched:
        grouped[row["group_id"]][row["drive_id"]].append(row)
    panel = []
    for group in sorted(grouped):
        files = grouped[group]
        if len(files) < 2:
            raise ValueError(f"F1 requires two historical check files: {group}")
        for file_id in sorted(files, key=lambda value: (selection_key(value), value))[:2]:
            panel.append(dict(min(files[file_id],
                                  key=lambda row: (selection_key(row["sample_path"]),
                                                   row["sample_path"]))))
    return panel


def select_f2_pool(rows, checks, provenance):
    """Exclude all train components connected to any historical check ID."""
    identities, matched = validate_checks(rows, checks)
    train = {key for key, (split, _) in identities.items() if split == "train"}
    check_ids = {row["drive_id"] for row in matched}
    records = {}
    for record in provenance["files"]:
        file_id = record["drive_id"]
        if file_id not in train:
            raise ValueError(f"Provenance ID absent or not train: {file_id}")
        if file_id in records:
            raise ValueError(f"Duplicate provenance ID: {file_id}")
        if record["group_id"] != identities[file_id][1]:
            raise ValueError(f"Provenance group mismatch: {file_id}")
        records[file_id] = record
    if set(records) != train:
        raise ValueError(f"Provenance does not cover train IDs: {sorted(train - records.keys())}")

    parent = {file_id: file_id for file_id in train}
    edges = []

    def find(file_id):
        while parent[file_id] != file_id:
            parent[file_id] = parent[parent[file_id]]
            file_id = parent[file_id]
        return file_id

    def connect(members, cause):
        members = sorted(set(members))
        if any(file_id not in train for file_id in members):
            raise ValueError("Provenance relation contains absent or non-train ID")
        if len(members) < 2:
            return
        for other in members[1:]:
            left, right = sorted((find(members[0]), find(other)))
            parent[right] = left
        edges.append({"members": members, **cause})

    ticks = defaultdict(set)
    for file_id in sorted(records):
        record = records[file_id]
        if record.get("clock_status") == "read":
            dtype = record["clock_dtype"]
            for tick in record["clock_native"]:
                # Native numeric equality is exact here; no temporal tolerance is used.
                if isinstance(tick, bool) or not isinstance(tick, (float, int)):
                    raise ValueError(f"Invalid clock tick: {file_id}")
                canonical(tick)  # Reject non-finite JSON numbers.
                ticks[(dtype, tick)].add(file_id)
    for (dtype, tick), members in sorted(ticks.items()):
        connect(members, {"kind": "exact_block_timestamp",
                          "clock_dtype": dtype, "tick": tick})
    for confirmation in provenance["complete_block_confirmations"]:
        if confirmation.get("complete_block_values_equal") is True:
            connect([confirmation["left"][0], confirmation["right"][0]],
                    {"kind": "complete_block_values_equal",
                     "left": confirmation["left"], "right": confirmation["right"]})
    for group in provenance["exact_clock_array_groups"]:
        connect([record["drive_id"] for record in group["files"]],
                {"kind": "exact_clock_array_group",
                 "clock_dtype": group["clock_dtype"],
                 "clock_native": group["clock_native"]})

    grouped = defaultdict(list)
    for file_id in sorted(train):
        grouped[find(file_id)].append(file_id)
    components = []
    component_for = {}
    for members in grouped.values():
        component_id = hashlib.sha256(canonical(members).encode("utf-8")).hexdigest()
        component = {"component_id": component_id, "members": members,
                     "check_ids": sorted(check_ids.intersection(members))}
        components.append(component)
        for file_id in members:
            component_for[file_id] = component
    components.sort(key=lambda item: item["component_id"])
    edges = sorted({canonical(edge): edge for edge in edges}.values(), key=canonical)
    causes = defaultdict(list)
    for edge in edges:
        causes[component_for[edge["members"][0]]["component_id"]].append(edge)
    ledger = []
    pool = []
    for file_id in sorted(train):
        component = component_for[file_id]
        entry = {"drive_id": file_id, "group_id": identities[file_id][1],
                 "component_id": component["component_id"]}
        if component["check_ids"]:
            ledger.append({**entry, "historical_check": file_id in check_ids,
                           "causes": (["historical_check_id"] if file_id in check_ids else [])
                           + ["connected_to_historical_check_component"],
                           "connected_check_ids": component["check_ids"]})
        else:
            pool.append(entry)
    return {
        "status": "pending_temporal_protocol",
        "split_status": "pending_temporal_protocol",
        "fit_check_assignment": None,
        "segment_assignment": None,
        "independence_claim": False,
        "absence_of_overlap_claim": False,
        "historical_check_ids": sorted(check_ids),
        "file_count": len(pool),
        "group_file_counts": dict(sorted(Counter(item["group_id"] for item in pool).items())),
        "files": pool,
        "components": components,
        "exclusion_ledger": ledger,
        "component_evidence": [{"component_id": component["component_id"],
                                "relations": causes[component["component_id"]]}
                               for component in components],
        "limits": "Known metadata links only; no absence of overlap or independence is established.",
    }


def sha256_file(path):
    """Hash only explicitly permitted metadata and source file formats."""
    path = Path(path)
    if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".py", ".md"}:
        raise ValueError(f"Disallowed input format: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected JSON input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def historical_hashes(root):
    """Require exactly one Markdown report for every index from 1 to 106."""
    reports = defaultdict(list)
    for path in (root / "progress_reports").glob("*.md"):
        prefix = path.name.split("-", 1)[0]
        if prefix.isdigit() and 1 <= int(prefix) <= 106:
            reports[int(prefix)].append(path)
    if any(len(reports[index]) != 1 for index in range(1, 107)):
        raise ValueError("Historical reports 1..106 must be complete and unambiguous")
    return {str(index): {"path": str(reports[index][0].relative_to(root)),
                         "sha256": sha256_file(reports[index][0])}
            for index in range(1, 107)}


def build_freeze(root, manifest, checks, provenance, reference, config):
    """Read metadata inputs and construct the freeze without signal access."""
    inputs = {"manifest": manifest, "checks": checks, "provenance": provenance,
              "manifest_reference": reference, "config": config}
    hashes = {key: {"path": str(path), "sha256": sha256_file(path)}
              for key, path in inputs.items()}
    if hashes["manifest"]["sha256"] != read_json(reference)["manifest_sha256"]:
        raise ValueError("Manifest SHA256 differs from historical protocol")
    plan = read_json(config)
    if (plan.get("kind") != "febus-feasibility-definition-v1"
            or plan["f1"]["panel_seed"] != "feasibility-v1"
            or plan["f1"]["files_per_group"] != 2
            or plan["f1"]["rows_per_file"] != 1):
        raise ValueError("Selection configuration differs from the implemented protocol")
    if manifest.suffix.lower() != ".csv":
        raise ValueError("Expected CSV manifest")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    historical_checks = read_json(checks)["checks"]
    panel = select_f1_panel(rows, historical_checks)
    pool = select_f2_pool(rows, historical_checks, read_json(provenance))
    if len(pool["historical_check_ids"]) != 43:
        raise ValueError("Expected all 43 historical check IDs")
    return {
        "kind": "feasibility-metadata-freeze-v1",
        "input_sha256": hashes,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": hashes["config"]["sha256"],
        "feasibility_metrics_sha256": sha256_file(root / "finetuning/feasibility_metrics.py"),
        "feasibility_context_sha256": sha256_file(root / "finetuning/feasibility_context.py"),
        "historical_report_sha256": historical_hashes(root),
        "f1_panel": {
            "rows": panel, "row_count": len(panel), "file_count": len(panel),
            "group_file_counts": dict(sorted(Counter(row["group_id"] for row in panel).items())),
            "historical_train_checks": True, "previously_seen": True,
            "independent": False,
            "status": "historical_seen_train_checks_not_independent",
            "context_plans": {
                row["sample_path"]: plan_context_windows(int(row["channels"])) for row in panel
            },
        },
        "f2_candidate_pool": pool,
        "parameter_updates": 0,
        "neural_inference": False,
        "raw_or_prepared_waveforms_read": False,
        "gpu_use": False,
    }


def write_freeze(output, metadata):
    """Create a new output, permitting only an identical existing freeze."""
    payload = (json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    output = Path(output)
    if output.name != "metadata_freeze.json":
        raise ValueError("--output must name metadata_freeze.json")
    if output.exists():
        if output.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different output: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    for name, default in (("manifest", MANIFEST), ("checks", CHECKS),
                          ("provenance", PROVENANCE), ("reference", REFERENCE),
                          ("config", CONFIG)):
        parser.add_argument("--" + name, type=Path, default=ROOT / default)
    args = parser.parse_args()
    metadata = build_freeze(ROOT, args.manifest, args.checks, args.provenance,
                            args.reference, args.config)
    write_freeze(args.output, metadata)


if __name__ == "__main__":
    main()
