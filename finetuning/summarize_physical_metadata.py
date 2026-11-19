"""Summarize header coverage/confounding without reading any HDF5 values."""
import json
import zlib
from collections import defaultdict
from pathlib import Path

from finetuning.audit_prospective_preparation import write_json

ROOT = Path("results/experiments/physical_metadata_preflight_fold0_20261115")


def summarize(records):
    clusters = defaultdict(list)
    geometry = defaultdict(set)
    extended = defaultdict(set)
    per_file = {}
    conflicts = []
    for record in records:
        if not record["source_exists"]:
            raise ValueError("Cannot establish complete physical coverage with missing sources")
        parent = "/" + str(Path(record["dataset"]).parent).lstrip("/")
        zone = record["attributes"][parent]
        spacing = zone["Spacing"]
        # Match the existing preparation's explicitly versioned ms-to-s convention.
        if (any(float(value) != float(spacing[0]) for value in record["prepared_dx_m"])
                or any(float(value) != float(spacing[1]) * 1e-3 for value in record["prepared_dt_s"])):
            conflicts.append(record["drive_id"])
        fields = {key: zone[key][0] for key in ("GaugeLength", "PulseWidth", "PulseRateFreq",
                  "SamplingRate", "SamplingRes", "Oversampling", "AmpliPower", "DerivationTime")}
        fields.update(dx_m=float(spacing[0]), dt_s=float(spacing[1]) * 1e-3)
        per_file[record["drive_id"]] = fields
        geometry_key = tuple(fields[key] for key in ("dx_m", "GaugeLength", "PulseWidth", "DerivationTime", "PulseRateFreq"))
        extended_key = tuple(sorted(fields.items()))
        geometry[geometry_key].add(record["group_id"])
        extended[extended_key].add(record["group_id"])
        clusters[(zone["Hostname"], tuple(zone["AcqStartTime"]))].append({
            "drive_id": record["drive_id"], "group_id": record["group_id"]})
    groups = defaultdict(set)
    for record in records:
        groups[record["group_id"]].add(tuple(sorted(per_file[record["drive_id"]].items())))
    registry = zlib.compress(json.dumps(per_file, sort_keys=True, separators=(",", ":")).encode(), 6)
    return {
        "files": len(records), "prepared_spacing_conflict_files": conflicts,
        "physical_combinations": len(geometry), "extended_hardware_combinations": len(extended),
        "distinct_extended_vectors_per_group": {group: len(values) for group, values in groups.items()},
        "header_hostname_start_combinations": len(clusters),
        "hostname_start_clusters": [{"hostname": host, "start_native": list(start), "files": rows}
                                    for (host, start), rows in clusters.items()],
        "prospective_numeric_metadata_registry_bytes": len(registry), "numeric_fields_by_file": per_file,
        "dataset_values_read": False, "parameter_updates": 0,
        "units": "dx/dt match existing preparation. Other fields retain native header semantics; no inferred pulse units.",
        "limits": "Repeated header start time neither proves duplicates nor establishes independent acquisitions; hardware vectors constant within groups",
    }


def main():
    original = json.loads((ROOT / "results.json").read_text())
    result = summarize(original["records"])
    write_json(ROOT / "CONSISTENCY_AND_PROVENANCE.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in (
        "hostname_start_clusters", "numeric_fields_by_file")}), flush=True)


if __name__ == "__main__":
    main()
