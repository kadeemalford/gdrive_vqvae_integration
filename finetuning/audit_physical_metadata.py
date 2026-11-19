"""Read only train HDF5 attributes and dataset schema; never read dataset values."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from finetuning.audit_prospective_preparation import sha256, write_json

CORPUS = Path("data/finetune/spatial_band_v1_fold0_20261024")
OUTPUT = Path("results/experiments/physical_metadata_preflight_fold0_20261115")
FIELDS = {
    "spacing": ("spacing", "dx"),
    "gauge": ("gauge",),
    "pulse": ("pulse",),
    "derivation": ("deriv",),
    "acquisition_provenance": ("date", "time", "uuid", "serial", "acquisition"),
}


def native(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="backslashreplace")
    if isinstance(value, np.ndarray):
        return [native(item) for item in value.tolist()] if value.ndim else native(value.item())
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def header_only(path, dataset):
    record = {"attributes": {}, "physical_fields": {field: [] for field in FIELDS}}
    with h5py.File(path, "r") as source:
        target = source[dataset]
        if not isinstance(target, h5py.Dataset):
            raise ValueError("Prepared target does not identify a dataset")
        record.update(dataset=dataset, shape=list(target.shape), dtype=str(target.dtype))
        chain = [target]
        while chain[-1].name != "/":
            chain.append(chain[-1].parent)
        for node in reversed(chain):
            attributes = {name: native(value) for name, value in node.attrs.items()}
            record["attributes"][node.name] = attributes
            for name, value in attributes.items():
                for field, keywords in FIELDS.items():
                    if any(term in name.lower() for term in keywords):
                        record["physical_fields"][field].append({"path": node.name, "name": name, "value": value})
    return record


def main():
    rows = list(csv.DictReader((CORPUS / "samples_manifest.csv").open()))
    files = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            files[row["drive_id"]].append(row)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "protocol.json", {
        "kind": "train-header-only-physical-metadata-v1", "manifest_sha256": sha256(CORPUS / "samples_manifest.csv"),
        "script_sha256": sha256(Path(__file__)), "train_file_ids": sorted(files),
        "dataset_values_read": False, "parameter_updates": 0, "neural_inference": False,
        "reserved_hdf5_opened": False, "fields": FIELDS})
    records = []
    for file_id, samples in sorted(files.items()):
        row = samples[0]
        datasets = {sample["dataset"] for sample in samples}
        if len(datasets) != 1:
            raise ValueError("One prepared source dataset per file required")
        path = Path("data/finetune/raw") / f"{file_id}.h5"
        record = {"drive_id": file_id, "group_id": row["group_id"], "drive_path": row["drive_path"],
                  "prepared_dx_m": sorted({sample["dx_m"] for sample in samples}),
                  "prepared_dt_s": sorted({sample["dt_s"] for sample in samples}),
                  "source_exists": path.exists()}
        if path.exists():
            record.update(header_only(path, row["dataset"]))
        records.append(record)
    coverage = {field: sum(bool(record.get("physical_fields", {}).get(field)) for record in records) for field in FIELDS}
    combinations = defaultdict(list)
    for record in records:
        physical = record.get("physical_fields", {})
        signature = json.dumps({field: [{"name": item["name"], "value": item["value"]}
                                      for item in physical.get(field, [])]
                                for field in ("spacing", "gauge", "pulse", "derivation")}, sort_keys=True)
        combinations[signature].append(record["group_id"])
    result = {"records": records, "files": len(records), "source_files_present": sum(r["source_exists"] for r in records),
              "groups": dict(Counter(r["group_id"] for r in records)), "field_file_coverage": coverage,
              "physical_combinations": [{"signature": json.loads(key), "groups": dict(Counter(groups))}
                                        for key, groups in combinations.items()],
              "dataset_values_read": False, "neural_inference": False, "parameter_updates": 0,
              "limits": "Header presence does not establish units, causal effects, independent acquisitions or core synchronization"}
    write_json(OUTPUT / "results.json", result)
    (OUTPUT / "COMPLETE").touch()
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "physical_combinations")}), flush=True)


if __name__ == "__main__":
    main()
