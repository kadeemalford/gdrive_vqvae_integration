"""Validate candidate provenance metadata only; never opens source waveforms."""
import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

FIELDS = (
    "source_id", "session_id", "parent_source_id", "is_derived", "acquired_at_utc",
    "device_id", "fiber_id", "site_id", "excitation_protocol", "variable", "unit",
    "source_dtype", "dt_s", "dx_m", "gauge_length_m", "pulse_width_value",
    "pulse_width_unit", "derivation_time_s", "partition_role", "source_sha256",
)


def validate(rows):
    if not rows:
        raise ValueError("At least one real acquisition record is required; template is not data")
    ids, roles = set(), {}
    for row in rows:
        if any(field not in row for field in FIELDS):
            raise ValueError("Acquisition manifest is missing required columns")
        if any(not row[field].strip() for field in FIELDS if field != "parent_source_id"):
            raise ValueError("Explicit acquisition metadata and units required")
        if row["source_id"] in ids:
            raise ValueError("Duplicate source ID")
        ids.add(row["source_id"])
        if row["is_derived"] not in ("true", "false"):
            raise ValueError("is_derived must be true or false")
        if row["is_derived"] == "true" and not row["parent_source_id"]:
            raise ValueError("Derived sources require parent provenance")
        if row["parent_source_id"] == row["source_id"]:
            raise ValueError("Source cannot derive from itself")
        if row["partition_role"] not in ("development", "independent_confirmation", "reserved"):
            raise ValueError("Unknown acquisition partition role")
        # Same session must never straddle development/confirmation/reserve.
        key = (row["device_id"], row["session_id"])
        previous = roles.setdefault(key, row["partition_role"])
        if previous != row["partition_role"]:
            raise ValueError("One acquisition session crosses partition roles")
        try:
            timestamp = datetime.fromisoformat(row["acquired_at_utc"].replace("Z", "+00:00"))
            if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
                raise ValueError("UTC timestamp required")
        except ValueError as error:
            raise ValueError("Acquisition timestamp must specify UTC") from error
        for field in ("dt_s", "dx_m", "gauge_length_m", "pulse_width_value", "derivation_time_s"):
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite physical metadata required: {field}")
        if row["pulse_width_unit"] not in ("s", "ms", "us", "ns", "ps"):
            raise ValueError("Resolve pulse width to an explicit time unit before admission")
        if not re.fullmatch("[0-9a-fA-F]{64}", row["source_sha256"]):
            raise ValueError("Explicit SHA-256 source fingerprint required")
    by_id = {row["source_id"]: row for row in rows}
    for row in rows:
        parent = by_id.get(row["parent_source_id"])
        if parent is not None and parent["partition_role"] != row["partition_role"]:
            raise ValueError("Known parent and derivative cross partition roles")
    return {"metadata_valid": True, "records": len(rows), "device_session_keys": len(roles),
            "waveforms_opened": False, "independence_certified": False,
            "limits": "Metadata admission only; verify actual source hashes, copies, clocks and session documentation separately"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open() as stream:
        result = validate(list(csv.DictReader(stream)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
