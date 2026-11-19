"""CPU-only report122 audit using historical bundles, geometry and endpoint JSON."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib

from compression.full_tile_compact_bundle import (
    ENTRY, HEADER, Reader, geometry_rows, literal, pack,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/experiments/feasibility_context_alignment_20261118/mature/mature/endpoint"
OUTPUT = ROOT / "results/experiments/feasibility_compact_bundle_20261118"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def independent_proof(original, mode, shape, payload):
    """Parse historical framing independently; never call compact_tile to verify."""
    if zlib.crc32(original[:-4]) != struct.unpack("<I", original[-4:])[0]:
        raise ValueError("Original bundle CRC mismatch")
    magic, h, w, n = struct.unpack_from("<4sIII", original)
    if magic != b"FPB1" or shape != (h, w):
        raise ValueError("Coverage mismatch")
    if not mode:
        if payload != original:
            raise ValueError("Fallback changed")
        return {"branch": "intact_fallback", "overlay_entries": None}
    kind, start, rows, size = struct.unpack_from("<BIII", original, 16)
    outer = original[29:-4]
    if n != 1 or (kind, start, rows, size) != (0, 0, h, len(outer)):
        raise ValueError("Wrapper elision not proven")
    magic, length, count, crc = struct.unpack_from("<4sIII", outer)
    body = outer[16:]
    if magic != b"FZC1" or zlib.crc32(body) != crc or body[:length] != payload:
        raise ValueError("Inner frame not identical")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(body[length:], 8 * count + 1)
    if (len(raw) != count * 8 or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError("Overlay termination mismatch")
    positions = struct.unpack(f"<{count}I", raw[:count * 4])
    bits = struct.unpack(f"<{count}I", raw[count * 4:])
    if (any(p >= h * w for p in positions)
            or any(a >= b for a, b in zip(positions, positions[1:]))
            or any(b & 0x7fffffff for b in bits)):
        raise ValueError("Overlay grammar mismatch")
    if payload[:4] == b"FAK1":
        if count:
            raise ValueError("FAK1 nonempty overlay cannot be elided")
        # Identical frame, same installed decoder, overlay is identity.
        return {"branch": "symbolic_decoder_branch_equivalence", "overlay_entries": 0,
                "inner_frame_sha256": sha(payload), "waveform_roundtrip_performed": False}
    # Independent literal path: the actual installed method, unbound, needs no model.
    from compression.full_tile_vq_stream import FullTileVQWire
    decoded = FullTileVQWire.decode(None, payload)
    before = decoded.tobytes()
    flat = decoded.view("uint32").ravel()
    for position, value in zip(positions, bits):
        flat[position] = value
    if decoded.shape != shape or decoded.tobytes() != before:
        raise ValueError("Literal decoded bits changed")
    return {"branch": "literal_decoded_bits_equal", "overlay_entries": count,
            "literal_bits": literal(payload)[1], "decoded_bits_sha256": sha(before)}


def run(source=SOURCE, output=OUTPUT):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    freeze = json.loads((output / "freeze.json").read_text())
    protocol = (ROOT / "progress_reports/archive/feasibility_protocols_20261118/122-FEBUS_COMPACT_VQ_WIRE_PROTOCOL.md").read_bytes()
    if freeze["protocol_sha256"] != sha(protocol) or (output / "protocol122.md").read_bytes() != protocol:
        raise ValueError("Protocol freeze changed")
    raw_result = (source / "results.json").read_bytes()
    historical = json.loads(raw_result)
    bundles = []
    for i, tile in enumerate(historical["tiles"]):
        data = (source / f"{i:03d}.fpb").read_bytes()
        if sha(data) != tile["sha256"] or len(data) != tile["bundle_bytes"]:
            raise ValueError("Historical endpoint digest mismatch")
        bundles.append(data)
    if len(bundles) != 136:
        raise ValueError("Expected complete 136-tile panel")
    geometry = (source / "transport.zlib").read_bytes()
    locations = geometry_rows(geometry)
    if len(locations) != len(bundles):
        raise ValueError("Geometry coverage mismatch")
    original_bytes = sum(map(len, bundles)) + len(geometry)
    if original_bytes != historical["rate"]["shared_model"]["bytes"]:
        raise ValueError("Historical byte ledger mismatch")
    encoded = pack(bundles, geometry)
    path = output / "deployment.ftc"
    with path.open("xb") as stream:
        stream.write(encoded)
    reader = Reader(path.read_bytes())
    if reader.geometry != geometry:
        raise ValueError("Geometry bytes changed")
    rows = []
    for i in reversed(range(len(bundles))):
        mode, shape, payload = reader.tile(i)
        if shape != (len(locations[i]["source_channel_indices"]), 1000):
            raise ValueError("Transport/frame geometry differs")
        proof = independent_proof(bundles[i], mode, shape, payload)
        rows.append({"tile": i, "sample_path": historical["tiles"][i]["sample_path"],
                     "original_sha256": sha(bundles[i]), "original_bytes": len(bundles[i]),
                     "stored_bytes": len(payload), "index_bytes": ENTRY.size,
                     "mode": mode, "shape": shape, **proof})
    rows.sort(key=lambda row: row["tile"])
    cells = sum(row["shape"][0] * row["shape"][1] for row in rows)
    if cells != historical["cells"]:
        raise ValueError("Cell coverage differs")
    breakdown = {}
    for branch in sorted({row["branch"] for row in rows}):
        subset = [row for row in rows if row["branch"] == branch]
        breakdown[branch] = {"tiles": len(subset),
                             "original_bytes": sum(row["original_bytes"] for row in subset),
                             "stored_bytes": sum(row["stored_bytes"] for row in subset),
                             "paid_index_bytes": len(subset) * ENTRY.size,
                             "overlay_entries": sum(row["overlay_entries"] or 0 for row in subset)}
    saved = original_bytes - len(encoded)
    result = {
        "kind": "feasibility-compact-bundle-v1", "protocol_sha256": sha(protocol),
        "source": str(source.relative_to(ROOT)), "source_results_sha256": sha(raw_result),
        "geometry_sha256": sha(geometry), "deployment_sha256": sha(encoded),
        "tiles": len(rows), "cells": cells, "original_transport_bytes": original_bytes,
        "compact_transport_bytes": len(encoded), "saved_bytes": saved,
        "saving_fraction": saved / original_bytes,
        "primary_ge_3pct": saved / original_bytes >= freeze["primary_saving_fraction"],
        "ledger": {"header": HEADER.size, "index": ENTRY.size * len(rows),
                   "geometry": len(geometry), "total_sha256": 32,
                   "tile_payloads": sum(row["stored_bytes"] for row in rows)},
        "installation_common_historical_bytes": historical["rate"]["installed_shared_bytes"],
        "installation_loaded": False, "runtime_common": "Python/NumPy/Torch/constriction and OS",
        "self_contained_transport_integrity": True, "per_tile_access_verified_reverse_order": True,
        "new_training_updates": 0, "gpu_used": False, "waveform_input_reads": 0,
        "neural_waveform_roundtrip_performed": False,
        "fak1_claim": "symbolic branch equivalence only: same exact inner frame and identity overlay",
        "range_symbol_crc": "retained intact; no registry loaded or entropy symbols redecoded",
        "model_promoted": False, "artifact_specific": True, "breakdown": breakdown, "rows": rows,
    }
    if sum(result["ledger"].values()) != len(encoded):
        raise AssertionError("Object ledger mismatch")
    with (output / "results.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("tiles", "cells", "original_transport_bytes",
                                            "compact_transport_bytes", "saved_bytes",
                                            "saving_fraction", "primary_ge_3pct", "breakdown")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
