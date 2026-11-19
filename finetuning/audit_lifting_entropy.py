"""Lossless recoding of existing coefficients: isolate coder from architecture."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compression.adaptive_coefficient_stream import pack, unpack_any
from compression.research_coefficient_stream import unpack
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.run_spatial_opportunity import select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = {Path(row["sample_path"]).stem: row for row in select(args.corpus) if row["role"] == "check"}
    inputs = sorted(p for p in args.curve.glob("candidate*_step*") if p.is_dir()
                    and not p.name.startswith("candidate3_"))
    write_json(args.output / "protocol.json", {
        "kind": "frozen_coefficients_lossless_recode_v1",
        "sources": {str(p): sha256(p) for p in [Path(__file__), Path("compression/adaptive_coefficient_stream.py")]},
        "input_curve_protocol_sha256": sha256(args.curve / "protocol.json"),
        "directories": [str(p) for p in inputs],
        "skip": "candidate3 compact selected is the same initial Haar function as candidate2",
        "signal_access": "none; existing coefficient frames only",
        "coder": "per-band empirical categorical counts explicitly transmitted/compressed; constant/zlib fallbacks; whole-frame legacy fallback",
        "interpretation": "Coefficient/descriptor/side identity required. No quantization, model, metric, selection or historical result changes.",
        "entropy_reference": "Per-frame/per-band empirical marginal entropy WITHOUT tables, setup or framing; not an information bound"})
    summaries = []
    for source in inputs:
        original = json.loads((source / "results.json").read_text())
        output = args.output / source.name
        output.mkdir()
        costs = Counter()
        methods = Counter()
        entropy_bits = 0.
        old_bytes = new_bytes = adaptive_bytes = 0
        band_bytes = Counter()
        for path in sorted(source.glob("*.frc")):
            legacy = path.read_bytes()
            arrays, descriptor, side = unpack(legacy)
            adaptive, details = pack(arrays, descriptor, side, return_details=True)
            chosen = min((legacy, adaptive), key=lambda frame: (len(frame), frame[:4]))
            recovered, recovered_descriptor, recovered_side = unpack_any(chosen)
            if descriptor != recovered_descriptor or side != recovered_side:
                raise AssertionError("Coefficient metadata/side changed")
            for index, (a, b) in enumerate(zip(arrays, recovered)):
                if a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise AssertionError("Coefficients changed")
                _, counts = np.unique(a, return_counts=True)
                entropy_bits += float(-(counts * np.log2(counts / counts.sum())).sum())
                methods[str(details["methods"][index])] += 1
                band_bytes[str(index)] += details["band_bytes"][index]
            (output / path.name).write_bytes(chosen)
            old_bytes += len(legacy)
            new_bytes += len(chosen)
            adaptive_bytes += len(adaptive)
            methods[chosen[:4].decode()] += 1
            costs[samples[path.stem]["drive_id"]] += len(chosen)
        result = {**original, "frame_bytes": new_bytes,
                  "stream_bytes": new_bytes + original["setup_bytes"],
                  "stream_bpc": 8 * (new_bytes + original["setup_bytes"]) / original["cells"],
                  "legacy_frame_bytes": old_bytes, "adaptive_frame_bytes_without_legacy_fallback": adaptive_bytes,
                  "empirical_marginal_bpc_no_framing": entropy_bits / original["cells"],
                  "methods": dict(methods), "adaptive_band_bytes": dict(band_bytes),
                  "coefficient_descriptor_side_identity": True,
                  "records": [{**record, "frame_bytes": costs[record["drive_id"]]} for record in original["records"]]}
        write_json(output / "results.json", result)
        summaries.append({k: v for k, v in result.items() if k != "records"})
        print(json.dumps({"point": source.name, "old_bpc": original["stream_bpc"],
                          "new_bpc": result["stream_bpc"], "nrmse": result["macro_families"]}), flush=True)
    write_json(args.output / "results.json", summaries)
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
