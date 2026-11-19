"""Fixed-coefficient integer prediction probe; all modes/table costs transmitted."""
import argparse
import json
from collections import Counter
from pathlib import Path

from compression.adaptive_coefficient_stream import unpack_any
from compression.predictive_coefficient_stream import pack_predictive, unpack_predictive
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.run_spatial_opportunity import select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = {Path(row["sample_path"]).stem: row for row in select(args.corpus) if row["role"] == "check"}
    points = sorted(p for p in args.source.glob("candidate*_step*") if p.is_dir())
    write_json(args.output / "protocol.json", {
        "kind": "lossless_integer_coefficient_prediction_v1", "parameter_updates": 0,
        "source_protocol_sha256": sha256(args.source / "protocol.json"),
        "modes": ["identity", "time_difference", "space_difference", "lorenzo_2d"],
        "sources": {str(p): sha256(p) for p in
                    [Path(__file__), Path("compression/predictive_coefficient_stream.py")]},
        "scope": "Same coefficients, no waveform reads; per-band mode/header and empirical tables charged",
        "limits": "Universal online mode selection; not entropy bound or neural training"})
    summaries = []
    for point in points:
        original = json.loads((point / "results.json").read_text())
        output = args.output / point.name
        output.mkdir()
        size = 0
        modes, costs = Counter(), Counter()
        for path in sorted(point.glob("*.frc")):
            old = path.read_bytes()
            arrays, descriptor, side = unpack_any(old)
            new = pack_predictive(arrays, descriptor, side)
            recovered, recovered_descriptor, recovered_side = unpack_predictive(new)
            if descriptor != recovered_descriptor or side != recovered_side or len(arrays) != len(recovered):
                raise AssertionError("Predictive coefficient metadata changed")
            for a, b in zip(arrays, recovered):
                if a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise AssertionError("Predictive coefficient values changed")
            if len(new) > len(old):
                raise AssertionError("Complete fallback unexpectedly increased bytes")
            (output / path.name).write_bytes(new)
            size += len(new)
            modes[new[:4].decode()] += 1
            costs[samples[path.stem]["drive_id"]] += len(new)
        result = {**original, "previous_stream_bpc": original["stream_bpc"],
                  "frame_bytes": size, "stream_bytes": size + original["setup_bytes"],
                  "stream_bpc": 8 * (size + original["setup_bytes"]) / original["cells"],
                  "predictive_methods": dict(modes), "coefficient_descriptor_side_identity": True,
                  "records": [{**r, "frame_bytes": costs[r["drive_id"]]} for r in original["records"]]}
        write_json(output / "results.json", result)
        summaries.append({k: v for k, v in result.items() if k != "records"})
        print(json.dumps({"point": point.name, "previous_bpc": original["stream_bpc"],
                          "new_bpc": result["stream_bpc"], "nrmse": result["macro_families"]}), flush=True)
    write_json(args.output / "results.json", summaries)
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
