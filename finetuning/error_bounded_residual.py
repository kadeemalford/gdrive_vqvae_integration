"""Train-only H4 preflight: charge neural base plus bounded physical correction."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from compression.bounded_residual_stream import audit, decode, encode
from compression.full_tile_vq_stream import FullTileVQWire
from compression.predictive_coefficient_stream import unpack_predictive
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.compare_physical_probe import compare_records
from finetuning.model_factory import build_model
from finetuning.run_dense_product_study import CORPUS, SCALES, SOURCE, load
from finetuning.streaming_physical_summary import StreamingPhysicalSummary

ALPHAS = (.1, .03, .01)
PRIOR = Path("results/experiments/direct_discrete_expanded_20261115/gradient")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path("vendor/smartgrids").resolve()))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    prior = json.loads((PRIOR / "protocol.json").read_text())
    rows = prior["checks"]
    if (len(rows) != 136 or len({r["drive_id"] for r in rows}) != 43
            or any(r["split"] != "train" for r in rows)
            or sha256(SOURCE) != prior["source_sha256"]
            or sha256(SCALES) != prior["physical_scales_sha256"]
            or not (PRIOR / "COMPLETE").exists()):
        raise ValueError("Frozen train-only baseline preflight failed")
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve().relative_to(Path.cwd().resolve()), Path("compression/bounded_residual_stream.py"),
             Path("compression/predictive_coefficient_stream.py"),
             Path("compression/adaptive_coefficient_stream.py"),
             Path("compression/research_coefficient_stream.py"),
             Path("compression/full_tile_vq_stream.py"),
             Path("finetuning/streaming_physical_summary.py")]
    hashes = {str(p): sha256(p) for p in files}
    write_json(args.output / "protocol.json", {
        "kind": "absolute-bounded-residual-preflight-v1", "alphas": ALPHAS,
        "checks": rows, "source_sha256": sha256(SOURCE), "sources": hashes,
        "prior_protocol_sha256": sha256(PRIOR / "protocol.json"),
        "scale_sha256": sha256(SCALES), "parameter_updates": 0,
        "operational_tolerances_validated": False, "reserved_waveform_access": False})
    for p in files:
        target = args.output / "SOURCE_SNAPSHOT" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    config = {**saved["config"], "exact_eval_quantization": True}
    model = build_model(config, saved, "cuda").eval().requires_grad_(False)
    registry = (PRIOR / "registry.zlib").read_bytes()
    receiver = FullTileVQWire.from_setup(model, registry)
    scales = json.loads(SCALES.read_text())["scales"]
    maps = {r["channel_map_id"]: r for r in map(json.loads, (CORPUS / "channel_maps.jsonl").open())}
    geometry = (PRIOR / "transport.zlib").read_bytes()
    (args.output / "transport.zlib").write_bytes(geometry)
    (args.output / "registry.zlib").write_bytes(registry)
    summaries = {(a, kind): StreamingPhysicalSummary(scales, maps)
                 for a in ALPHAS for kind in ("zero", "neural")}
    for a, kind in summaries:
        (args.output / f"alpha_{a:g}" / kind).mkdir(parents=True)
    diagnostics = []
    start = time.monotonic()
    for index, row in enumerate(rows):
        raw = load(row)
        if sha256(CORPUS / "train" / Path(row["sample_path"]).name) != prior["input_sha256"][row["sample_path"]]:
            raise ValueError("Training input changed since frozen baseline")
        name = Path(row["sample_path"]).stem
        baseline_frame = (PRIOR / "baseline" / f"{name}.frame").read_bytes()
        neural = receiver.decode(baseline_frame)
        for alpha in ALPHAS:
            epsilon = alpha * scales[row["group_id"]]
            for kind, base in (("zero", np.zeros_like(raw)), ("neural", neural)):
                frame = encode(raw, base, epsilon)
                restored = decode(frame, base.copy())
                checked = audit(raw, restored, epsilon)
                _, descriptor, _ = unpack_predictive(frame)
                charged = len(frame) + (len(baseline_frame) if kind == "neural" else 0)
                summaries[alpha, kind].update(row, raw, restored, charged)
                (args.output / f"alpha_{alpha:g}" / kind / f"{name}.frame").write_bytes(frame)
                diagnostics.append({"sample_path": row["sample_path"], "alpha": alpha, "kind": kind,
                                    "epsilon": epsilon, "escapes": descriptor["escapes"],
                                    "correction_bytes": len(frame), "charged_frame_bytes": charged, **checked})
        print(json.dumps({"completed": index + 1, "windows": len(rows),
                          "elapsed_seconds": time.monotonic() - start}), flush=True)
    points = []
    for alpha in ALPHAS:
        results = {}
        for kind in ("zero", "neural"):
            setup = len(geometry) + (len(registry) if kind == "neural" else 0)
            results[kind] = summaries[alpha, kind].result(setup)
            results[kind].update(full_shared_model_bytes=SOURCE.stat().st_size if kind == "neural" else 0)
            write_json(args.output / f"alpha_{alpha:g}" / kind / "results.json", results[kind])
        saving = 1 - results["neural"]["stream_bytes"] / results["zero"]["stream_bytes"]
        # The family means are error scores; compare bytes directly from paired records instead.
        families = {}
        for kind in ("zero", "neural"):
            totals = {}
            for record in results[kind]["records"]:
                family = record["group_id"].split("/")[0]
                totals[family] = totals.get(family, 0) + record["frame_bytes"]
            families[kind] = totals
        no_family_increase = all(families["neural"][f] <= value for f, value in families["zero"].items())
        points.append({"alpha": alpha, "relative_complete_stream_saving": saving,
                       "zero_bpc": results["zero"]["stream_bpc"],
                       "neural_bpc": results["neural"]["stream_bpc"],
                       "family_frame_bytes": families,
                       "opportunity_gate": saving >= .05 and no_family_increase})
        write_json(args.output / f"alpha_{alpha:g}" / "PHYSICAL_COMPARISON.json",
                   compare_records(results["zero"], results["neural"]))
    if any(sha256(Path(p)) != expected for p, expected in hashes.items()):
        raise AssertionError("Sources changed during execution")
    write_json(args.output / "results.json", {
        "points": points, "conditional_model_opportunity": sum(p["opportunity_gate"] for p in points) >= 2,
        "guarantee_violations": 0, "parameter_updates": 0,
        "elapsed_seconds": time.monotonic() - start, "diagnostics": diagnostics,
        "limits": "Equal absolute bounds, not equal realized distortion; experimental not operational tolerances."})
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
