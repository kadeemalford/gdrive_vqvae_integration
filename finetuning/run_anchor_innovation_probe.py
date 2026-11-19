"""H2 frozen neural anchors preflight: independent versus closed-loop innovation."""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from compression.anchor_innovation_stream import AnchorInnovationCodec, predict
from compression.lifting_stream import receiver_bytes
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.evaluate_lifting_curve import summarize
from finetuning.fit_lifting_allocation import transport
from finetuning.learned_lifting import LiftingCodec
from finetuning.model_factory import build_model
from finetuning.run_index_search import FrozenIndexStream
from finetuning.run_spatial_opportunity import load, select

STEPS = [.25, .5, 1., 2., 4.]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor.resolve()))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = {**checkpoint["config"], "exact_eval_quantization": True}
    model = build_model(config, checkpoint, "cuda").eval()
    receiver = build_model(config, checkpoint, "cuda").eval()
    residual = LiftingCodec(learned=False).cuda().eval()
    identifier = sha256(args.checkpoint)
    codec = AnchorInnovationCodec(model, receiver, None, residual, identifier)
    selected = select(args.corpus)
    fitting = [r for r in selected if r["role"] == "fit"]
    checks = [r for r in selected if r["role"] == "check"]
    scales = json.loads(args.calibration.read_text())["scales"]
    maps = {m["channel_map_id"]: m for m in map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    write_json(args.output / "protocol.json", {
        "kind": "frozen_vq_anchor_closed_loop_innovation_v1", "parameter_updates": 0,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": identifier, "steps": STEPS,
        "fit": [r["sample_path"] for r in fitting], "check": [r["sample_path"] for r in checks],
        "predictor": "per training group two-coefficient ridge on decoded anchors, no intercept; clip [-2,2]; coefficients in each frame",
        "control": "same anchors, same innovation transform/quantizer/coder, predictor identically zero",
        "normalization": "one global full-window robust FNE1, not separate hidden anchor/residual scales",
        "geometry": "even acquired channels as anchors; their effective dx doubles; no waveform resampling; frozen model was not trained for this altered spacing",
        "calibration_sha256": sha256(args.calibration),
        "manifest_sha256": sha256(args.corpus / "samples_manifest.csv"),
        "sources": {str(p): sha256(p) for p in [Path(__file__),
            Path("compression/anchor_innovation_stream.py"), Path("compression/adaptive_coefficient_stream.py")]},
        "limits": "Reused training configurations; largest tiles; not full joint neural H2 training or independent validation"})
    counts = np.zeros(2048, np.int64)
    moments = defaultdict(lambda: [np.zeros((2, 2)), np.zeros(2)])
    for row in fitting:
        raw = load(args.corpus, row)
        packet, indices, _, anchors = codec.analyze(raw)
        counts += np.bincount(indices.cpu().numpy(), minlength=2048)
        n = len(raw) // 2
        if n:
            left = anchors[:n].astype(np.float64)
            right = anchors[np.minimum(np.arange(n) + 1, len(anchors) - 1)].astype(np.float64)
            target = packet.values[1::2].astype(np.float64)
            valid = packet.valid[1::2]
            features = np.stack([left[valid], right[valid]], axis=1)
            matrix, rhs = moments[row["group_id"]]
            matrix += features.T @ features
            rhs += features.T @ target[valid]
    coefficients = {}
    for group, (matrix, rhs) in moments.items():
        ridge = max(float(np.trace(matrix)) * 1e-4, 1e-8)
        coefficients[group] = np.clip(np.linalg.solve(matrix + ridge * np.eye(2), rhs), -2, 2).tolist()
    write_json(args.output / "coefficients.json", coefficients)
    stream = FrozenIndexStream(counts, identifier)
    codec.stream = stream
    (args.output / "huffman_table.json").write_bytes(stream.huffman.serialized)
    (args.output / "range_descriptor.json").write_bytes(stream.codec.serialized)
    (args.output / "range_probabilities.f32").write_bytes(stream.probabilities.tobytes())
    setup = transport(checks, maps)
    (args.output / "transport_setup.zlib").write_bytes(setup)
    table_bytes = sum((args.output / name).stat().st_size for name in
                      ("huffman_table.json", "range_descriptor.json", "range_probabilities.f32"))
    export = io.BytesIO()
    torch.save({"config": checkpoint["config"], "model_state_dict": model.state_dict()}, export)
    (args.output / "shared_full_model.pt").write_bytes(export.getvalue())
    residual_export = receiver_bytes(residual)
    (args.output / "residual_receiver.pt").write_bytes(residual_export)
    (args.output / "FIT_COMPLETE").touch()
    # Decoded anchor context is cached, never replaced by original anchors.
    contexts = []
    optimistic_reconstructions = []
    for row in checks:
        raw = load(args.corpus, row)
        context = codec.context(raw)
        packet, _, anchors = context
        values = packet.values.copy()
        values[::2] = anchors
        from finetuning.reversible_normalization import decode
        optimistic_reconstructions.append((raw, decode(values, packet.side_stream)))
        contexts.append((raw, context))
    # Zero error on nonanchors is a diagnostic ceiling, not a transmitted codec.
    anchor_floor = summarize(checks, optimistic_reconstructions, [b""] * len(checks),
                             scales, maps, 0)
    write_json(args.output / "anchor_error_floor.json", {
        "macro_families": anchor_floor["macro_families"], "groups": anchor_floor["groups"],
        "meaning": "Decoded-anchor error with perfect original nonanchors; not a bitrate/fidelity achievement"})
    results = []
    for method in ("independent", "closed_loop"):
        for step in STEPS:
            directory = args.output / f"{method}_step{step:g}"
            directory.mkdir()
            frames, reconstructions = [], []
            for row, (raw, context) in zip(checks, contexts):
                coefficient = coefficients.get(row["group_id"], [0., 0.]) if method == "closed_loop" else [0., 0.]
                frame = codec.encode_context(raw, context, coefficient, step)
                restored = codec.decode(frame)
                frames.append(frame)
                reconstructions.append((raw, restored))
                (directory / (Path(row["sample_path"]).stem + ".fai")).write_bytes(frame)
            result = summarize(checks, reconstructions, frames, scales, maps, len(setup) + table_bytes)
            result.update(label=f"{args.output.name}/{method}/step{step:g}", method=method, step=step,
                          tables_bytes=table_bytes, shared_full_model_bytes=len(export.getvalue()),
                          shared_bytes=len(export.getvalue()) + len(residual_export),
                          neural_parameter_updates=0)
            write_json(directory / "results.json", result)
            results.append({k: v for k, v in result.items() if k != "records"})
            print(json.dumps({k: result[k] for k in ("label", "stream_bpc", "macro_families")}), flush=True)
    write_json(args.output / "results.json", results)
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
