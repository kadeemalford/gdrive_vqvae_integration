"""Mature VQ reference on exactly the prospective train-file checks, no updates."""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression.research_coefficient_stream import pack
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.evaluate_lifting_curve import summarize
from finetuning.model_factory import build_model
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust
from finetuning.run_index_search import FrozenIndexStream, nearest, vectors
from finetuning.run_spatial_opportunity import load, select


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
    selected = select(args.corpus)
    fitting = [r for r in selected if r["role"] == "fit"]
    checks = [r for r in selected if r["role"] == "check"]
    scales = json.loads(args.calibration.read_text())["scales"]
    maps = {m["channel_map_id"]: m for m in
            map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    write_json(args.output / "protocol.json", {
        "kind": "mature_vq_prospective_train_checks_v1", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "script_sha256": sha256(__file__),
        "sources": {str(p): sha256(p) for p in [
            Path("finetuning/evaluate_lifting_curve.py"), Path("finetuning/prospective_metrics.py"),
            Path("finetuning/prospective_spectral.py"), Path("finetuning/run_index_search.py"),
            Path("finetuning/model_factory.py")]},
        "vendor_sources": {name: sha256(args.vendor / "vqvae" / name)
                           for name in ("train_vqvae.py", "evaluate_vqvae.py")},
        "manifest_sha256": sha256(args.corpus / "samples_manifest.csv"),
        "calibration_sha256": sha256(args.calibration),
        "fit": [r["sample_path"] for r in fitting], "check": [r["sample_path"] for r in checks],
        "parameter_updates": 0, "split": "train",
        "limits": "Mature model historically saw checks; development only, pretraining/exposure not matched",
        "coder": "historical frozen marginal range/fixed/Huffman, training-fitted tables fully charged"})

    def analyze(raw):
        packet = encode_robust(raw)
        tensor = torch.from_numpy(packet.values)[None, None].cuda()
        h, w = raw.shape
        z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
        return packet, nearest(z, model.quantizer.embedding.weight), tuple(z.shape[-2:])

    counts = np.zeros(2048, np.int64)
    for row in fitting:
        _, indices, _ = analyze(load(args.corpus, row))
        counts += np.bincount(indices.cpu().numpy(), minlength=2048)
    stream = FrozenIndexStream(counts, sha256(args.checkpoint))
    (args.output / "huffman_table.json").write_bytes(stream.huffman.serialized)
    (args.output / "range_descriptor.json").write_bytes(stream.codec.serialized)
    (args.output / "range_probabilities.f32").write_bytes(stream.probabilities.tobytes())
    setup = [{"drive_id": r["drive_id"], "dataset": r["dataset"],
              "block": int(r["block_index"]), "start": int(r["time_start"]),
              "dt_s": float(r["dt_s"]), "dx_m": float(r["dx_m"]),
              "source_channel_indices": maps[r["channel_map_id"]]["source_channel_indices"]}
             for r in checks]
    import zlib
    wire_setup = zlib.compress(json.dumps(setup, sort_keys=True, separators=(",", ":")).encode(), 6)
    (args.output / "transport_setup.zlib").write_bytes(wire_setup)
    table_bytes = sum((args.output / name).stat().st_size for name in
                      ("huffman_table.json", "range_descriptor.json", "range_probabilities.f32"))
    frames, reconstructions = [], []
    for row in checks:
        raw = load(args.corpus, row)
        bits = raw.view(np.uint32)
        if np.all(bits == bits.flat[0]):
            frame = pack([np.array([bits.flat[0]], np.uint32).view(np.int32)],
                         {"kind": "literal-float32-v1", "shape": list(raw.shape)}, b"")
            reconstructed = np.full(raw.shape, bits.flat[0], np.uint32).view(np.float32)
        else:
            packet, indices, latent = analyze(raw)
            frame = stream.encode(indices, raw.shape, latent, packet.side_stream)
            recovered, shape, recovered_latent, side = stream.decode(frame)
            np.testing.assert_array_equal(recovered, indices.cpu().numpy())
            prediction = receiver.decoder(vectors(torch.from_numpy(
                recovered.astype(np.int64)).cuda(), receiver.quantizer.embedding.weight,
                recovered_latent))[0, 0, :shape[0], :shape[1]].cpu().numpy()
            reconstructed = decode(prediction, side)
        frames.append(frame)
        reconstructions.append((raw, reconstructed))
        (args.output / (Path(row["sample_path"]).stem + ".frame")).write_bytes(frame)
    result = summarize(checks, reconstructions, frames, scales, maps, len(wire_setup) + table_bytes)
    export = io.BytesIO()
    torch.save({"config": checkpoint["config"], "model_state_dict": model.state_dict()}, export)
    # This is a complete waveform model export, not a minimal decoder-only export.
    (args.output / "shared_full_model.pt").write_bytes(export.getvalue())
    result.update(shared_full_model_bytes=len(export.getvalue()), tables_bytes=table_bytes,
                  no_parameter_updates=True, historically_exposed_checks=True)
    write_json(args.output / "results.json", result)
    (args.output / "COMPLETE").touch()
    print(json.dumps({k: result[k] for k in ("macro_families", "stream_bpc", "shared_full_model_bytes")}),
          flush=True)


if __name__ == "__main__":
    main()
