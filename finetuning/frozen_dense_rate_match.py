"""Frozen full-tile index RD curves with charged FAK1 and training-only scales."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression.full_tile_vq_stream import FullTileVQWire
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.dense_product_vq import build
from finetuning.fit_lifting_allocation import transport
from finetuning.full_tile_bottleneck import select_probes
from finetuning.robust_escape import encode_robust
from finetuning.run_dense_product_study import CORPUS, SCALES, load
from finetuning.streaming_physical_summary import StreamingPhysicalSummary

MULTIPLIERS = [0., .03, .1, .3, 1., 3.]


def distances(z, weight):
    flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    return flat.square().sum(1, keepdim=True) - 2 * flat @ weight.T + weight.square().sum(1)


def choose_indices(squared_distances, bits, penalty, dimension=128):
    if not np.isfinite(penalty) or penalty < 0 or dimension < 1:
        raise ValueError("Finite nonnegative rate penalty required")
    return (squared_distances / dimension + penalty * bits).argmin(1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("flat", "product"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path("vendor/smartgrids").resolve()))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    study = Path("results/experiments/dense_full_tile_product_v2_fold0_20261115")
    source = study / "arms" / args.variant / "best.pt"
    selected = json.loads((source.parent / "selection.json").read_text())["update"]
    if selected != {"flat": 0, "product": 4096}[args.variant] or not (study / "COMPLETE").exists():
        raise ValueError("Expected completed pilot and previously frozen selections")
    fitting, probes = select_probes(CORPUS)
    checks = [row for row in probes if row["role"] == "check"]
    if any(row["split"] != "train" for row in fitting + checks):
        raise ValueError("Frozen rate curve is train-only")
    if {row["drive_id"] for row in fitting} & {row["drive_id"] for row in checks}:
        raise ValueError("Rate calibration/check files overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    identifier = sha256(source)
    write_json(args.output / "protocol.json", {
        "kind": "frozen-dense-rate-match-v1", "variant": args.variant,
        "source": str(source), "source_sha256": identifier, "selected_update": selected,
        "parameter_updates": 0, "multipliers": MULTIPLIERS,
        "fitting": [row["sample_path"] for row in fitting], "checks": [row["sample_path"] for row in checks],
        "input_sha256": {row["sample_path"]: sha256(CORPUS / "train" / Path(row["sample_path"]).name)
                         for row in fitting + checks},
        "sources": {str(p): sha256(p) for p in [Path(__file__), Path("finetuning/dense_product_vq.py"),
                   Path("finetuning/factorized_vq.py"), Path("compression/full_tile_vq_stream.py"),
                   Path("compression/alphabet_stream.py"), Path("finetuning/streaming_physical_summary.py")]},
        "scales_sha256": sha256(SCALES), "manifest_sha256": sha256(CORPUS / "samples_manifest.csv"),
        "limits": "Frozen index decisions only; train checks; no interpolation, checkpoint reselection or final acceptance"})
    saved = torch.load(source, map_location="cpu", weights_only=True)
    model = build(saved["config"], saved["model_state_dict"], "cuda").requires_grad_(False)
    receiver_model = build(saved["config"], saved["model_state_dict"], "cuda").requires_grad_(False)
    weight = model.quantizer.embedding.weight
    counts = np.zeros(2048, np.int64)
    error_sum, vector_count = 0., 0
    for row in fitting:
        raw = load(row)
        if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
            continue
        packet = encode_robust(raw)
        h, w = raw.shape
        z = model.encoder(F.pad(torch.from_numpy(packet.values)[None, None].cuda(),
                                (0, (-w) % 16, 0, (-h) % 16)))
        distance = distances(z, weight)
        ids = distance.argmin(1)
        flat = z.permute(0, 2, 3, 1).reshape(-1, 128)
        q = F.embedding(ids, weight)
        error_sum += float((flat - q).double().square().sum())
        vector_count += len(ids)
        counts += np.bincount(ids.cpu().numpy(), minlength=2048)
    sender = FullTileVQWire(model, counts, identifier)
    registry = sender.setup_bytes()
    receiver = FullTileVQWire.from_setup(receiver_model, registry)
    (args.output / "registry.zlib").write_bytes(registry)
    probabilities = sender.stream.probabilities.astype(np.float64)
    mean_information = float((counts * -np.log2(probabilities)).sum() / vector_count)
    latent_mse = error_sum / (vector_count * 128)
    scale = latent_mse / mean_information
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate fixed rate calibration")
    write_json(args.output / "calibration.json", {
        "latent_mse": latent_mse, "mean_information_bits": mean_information,
        "scale": scale, "vectors": vector_count, "lambda": [scale * factor for factor in MULTIPLIERS],
        "parameter_updates": 0, "fitted_only_on_training_fit": True})
    (args.output / "FIT_COMPLETE").touch()
    scales = json.loads(SCALES.read_text())["scales"]
    maps = {row["channel_map_id"]: row for row in map(json.loads, (CORPUS / "channel_maps.jsonl").open())}
    geometry = transport(checks, maps)
    (args.output / "transport.zlib").write_bytes(geometry)
    summaries = [StreamingPhysicalSummary(scales, maps) for _ in MULTIPLIERS]
    folders = [args.output / f"multiplier_{factor:g}" for factor in MULTIPLIERS]
    for folder in folders:
        folder.mkdir()
    bits = -torch.log2(torch.from_numpy(sender.stream.probabilities).cuda())
    for row in checks:
        raw = load(row)
        h, w = raw.shape
        constant = bool(np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]))
        if not constant:
            packet = encode_robust(raw)
            z = model.encoder(F.pad(torch.from_numpy(packet.values)[None, None].cuda(),
                                    (0, (-w) % 16, 0, (-h) % 16)))
            distance = distances(z, weight)
            latent = tuple(z.shape[-2:])
        baseline_frame = sender.encode(raw)
        for factor, summary, folder in zip(MULTIPLIERS, summaries, folders):
            if constant:
                frame = baseline_frame
            else:
                ids = choose_indices(distance, bits, scale * factor)
                frame = sender.stream.encode(ids.cpu().numpy(), raw.shape, latent, packet.side_stream)
                if factor == 0 and frame != baseline_frame:
                    raise AssertionError("Zero rate penalty changed paid baseline indices")
            reconstructed = receiver.decode(frame)
            summary.update(row, raw, reconstructed, len(frame))
            (folder / (Path(row["sample_path"]).stem + ".frame")).write_bytes(frame)
    results = []
    for factor, summary, folder in zip(MULTIPLIERS, summaries, folders):
        result = summary.result(len(registry) + len(geometry))
        result.update(multiplier=factor, rate_lambda=scale * factor,
                      parameter_updates=0, shared_full_model_bytes=source.stat().st_size)
        if result["windows"] != 136 or result["cells"] != 12799000:
            raise AssertionError("Frozen rate curve lost acquired coverage")
        write_json(folder / "results.json", result)
        results.append({key: value for key, value in result.items() if key != "records"})
        print(json.dumps({key: result[key] for key in ("multiplier", "macro_families", "stream_bpc")}), flush=True)
    if sha256(source) != identifier:
        raise AssertionError("Frozen rate source changed")
    write_json(args.output / "results.json", results)
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
