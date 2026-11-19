"""Bounded paid-index substitutions, no continuous optimizer or model updates."""
import argparse
import ast
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression.full_tile_vq_stream import FullTileVQWire
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.fit_lifting_allocation import fitting_rows, transport
from finetuning.model_factory import build_model
from finetuning.prospective_metrics import PhysicalAccumulator
from finetuning.reversible_normalization import decode, parse_side_stream
from finetuning.robust_escape import encode_robust
from finetuning.run_dense_product_study import CORPUS, SCALES, SOURCE, load
from finetuning.run_index_search import nearest, vectors
from finetuning.run_spatial_opportunity import select
from finetuning.streaming_physical_summary import StreamingPhysicalSummary

FRACTIONS = (.01, .05, .1, .2)


def algorithm_signature(text):
    tree = ast.parse(text)
    nodes = [node for node in tree.body if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("physical_score", "proposals", "probe")
    ) or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FRACTIONS"
                                             for target in node.targets))]
    if len(nodes) != 4:
        raise ValueError("Incomplete algorithm definition")
    return hashlib.sha256("".join(ast.dump(node, include_attributes=False) for node in nodes).encode()).hexdigest()


def physical_score(raw, reconstructed, scale):
    scores = []
    for target, prediction in zip(raw, reconstructed):
        accumulator = PhysicalAccumulator()
        accumulator.update(target, prediction)
        scores.append(accumulator.result(scale)["nrmse_star"])
    return float(np.mean(scores))


def proposals(current, alternatives, costs, ranking, negative_only):
    valid = (alternatives != current[:, None]) & (costs[alternatives] <= costs[current, None])
    if negative_only:
        valid &= ranking < 0
    ranked = ranking.masked_fill(~valid, float("inf"))
    score, choice = ranked.min(1)
    eligible = torch.where(torch.isfinite(score))[0]
    order = eligible[torch.argsort(score[eligible], stable=True)]
    selected = alternatives.gather(1, choice[:, None])[:, 0]
    for fraction in FRACTIONS:
        proposal = current.clone()
        count = min(len(order), max(1, math.ceil(fraction * len(current))))
        proposal[order[:count]] = selected[order[:count]]
        yield fraction, proposal


def probe(model, receiver, sender, raw, reference_scale, method):
    baseline_frame = sender.encode(raw)
    restored = receiver.decode(baseline_frame)
    if baseline_frame[:4] != b"FAK1":
        return baseline_frame, baseline_frame, restored, restored, {"literal": True, "trials": []}
    packet = encode_robust(raw)
    device = next(model.parameters()).device
    h, w = raw.shape
    with torch.no_grad():
        z = model.encoder(F.pad(torch.from_numpy(packet.values)[None, None].to(device),
                                (0, (-w) % 16, 0, (-h) % 16)))
        weight = model.quantizer.embedding.weight.detach()
        flat = z.permute(0, 2, 3, 1).reshape(-1, weight.shape[1])
        distance = flat.square().sum(1, keepdim=True) - 2 * flat @ weight.T + weight.square().sum(1)
        alternatives = distance.topk(8, largest=False, sorted=True).indices
        current = distance.argmin(1)
        paid_ids, _, latent, side = sender.stream.decode(baseline_frame)
        np.testing.assert_array_equal(paid_ids, current.cpu().numpy())
        costs = -torch.log2(torch.from_numpy(sender.stream.probabilities).to(device))
    original = restored.copy()
    best_score = physical_score(raw, restored, reference_scale)
    best_frame = baseline_frame
    trials = []
    valid = torch.from_numpy(packet.valid)[None, None].to(device)
    target = torch.from_numpy(packet.values)[None, None].to(device)
    _, _, physical_scale, _, _ = parse_side_stream(packet.side_stream)
    denominator = np.maximum(raw.astype(np.float64).std(axis=1), reference_scale) / physical_scale
    denominator = torch.from_numpy(denominator)[None, None, :, None].to(device)
    for iteration in range(2):
        if method == "gradient":
            q = vectors(current, weight, latent).detach().requires_grad_(True)
            prediction = model.decoder(q)[..., :h, :w]
            loss = ((((prediction.double() - target.double()) / denominator) ** 2) * valid).mean()
            gradient, = torch.autograd.grad(loss, q)
            if not torch.isfinite(gradient).all():
                raise ValueError("Nonfinite decoder gradient")
            gradient = gradient.permute(0, 2, 3, 1).reshape(-1, weight.shape[1]).detach()
            with torch.no_grad():
                ranking = ((F.embedding(alternatives, weight) - F.embedding(current, weight)[:, None])
                           * gradient[:, None]).sum(-1)
            del q, prediction, loss, gradient
        else:
            ranking = distance.gather(1, alternatives) - distance.gather(1, current[:, None])
        accepted_ids = current
        round_score = best_score
        for fraction, ids in proposals(current, alternatives, costs, ranking, method == "gradient"):
            record = {"round": iteration, "fraction": fraction,
                      "changed_positions": int((ids != current).sum())}
            frame = sender.stream.encode(ids.cpu().numpy(), raw.shape, latent, side)
            record["frame_bytes"] = len(frame)
            if len(frame) > len(baseline_frame):
                record["status"] = "rejected_bytes"
            else:
                try:
                    with torch.no_grad():
                        normalized = model.decoder(vectors(ids, weight, latent))[0, 0, :h, :w].cpu().numpy()
                    candidate = decode(normalized, side)
                    score = physical_score(raw, candidate, reference_scale)
                except ValueError as error:
                    record.update(status="rejected_invalid", reason=str(error))
                    trials.append(record)
                    continue
                record["physical_window_nrmse_star"] = score
                if score < round_score:
                    round_score, accepted_ids = score, ids.clone()
                    best_frame, restored = frame, candidate.copy()
                    record["status"] = "accepted_incumbent"
                else:
                    record["status"] = "rejected_error"
            trials.append(record)
        current, best_score = accepted_ids, round_score
    replay = receiver.decode(best_frame)
    if replay.tobytes() != restored.tobytes() or len(best_frame) > len(baseline_frame):
        raise AssertionError("Paid discrete replay or hard frame cap violated")
    return baseline_frame, best_frame, original, replay, {"literal": False, "trials": trials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("gradient", "latent"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", choices=("pilot", "expanded"), default="pilot")
    args = parser.parse_args()
    sys.path.insert(0, str(Path("vendor/smartgrids").resolve()))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = select(CORPUS)
    fitting = fitting_rows(rows)
    files = defaultdict(list)
    for row in rows:
        if row["role"] == "check":
            files[row["group_id"]].append(row)
    checks = []
    for group, candidates in sorted(files.items()):
        file_id = min(row["drive_id"] for row in candidates)
        candidates = sorted([row for row in candidates if row["drive_id"] == file_id],
                            key=lambda row: row["sample_path"])
        checks.append(candidates[len(candidates) // 2])
    if args.population == "expanded":
        pilot = Path("results/experiments/direct_discrete_index_probe_20261115")
        screening = json.loads((pilot / "gradient/results.json").read_text())
        prior = json.loads((pilot / "gradient/protocol.json").read_text())
        snapshot = pilot / "SOURCE_SNAPSHOT/finetuning/direct_discrete_index_probe.py"
        if (args.method != "gradient" or not screening["expansion_eligible"]
                or prior["source_sha256"] != sha256(SOURCE)
                or algorithm_signature(snapshot.read_text()) != algorithm_signature(Path(__file__).read_text())):
            raise ValueError("Expansion must preserve the eligible frozen pilot algorithm/source")
        from finetuning.full_tile_bottleneck import select_probes
        _, expanded = select_probes(CORPUS)
        checks = [row for row in expanded if row["role"] == "check"]
    expected_windows = 136 if args.population == "expanded" else 10
    if len(checks) != expected_windows or any(row["split"] != "train" for row in fitting + checks):
        raise ValueError("Incomplete authorized training population")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(SOURCE)
    write_json(args.output / "protocol.json", {
        "kind": "direct-discrete-index-probe-v1", "method": args.method, "population": args.population,
        "source_sha256": source_hash, "algorithm_signature": algorithm_signature(Path(__file__).read_text()),
        "physical_scales_sha256": sha256(SCALES),
        "checks": checks, "fitting": [row["sample_path"] for row in fitting],
        "sources": {str(path): sha256(path) for path in [Path(__file__), Path("compression/full_tile_vq_stream.py"),
                    Path("compression/alphabet_stream.py"), Path("finetuning/run_index_search.py"),
                    Path("finetuning/streaming_physical_summary.py"), Path("finetuning/prospective_metrics.py")]},
        "input_sha256": {row["sample_path"]: sha256(CORPUS / "train" / Path(row["sample_path"]).name)
                         for row in fitting + checks},
        "rounds": 2, "fractions": FRACTIONS, "shortlist": 8, "model_parameter_updates": 0,
        "decoder_gradient_backwards_maximum": 2 * len(checks) if args.method == "gradient" else 0,
        "hard_candidate_maximum": 8 * len(checks), "reserved_signal_access": False,
        "limits": "Consumed train checks, not novel architectural family, no equal-FLOP claim"})
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    config = {**saved["config"], "exact_eval_quantization": True}
    model = build_model(config, saved, "cuda").eval().requires_grad_(False)
    receiver_model = build_model(config, saved, "cuda").eval().requires_grad_(False)
    counts = np.zeros(2048, np.int64)
    with torch.no_grad():
        for row in fitting:
            raw = load(row)
            if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
                continue
            packet = encode_robust(raw)
            x = torch.from_numpy(packet.values)[None, None].cuda()
            h, w = packet.values.shape
            z = model.encoder(F.pad(x, (0, (-w) % 16, 0, (-h) % 16)))
            counts += np.bincount(nearest(z, model.quantizer.embedding.weight).cpu().numpy(), minlength=2048)
    sender = FullTileVQWire(model, counts, source_hash)
    registry = sender.setup_bytes()
    if args.population == "expanded" and registry != (pilot / "gradient/registry.zlib").read_bytes():
        raise ValueError("Frozen pilot probability registry changed")
    receiver = FullTileVQWire.from_setup(receiver_model, registry)
    (args.output / "registry.zlib").write_bytes(registry)
    geometry = transport(checks, {row["channel_map_id"]: row for row in
                                 map(json.loads, (CORPUS / "channel_maps.jsonl").open())})
    (args.output / "transport.zlib").write_bytes(geometry)
    scales = json.loads(SCALES.read_text())["scales"]
    maps = {row["channel_map_id"]: row for row in map(json.loads, (CORPUS / "channel_maps.jsonl").open())}
    summaries = {label: StreamingPhysicalSummary(scales, maps) for label in ("baseline", "searched")}
    for label in summaries:
        (args.output / label).mkdir()
    diagnostics = []
    reproduced_pilot_frames = 0
    for row in checks:
        raw = load(row)
        original_frame, frame, original, reconstructed, diagnostic = probe(
            model, receiver, sender, raw, scales[row["group_id"]], args.method)
        if args.population == "expanded":
            previous = pilot / "gradient/baseline" / (Path(row["sample_path"]).stem + ".frame")
            if previous.exists():
                if previous.read_bytes() != original_frame:
                    raise AssertionError("Expanded baseline differs from original paid pilot frame")
                reproduced_pilot_frames += 1
        for label, decoded, packet in (("baseline", original, original_frame), ("searched", reconstructed, frame)):
            summaries[label].update(row, raw, decoded, len(packet))
            (args.output / label / (Path(row["sample_path"]).stem + ".frame")).write_bytes(packet)
        diagnostics.append({"sample_path": row["sample_path"], **diagnostic})
        print(json.dumps({"method": args.method, "file": row["drive_id"],
                          "bytes_before": len(original_frame), "bytes_after": len(frame)}), flush=True)
    results = {label: summary.result(len(registry) + len(geometry)) for label, summary in summaries.items()}
    for label, result in results.items():
        write_json(args.output / label / "results.json", result)
    improvement = 1 - results["searched"]["macro_families"] / results["baseline"]["macro_families"]
    if args.population == "expanded" and reproduced_pilot_frames != 10:
        raise AssertionError("Expanded population must contain all original pilot windows")
    gate = (improvement >= .01 and results["searched"]["stream_bytes"] <= results["baseline"]["stream_bytes"]
            and all(results["searched"]["groups"][group] <= value for group, value in results["baseline"]["groups"].items()))
    if sha256(SOURCE) != source_hash:
        raise AssertionError("Frozen neural source changed")
    write_json(args.output / "results.json", {
        "method": args.method, "relative_error_reduction": improvement,
        "relative_byte_saving": 1 - results["searched"]["stream_bytes"] / results["baseline"]["stream_bytes"],
        "expansion_eligible": gate, "parameter_updates": 0, "full_model_bytes": SOURCE.stat().st_size,
        "reproduced_pilot_baseline_frames": reproduced_pilot_frames,
        "diagnostics": diagnostics, "trial_status_counts": dict(Counter(
            trial["status"] for row in diagnostics for trial in row["trials"]))})
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
