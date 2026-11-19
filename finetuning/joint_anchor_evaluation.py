"""Shared hard-wire calibration and physical scoring for the joint H2 pilot."""
import hashlib
import io
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from compression.alphabet_stream import AlphabetStream
from compression.adaptive_coefficient_stream import pack_best
from compression.joint_anchor_vq_stream import JointAnchorWire
from finetuning.audit_prospective_preparation import write_json
from finetuning.evaluate_lifting_curve import summarize
from finetuning.fit_lifting_allocation import transport
from finetuning.joint_anchor_vq import build, exact_branch_forward
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust
from finetuning.run_spatial_opportunity import load


def export(model, config):
    stream = io.BytesIO()
    torch.save({"config": config, "model_state_dict": {
        key: value.detach().cpu() for key, value in model.state_dict().items()}}, stream)
    return stream.getvalue()


def load_export(data, device):
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    return build(saved["config"], saved["model_state_dict"], device)


@torch.no_grad()
def calibrate(model, rows, corpus):
    counts = np.zeros((2, 2048), np.int64)
    device = next(model.parameters()).device
    model.eval()
    for row in rows:
        raw = load(corpus, row)
        if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
            continue
        packet = encode_robust(raw)
        _, _, info = model(torch.from_numpy(packet.values)[None, None].to(device))
        counts[0] += np.bincount(info["anchor_ids"].cpu().numpy(), minlength=2048)
        if info["innovation_ids"] is not None:
            counts[1] += np.bincount(info["innovation_ids"].cpu().numpy(), minlength=2048)
    return counts


@torch.no_grad()
def _evaluate(model, config, fitting, checks, corpus, scales, maps, output, update):
    model.eval()
    directory = output / f"check_{update:06d}"
    directory.mkdir(exist_ok=False)
    counts = calibrate(model, fitting, corpus)
    data = export(model, config)
    identifier = hashlib.sha256(data).hexdigest()
    sender = JointAnchorWire(model, counts, identifier)
    receiver_model = load_export(data, next(model.parameters()).device)
    setup = sender.setup_bytes()
    receiver = JointAnchorWire.from_setup(receiver_model, setup)
    (directory / "registry.zlib").write_bytes(setup)
    geometry = transport(checks, maps)
    (directory / "transport.zlib").write_bytes(geometry)
    frames, reconstructions = [], []
    replay_max = 0.
    for row in checks:
        raw = load(corpus, row)
        frame = sender.encode(raw)
        restored = receiver.decode(frame)
        packet = encode_robust(raw)
        direct, _, _ = model(torch.from_numpy(packet.values)[None, None].to(next(model.parameters()).device))
        expected = decode(direct[0, 0].cpu().numpy(), packet.side_stream)
        bits = raw.view(np.uint32)
        if not np.all(bits == bits.flat[0]):
            np.testing.assert_array_equal(expected.view(np.uint32), restored.view(np.uint32))
            replay_max = max(replay_max, float(np.abs(expected.astype(np.float64) - restored).max()))
        else:
            if restored.tobytes() != raw.tobytes():
                raise AssertionError("Literal bits changed")
        frames.append(frame)
        reconstructions.append((raw, restored))
        (directory / (Path(row["sample_path"]).stem + ".fjh")).write_bytes(frame)
    result = summarize(checks, reconstructions, frames, scales, maps, len(setup) + len(geometry))
    result.update(update=update, receiver_full_bytes=len(data), model_sha256=identifier,
                  registry_bytes=len(setup), independent_receiver_max_difference=replay_max,
                  shared_weights_excluded_from_stream=True)
    write_json(directory / "results.json", result)
    return result, data


def evaluate(model, *args, **kwargs):
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Constructing an independent receiver must not perturb future training RNG.
    with torch.random.fork_rng(devices=devices):
        return _evaluate(model, *args, **kwargs)


@torch.no_grad()
def practical_reference(model, identifier, fitting, checks, corpus, scales, maps, output):
    """Mature full-tile VQ under the same compact FAK1 registry, not old table overhead."""
    output.mkdir(exist_ok=False)
    device = next(model.parameters()).device
    counts = np.zeros(2048, np.int64)
    model.eval()
    for row in fitting:
        raw = load(corpus, row)
        if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
            continue
        packet = encode_robust(raw)
        _, _, ids, _, _ = exact_branch_forward(model, torch.from_numpy(packet.values)[None, None].to(device))
        counts += np.bincount(ids.cpu().numpy(), minlength=2048)
    registry = zlib.compress(json.dumps({"identifier": identifier, "kind": "FAK1", "K": 2048},
                                      sort_keys=True, separators=(",", ":")).encode() +
                             b"\n" + counts.astype("<i8").tobytes(), 6)
    (output / "registry.zlib").write_bytes(registry)
    stream = AlphabetStream(counts, identifier)
    frames, reconstructions = [], []
    for row in checks:
        raw = load(corpus, row)
        bits = raw.view(np.uint32)
        if np.all(bits == bits.flat[0]):
            frame = pack_best([np.array([bits.flat[0]], np.uint32).view(np.int32)],
                              {"kind": "literal-float32-v1", "shape": list(raw.shape)}, b"")
            frames.append(frame)
            reconstructions.append((raw, raw.copy()))
            (output / (Path(row["sample_path"]).stem + ".fak")).write_bytes(frame)
            continue
        packet = encode_robust(raw)
        _, _, ids, latent, _ = exact_branch_forward(model, torch.from_numpy(packet.values)[None, None].to(device))
        frame = stream.encode(ids.cpu().numpy(), raw.shape, latent, packet.side_stream)
        recovered, shape, recovered_latent, side = stream.decode(frame)
        np.testing.assert_array_equal(ids.cpu().numpy(), recovered)
        weight = model.quantizer.embedding.weight
        q = torch.nn.functional.embedding(
            torch.from_numpy(recovered.astype(np.int64)).to(device), weight).view(
                1, *recovered_latent, weight.shape[1]).permute(0, 3, 1, 2).contiguous()
        restored = decode(model.decoder(q)[0, 0, :shape[0], :shape[1]].cpu().numpy(), side)
        frames.append(frame)
        reconstructions.append((raw, restored))
        (output / (Path(row["sample_path"]).stem + ".fak")).write_bytes(frame)
    geometry = transport(checks, maps)
    (output / "transport.zlib").write_bytes(geometry)
    result = summarize(checks, reconstructions, frames, scales, maps, len(registry) + len(geometry))
    result.update(parameter_updates=0, historical_exposure=True, checkpoint_sha256=identifier,
                  coding="Same FAK1 and compressed exact counts as joint H2, one book")
    write_json(output / "results.json", result)
    return result
