"""Bounded dense flat/product training with prospective physical wire selection."""
import argparse
import csv
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from compression.full_tile_vq_stream import FullTileVQWire
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.dense_product_vq import build, export, initialize
from finetuning.fit_lifting_allocation import transport
from finetuning.full_tile_bottleneck import select_probes
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust
from finetuning.run_joint_anchor_study import atomic_save
from finetuning.streaming_physical_summary import StreamingPhysicalSummary

CORPUS = Path("data/finetune/spatial_band_v1_fold0_20261024")
SCALES = Path("results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json")
SOURCE = Path("results/remote_setup/dense_transfer_fold0_weights_p93.pt")


def score(result):
    return result["macro_families"] ** 2 + .1 * result["stream_bpc"] / .7


def should_evaluate(update, endpoint, smoke):
    return smoke or update % 2048 == 0 or update == endpoint


@torch.no_grad()
def _evaluate(model, config, fitting, rows, scales, maps, output, update):
    model.eval()
    folder = output / f"check_{update:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    counts = np.zeros(2048, np.int64)
    for row in fitting:
        raw = load(row)
        if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
            continue
        packet = encode_robust(raw)
        _, _, ids, _, _ = exact_branch_forward(model, torch.from_numpy(packet.values)[None, None].cuda())
        counts += np.bincount(ids.cpu().numpy(), minlength=2048)
    data = export(model, config)
    import hashlib
    identifier = hashlib.sha256(data).hexdigest()
    sender = FullTileVQWire(model, counts, identifier)
    registry = sender.setup_bytes()
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    receiver_model = build(saved["config"], saved["model_state_dict"], "cuda")
    receiver = FullTileVQWire.from_setup(receiver_model, registry)
    (folder / "registry.zlib").write_bytes(registry)
    geometry = transport(rows, maps)
    (folder / "transport.zlib").write_bytes(geometry)
    summary = StreamingPhysicalSummary(scales, maps)
    for row in rows:
        raw = load(row)
        frame = sender.encode(raw)
        restored = receiver.decode(frame)
        bits = raw.view(np.uint32)
        if np.all(bits == bits.flat[0]):
            expected = raw
        else:
            packet = encode_robust(raw)
            direct, _, _, _, _ = exact_branch_forward(model, torch.from_numpy(packet.values)[None, None].cuda())
            expected = decode(direct[0, 0].cpu().numpy(), packet.side_stream)
        if expected.tobytes() != restored.tobytes():
            raise AssertionError("Full-tile hard physical replay differs from independent receiver")
        summary.update(row, raw, restored, len(frame))
        (folder / (Path(row["sample_path"]).stem + ".frame")).write_bytes(frame)
    result = summary.result(len(registry) + len(geometry))
    result.update(update=update, model_sha256=identifier, full_shared_model_bytes=len(data),
                  registry_bytes=len(registry), independent_receiver_bits_equal=True)
    if config.get("architecture") == "local-internal-normalization-v1":
        result["local_statistic_coefficients"] = {
            name: float(.5 * torch.tanh(parameter.detach())) for name, parameter in model.named_parameters()
            if name.endswith(".gate")}
    write_json(folder / "results.json", result)
    return result, data


def evaluate(model, *args):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        return _evaluate(model, *args)


def load(row):
    if row["split"] != "train":
        raise ValueError("Only training signals authorized")
    with np.load(CORPUS / "train" / Path(row["sample_path"]).name, allow_pickle=False) as data:
        raw = data["data"]
    if raw.dtype != np.float32 or raw.shape != (int(row["channels"]), int(row["time_samples"])) or not np.isfinite(raw).all():
        raise ValueError("Invalid full-tile training contract")
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("flat", "product"), required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--normalization", choices=("global", "local"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=8192)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expanded-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(Path("vendor/smartgrids").resolve()))
    if not 1 <= args.stop_after <= 8192:
        raise ValueError("Registered pilot maximum is8192 updates")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(2117)
    np.random.seed(2117)
    torch.manual_seed(2117)
    rng = random.Random(2117)
    fitting, probes = select_probes(CORPUS)
    expanded = [r for r in probes if r["role"] == "check"]
    largest = {}
    smallest = {}
    for row in expanded:
        group, file_id = row["group_id"], row["drive_id"]
        if group not in largest or int(row["channels"]) > int(largest[group]["channels"]):
            largest[group] = row
        if file_id not in smallest or int(row["channels"]) < int(smallest[file_id]["channels"]):
            smallest[file_id] = row
    checks = sorted({r["sample_path"]: r for r in list(largest.values()) + list(smallest.values())}.values(),
                    key=lambda r: r["sample_path"])
    excluded = {r["drive_id"] for r in expanded}
    training = sorted([r for r in csv.DictReader((CORPUS / "samples_manifest.csv").open())
                       if r["split"] == "train" and r["drive_id"] not in excluded], key=lambda r: r["sample_path"])
    scales = json.loads(SCALES.read_text())["scales"]
    maps = {r["channel_map_id"]: r for r in map(json.loads, (CORPUS / "channel_maps.jsonl").open())}
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if args.normalization is None:
        if args.assets is None:
            raise ValueError("Product study requires its registered initialization")
        assets = torch.load(args.assets, map_location="cpu", weights_only=True)
        preparation = json.loads((args.assets.parent / "results.json").read_text())
        if (not (args.assets.parent / "COMPLETE").exists()
                or sha256(args.assets) != preparation["assets_sha256"]
                or json.loads((args.assets.parent / "protocol.json").read_text())["source_sha256"] != sha256(SOURCE)):
            raise ValueError("Product assets/source integrity mismatch")
    else:
        if args.variant != "flat" or args.assets is not None:
            raise ValueError("Local-statistics contrast must not combine product quantization")
        assets = {}
    code = [Path(__file__), Path("finetuning/dense_product_vq.py"), Path("finetuning/factorized_vq.py"),
            Path("finetuning/joint_anchor_vq.py"), Path("finetuning/full_tile_bottleneck.py"),
            Path("compression/full_tile_vq_stream.py"), Path("compression/alphabet_stream.py"),
            Path("finetuning/streaming_physical_summary.py"), Path("finetuning/prospective_metrics.py"),
            Path("finetuning/prospective_spectral.py"), Path("finetuning/model_factory.py"),
            Path("finetuning/latent_geometry.py"), Path("finetuning/robust_escape.py"),
            Path("finetuning/reversible_normalization.py"), Path("finetuning/run_joint_anchor_study.py")]
    if args.normalization is not None:
        code.append(Path("finetuning/local_internal_norm.py"))
    protocol = {
        "kind": "local-internal-norm-pilot-v1" if args.normalization else "dense-full-tile-product-pilot-v1",
        "variant": args.variant, "normalization": args.normalization,
        "source_sha256": sha256(SOURCE), "assets_sha256": sha256(args.assets) if args.assets else None,
        "seed": 2117, "maximum_updates": 8192, "smoke": args.smoke,
        "training_windows": len(training), "excluded_check_files": sorted(excluded),
        "fitting": [r["sample_path"] for r in fitting], "checks": [r["sample_path"] for r in checks],
        "expanded_checks": [r["sample_path"] for r in expanded],
        "manifest_sha256": sha256(CORPUS / "samples_manifest.csv"), "scales_sha256": sha256(SCALES),
        "input_sha256": {row["sample_path"]: sha256(CORPUS / "train" / Path(row["sample_path"]).name)
                         for row in {r["sample_path"]: r for r in training + fitting + expanded}.values()},
        "sources": {str(p): sha256(p) for p in code},
        "vendor_sources": {name: sha256(Path("vendor/smartgrids/vqvae") / name)
                           for name in ("train_vqvae.py", "evaluate_vqvae.py")},
        "objective": "0.5 masked MSE+0.5 masked L1+VQ, exact hard forward, one full-tile grid2048",
        "selection": "macro_families**2+.1*stream_bpc/.7, initialization eligible",
        "optimizer": "Fresh AdamW both arms lr1e-4 wd1e-5 clip1; FP32 deterministic",
        "limits": "Consumed training checks, not independent confirmation; no val/test waveform access"}
    if args.resume or args.expanded_only:
        if json.loads((args.output / "protocol.json").read_text()) != protocol:
            raise ValueError("Pilot resume code/protocol mismatch")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "protocol.json", protocol)
        snapshot = args.output / "SOURCE_SNAPSHOT"
        for path in code:
            relative = Path("finetuning") / path.name if path.is_absolute() else path
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    if args.normalization is None:
        model, config = initialize(source, args.variant, assets, "cuda")
    else:
        from finetuning.local_internal_norm import initialize as initialize_local
        model, config = initialize_local(source, args.normalization == "local", "cuda")
        if not args.resume and not args.expanded_only:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                original, _ = initialize(source, "flat", {}, "cuda")
                original.eval()
                model.eval()
                row = max(fitting, key=lambda r: int(r["channels"]))
                packet = encode_robust(load(row))
                x = torch.from_numpy(packet.values)[None, None].cuda()
                expected, _, before, _, _ = exact_branch_forward(original, x)
                actual, _, after, _, _ = exact_branch_forward(model, x)
                if not torch.equal(expected.view(torch.int32), actual.view(torch.int32)) or not torch.equal(before, after):
                    raise AssertionError("Neutral local-statistics path changed the initial hard function")
                write_json(args.output / "INITIAL_FUNCTION_PARITY.json", {
                    "normalized_float32_bits_equal": True, "paid_indices_equal": True,
                    "norm_layers": config["norm_layers"], "sample": row["sample_path"]})
                del original, expected, actual, x
    if args.expanded_only:
        if not (args.output / "PILOT_COMPLETE").exists():
            raise ValueError("Scientific pilot must finish before expansion")
        destination = args.output / "expanded"
        destination.mkdir(exist_ok=False)
        for label, checkpoint in (("selected", "best.pt"), ("endpoint", "last.pt")):
            saved = torch.load(args.output / checkpoint, map_location="cpu", weights_only=False)
            receiver = build(saved["config"], saved["model_state_dict"], "cuda")
            folder = destination / label
            result, _ = evaluate(receiver, saved["config"], fitting, expanded, scales, maps, folder,
                                 json.loads((args.output / "selection.json").read_text())["update"] if label == "selected" else saved["update"])
            print(json.dumps({"expanded": label, "macro": result["macro_families"], "bpc": result["stream_bpc"]}), flush=True)
            del receiver, saved
        # Both arms publish the frozen source under identical physical/wire contract.
        teacher, source_config = initialize(source, "flat", assets, "cuda")
        evaluate(teacher, source_config, fitting, expanded, scales, maps, destination / "mature_reference", 0)
        (destination / "COMPLETE").touch()
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    update, best, selected_update, cells = 0, float("inf"), 0, 0
    if args.resume:
        state = torch.load(args.output / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        update, best, selected_update, cells = state["update"], state["best"], state["selected_update"], state["cells"]
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    else:
        result, data = evaluate(model, config, fitting, checks, scales, maps, args.output, 0)
        best = score(result)
        (args.output / "best.pt").write_bytes(data)
        write_json(args.output / "selection.json", {"update": 0, "score": best})
        del data
    endpoint = min(2, args.stop_after) if args.smoke else args.stop_after
    smoke_rows = [max(training, key=lambda r: int(r["channels"])), min(training, key=lambda r: int(r["channels"]))]
    health = []
    with (args.output / "history.jsonl").open("a") as history:
        for update in range(update + 1, endpoint + 1):
            row = smoke_rows[update - 1] if args.smoke else training[rng.randrange(len(training))]
            raw = load(row)
            packet = encode_robust(raw)
            x = torch.from_numpy(packet.values)[None, None].cuda()
            mask = torch.from_numpy(packet.valid)[None, None].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted, quantizer_loss, _, _, perplexity = exact_branch_forward(model, x)
            difference = (predicted - x)[mask]
            distortion = .5 * difference.square().mean() + .5 * difference.abs().mean()
            loss = distortion + quantizer_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite dense product loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if args.smoke and args.normalization == "local" and update == 1:
                gradients = [parameter.grad for name, parameter in model.named_parameters() if name.endswith(".gate")]
                if (any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients)
                        or sum(float(gradient.abs().sum()) for gradient in gradients) == 0):
                    raise AssertionError("Local-statistics gates need finite nonzero joint gradients")
            if args.smoke and args.variant == "product":
                for parameter in (model.quantizer.embedding.left, model.quantizer.embedding.right):
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not parameter.grad.abs().sum() > 0:
                        raise AssertionError("Both product books must receive finite nonzero gradients")
            optimizer.step()
            if args.smoke and args.variant == "product":
                left, right = model.quantizer.embedding.left, model.quantizer.embedding.right
                if optimizer.state[left]["step"].data_ptr() == optimizer.state[right]["step"].data_ptr():
                    raise AssertionError("Product optimizer counters alias")
            cells += raw.size
            health.append(float(perplexity))
            if update % 32 == 0 or update == endpoint:
                history.write(json.dumps({"update": update, "perplexity": float(np.mean(health)),
                    "distortion_last": float(distortion.detach()), "gradient_norm": float(norm), "cells": cells}) + "\n")
                history.flush()
                health.clear()
            if should_evaluate(update, endpoint, args.smoke):
                result, data = evaluate(model, config, fitting, checks, scales, maps, args.output, update)
                if score(result) < best:
                    best, selected_update = score(result), update
                    (args.output / "best.pt").write_bytes(data)
                    write_json(args.output / "selection.json", {"update": update, "score": best})
                atomic_save({"config": config, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "update": update, "best": best,
                    "selected_update": selected_update, "cells": cells, "python_rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, args.output / "last.pt")
                print(json.dumps({"variant": args.variant, "update": update, "macro": result["macro_families"],
                                  "bpc": result["stream_bpc"], "selected_update": selected_update}), flush=True)
                del data
    if sha256(SOURCE) != protocol["source_sha256"]:
        raise AssertionError("Original source changed")
    (args.output / ("SMOKE_COMPLETE" if args.smoke else "PILOT_COMPLETE")).touch()


if __name__ == "__main__":
    main()
