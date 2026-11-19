"""Joint H2 pilot/conditional continuation; two neural branches and exact hard replay."""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.fit_lifting_allocation import fitting_rows
from finetuning.joint_anchor_evaluation import evaluate, practical_reference, load_export
from finetuning.joint_anchor_vq import initialize
from finetuning.model_factory import build_model
from finetuning.robust_escape import encode_robust
from finetuning.run_spatial_opportunity import load, select


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def score(result):
    return result["macro_families"] ** 2 + .1 * result["stream_bpc"] / .8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--encoder-context", choices=("local", "full"))
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("independent", "closed-loop"), required=True)
    parser.add_argument("--maximum-updates", type=int, default=32768)
    parser.add_argument("--stop-after", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2115)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(args.vendor.resolve()))
    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA required for joint H2 training")
    if args.maximum_updates > 32768 or args.stop_after > args.maximum_updates or args.stop_after < 1:
        raise ValueError("Budget exceeds the registered H2 maximum")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    selected = select(args.corpus)
    excluded = {r["drive_id"] for r in selected if r["role"] == "check"}
    all_rows = list(csv.DictReader((args.corpus / "samples_manifest.csv").open()))
    training = sorted([r for r in all_rows if r["split"] == "train" and r["drive_id"] not in excluded],
                      key=lambda r: r["sample_path"])
    fitting = fitting_rows(selected)
    checks = {}
    for row in selected:
        if row["role"] == "check":
            checks.setdefault(row["group_id"], row)
    checks = list(checks.values())
    scales = json.loads(args.calibration.read_text())["scales"]
    maps = {m["channel_map_id"]: m for m in map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    source_hash = sha256(args.source)
    reference_source = args.reference_source or args.source
    reference_hash = sha256(reference_source)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    code = [Path(__file__), Path("finetuning/joint_anchor_vq.py"),
            Path("finetuning/joint_anchor_evaluation.py"), Path("compression/joint_anchor_vq_stream.py"),
            Path("compression/alphabet_stream.py"), Path("finetuning/prospective_metrics.py"),
            Path("finetuning/prospective_spectral.py"), Path("finetuning/evaluate_lifting_curve.py"),
            Path("finetuning/fit_lifting_allocation.py"), Path("finetuning/model_factory.py"),
            Path("finetuning/latent_geometry.py"), Path("finetuning/robust_escape.py"),
            Path("finetuning/reversible_normalization.py")]
    if args.encoder_context is not None:
        code.append(Path("finetuning/encoder_observation_vq.py"))
    protocol = {
        "kind": "joint_encoder_observation_controlled_v1" if args.encoder_context else "joint_anchor_innovation_controlled_v1",
        "mode": args.mode, "encoder_context": args.encoder_context, "seed": args.seed,
        "source": str(args.source), "source_sha256": source_hash,
        "reference_source": str(reference_source), "reference_source_sha256": reference_hash,
        "maximum_updates": args.maximum_updates, "pilot_updates": 8192, "smoke": args.smoke,
        "train_windows": len(training), "train_file_exclusion": sorted(excluded),
        "fitting": [r["sample_path"] for r in fitting], "checks": [r["sample_path"] for r in checks],
        "manifest_sha256": sha256(args.corpus / "samples_manifest.csv"),
        "calibration_sha256": sha256(args.calibration),
        "sources": {str(p): sha256(p) for p in code},
        "vendor_sources": {name: sha256(args.vendor / "vqvae" / name)
                           for name in ("train_vqvae.py", "evaluate_vqvae.py")},
        "optimizer": "Fresh AdamW both arms, lr1e-4, wd1e-5, clip1, FP32, deterministic CUDA",
        "objective": "0.5 masked MSE + 0.5 masked L1 + mean of two VQ losses; not an explicit RD loss",
        "branches": "two separate dense mature VQ copies, jointly trainable, identical base initialization, no alias",
        "predictor": "independent: zero; closed-loop: adjacent-anchor mean plus zero-initialized 3x5 CNN width32",
        "quantization": "exact hard vector forward with zero-valued z-z.detach() bridge; no uncoded residue",
        "selection": "min macro_families**2 + .1*stream_bpc/.8, initialization eligible",
        "wire": "FJH1, two FAK1 books, compressed exact calibration counts charged, one global FNE1 side",
        "reference": "same full-tile mature VQ measured on identical checks with same FAK1 compact registry",
        "exposure": "Source historically saw training-file checks; OOI/FEBUS exposure inherited by both branches",
        "limits": "internal training-file checks, not independent validation; nominal grids common but actual bytes measured"}
    protocol["environment"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    if args.encoder_context is not None:
        protocol.update(
            branches="Previously selected H2 pair, both arms same warm start; encoder context adapters zero-initialized",
            predictor="Same previously trained closed-loop predictor in both arms",
            contrast="Encoder cross-phase acquired observations disabled/enabled; decoder/quantizers/grids common",
            exposure="Dense source history plus selected H2 update16384 inherited; fresh AdamW in both new arms")
    if args.resume:
        if json.loads((args.output / "protocol.json").read_text()) != protocol:
            raise ValueError("Joint H2 resume protocol/source mismatch")
    else:
        write_json(args.output / "protocol.json", protocol)
    if args.encoder_context is not None:
        if args.mode != "closed-loop":
            raise ValueError("Observation contrast keeps the selected closed-loop predictor in both arms")
        from finetuning.encoder_observation_vq import initialize as initialize_observation
        model, config = initialize_observation(source, args.encoder_context == "full", "cuda")
        protocol_note = {
            "source": "Previously selected H2 update16384; both arms start from identical source",
            "contrast": "zero-initialized encoder cross-phase context, disabled/enabled; all operators jointly adapt",
            "predictor": "same previously trained closed-loop predictor in both arms",
            "new_context_parameters": sum(p.numel() for n, p in model.named_parameters() if ".extra." in n)}
        if not args.resume:
            write_json(args.output / "observation_design.json", protocol_note)
    else:
        model, config = initialize(source, args.mode, "cuda")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    update, best = 0, float("inf")
    selected_update = 0
    cells_seen = 0
    if args.resume:
        state = torch.load(args.output / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        update, best, selected_update = state["update"], state["best"], state["selected_update"]
        cells_seen = state["cells_seen"]
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    else:
        if args.encoder_context is not None:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
                original = load_export(args.source.read_bytes(), "cuda")
                parity_row = max(fitting, key=lambda r: int(r["channels"]))
                parity_packet = encode_robust(load(args.corpus, parity_row))
                x_parity = torch.from_numpy(parity_packet.values)[None, None].cuda()
                expected, _, old_info = original(x_parity)
                model.eval()
                actual, _, new_info = model(x_parity)
                if (not torch.equal(expected.view(torch.int32), actual.view(torch.int32))
                        or not torch.equal(old_info["anchor_ids"], new_info["anchor_ids"])
                        or not torch.equal(old_info["innovation_ids"], new_info["innovation_ids"])):
                    raise AssertionError("Zero-initialized observation adapter changed the source function")
                write_json(args.output / "INITIAL_FUNCTION_PARITY.json", {"physical_normalized_float32_bits_equal": True,
                           "both_index_streams_equal": True, "sample": parity_row["sample_path"]})
                del original, x_parity, expected, actual
        reference_saved = source if reference_source == args.source else torch.load(reference_source, map_location="cpu", weights_only=False)
        reference = build_model({**reference_saved["config"], "exact_eval_quantization": True}, reference_saved, "cuda").eval()
        reference.requires_grad_(False)
        practical_reference(reference, reference_hash, fitting, checks, args.corpus, scales, maps,
                            args.output / "practical_reference")
        del reference
        result, data = evaluate(model, config, fitting, checks, args.corpus, scales, maps, args.output, 0)
        best = score(result)
        (args.output / "best.pt").write_bytes(data)
        write_json(args.output / "selection.json", {"update": 0, "score": best})
        del data
    if args.smoke:
        endpoint = min(2, args.stop_after)
        smoke_rows = [max(training, key=lambda r: int(r["channels"])),
                      min(training, key=lambda r: int(r["channels"]))]
    else:
        endpoint = args.stop_after
    start = time.monotonic()
    losses, anchor_perplexities, innovation_perplexities = [], [], []
    with (args.output / "history.jsonl").open("a") as history:
        for update in range(update + 1, endpoint + 1):
            row = smoke_rows[update - 1] if args.smoke else training[rng.randrange(len(training))]
            raw = load(args.corpus, row)
            packet = encode_robust(raw)
            x = torch.from_numpy(packet.values)[None, None].cuda()
            mask = torch.from_numpy(packet.valid)[None, None].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted, quantizer_loss, information = model(x)
            difference = (predicted - x)[mask]
            distortion = .5 * difference.square().mean() + .5 * difference.abs().mean()
            loss = distortion + quantizer_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite joint H2 objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if args.smoke and update == 1 and args.mode == "closed-loop":
                gradient = model.predictor.net[-1].weight.grad
                if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
                    raise AssertionError("Predictor must receive a finite nonzero joint gradient")
            if args.smoke and update == 1 and args.encoder_context is not None:
                gradients = [parameter.grad for name, parameter in model.named_parameters() if ".extra." in name]
                if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
                    raise AssertionError("Invalid observation adapter gradient")
                magnitude = sum(float(gradient.abs().sum()) for gradient in gradients)
                if (args.encoder_context == "full" and magnitude == 0
                        or args.encoder_context == "local" and magnitude != 0):
                    raise AssertionError("Encoder observation mask/gradient isolation failed")
            optimizer.step()
            if update == 1 and information["innovation_ids"] is not None:
                a = optimizer.state[model.anchor.quantizer.embedding.weight]["step"]
                b = optimizer.state[model.innovation.quantizer.embedding.weight]["step"]
                if a.data_ptr() == b.data_ptr():
                    raise AssertionError("Anchor/innovation optimizer counter alias")
            cells_seen += raw.size
            losses.append(float(distortion.detach()))
            anchor_perplexities.append(float(information["anchor_perplexity"]))
            if information["innovation_perplexity"] is not None:
                innovation_perplexities.append(float(information["innovation_perplexity"]))
            if update % 32 == 0 or update == endpoint:
                record = {"update": update, "distortion_mean": float(np.mean(losses)),
                          "anchor_perplexity_mean": float(np.mean(anchor_perplexities)),
                          "innovation_perplexity_mean": float(np.mean(innovation_perplexities)) if innovation_perplexities else None,
                          "gradient_norm": float(norm), "cells_seen": cells_seen,
                          "elapsed_seconds_this_run": time.monotonic() - start}
                history.write(json.dumps(record, allow_nan=False) + "\n")
                history.flush()
                losses.clear()
                anchor_perplexities.clear()
                innovation_perplexities.clear()
            if update % 2048 == 0 or update == endpoint:
                result, data = evaluate(model, config, fitting, checks, args.corpus, scales, maps, args.output, update)
                if score(result) < best:
                    best, selected_update = score(result), update
                    (args.output / "best.pt").write_bytes(data)
                    write_json(args.output / "selection.json", {"update": update, "score": best})
                atomic_save({"config": config, "model_state_dict": model.state_dict(),
                             "optimizer_state_dict": optimizer.state_dict(), "update": update,
                             "best": best, "selected_update": selected_update, "cells_seen": cells_seen,
                             "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                             "cuda_rng": torch.cuda.get_rng_state()}, args.output / "last.pt")
                print(json.dumps({"update": update, "mode": args.mode,
                                  "macro": result["macro_families"], "bpc": result["stream_bpc"],
                                  "selected_update": selected_update,
                                  "max_gpu_memory_bytes": torch.cuda.max_memory_allocated()}), flush=True)
                del data
    if sha256(args.source) != source_hash:
        raise AssertionError("Original source checkpoint changed")
    marker = "SMOKE_COMPLETE" if args.smoke else (
        "BUDGET_COMPLETE" if endpoint == args.maximum_updates else "PILOT_COMPLETE")
    (args.output / marker).touch()


if __name__ == "__main__":
    main()
