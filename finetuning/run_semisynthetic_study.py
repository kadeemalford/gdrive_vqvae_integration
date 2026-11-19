"""Fixed-budget real/mixed continuation, evaluated exclusively on measured train checks."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.dense_product_vq import initialize
from finetuning.full_tile_bottleneck import select_probes
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.robust_escape import encode_robust
from finetuning.run_dense_product_study import CORPUS, SCALES, SOURCE, evaluate, load
from finetuning.run_joint_anchor_study import atomic_save
from finetuning.semisynthetic_fields import compose

UPDATES = 8192


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("real", "mixed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(Path("vendor/smartgrids").resolve()))
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
    augmentation_rng = np.random.default_rng(104)
    fitting, probes = select_probes(CORPUS)
    checks = [r for r in probes if r["role"] == "check"]
    excluded = {r["drive_id"] for r in checks}
    training = sorted(
        [r for r in csv.DictReader((CORPUS / "samples_manifest.csv").open())
         if r["split"] == "train" and r["drive_id"] not in excluded],
        key=lambda r: r["sample_path"])
    if (len(checks) != 136 or not training or len(excluded) != 43
            or any(r["split"] != "train" for r in fitting + training + checks)
            or {r["drive_id"] for r in fitting} & excluded):
        raise ValueError("Semisynthetic train/check population mismatch")
    scales = json.loads(SCALES.read_text())["scales"]
    maps = {r["channel_map_id"]: r for r in map(json.loads, (CORPUS / "channel_maps.jsonl").open())}
    sources = [Path(__file__).resolve().relative_to(Path.cwd().resolve()),
               Path("finetuning/semisynthetic_fields.py"), Path("finetuning/run_dense_product_study.py"),
               Path("finetuning/dense_product_vq.py"), Path("finetuning/joint_anchor_vq.py"),
               Path("finetuning/model_factory.py"), Path("finetuning/latent_geometry.py"),
               Path("compression/full_tile_vq_stream.py"), Path("compression/alphabet_stream.py"),
               Path("finetuning/streaming_physical_summary.py"), Path("finetuning/prospective_metrics.py"),
               Path("finetuning/prospective_spectral.py"), Path("finetuning/reversible_normalization.py"),
               Path("finetuning/robust_escape.py")]
    hashes = {str(p): sha256(p) for p in sources}
    endpoint = 2 if args.smoke else UPDATES
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "kind": "measured-carrier-semisynthetic-v1", "method": args.method, "smoke": args.smoke,
        "updates": endpoint, "sampling_seed": 2117, "augmentation_seed": 104,
        "source_sha256": sha256(SOURCE), "scales_sha256": sha256(SCALES),
        "sources": hashes, "training_windows": len(training),
        "training_files": sorted({r["drive_id"] for r in training}),
        "check_files": sorted(excluded), "checks": checks,
        "input_sha256": {r["sample_path"]: sha256(CORPUS / "train" / Path(r["sample_path"]).name)
                         for r in {r["sample_path"]: r for r in fitting + training + checks}.values()},
        "optimizer": "fresh AdamW lr1e-4 wd1e-5 clip1; FP32",
        "selection": "endpoint only", "reserved_waveform_access": False,
        "limits": "Complete measured carriers, not pure backgrounds; no session independence certificate."}
    write_json(args.output / "protocol.json", protocol)
    for path in sources:
        target = args.output / "SOURCE_SNAPSHOT" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model, config = initialize(saved, "flat", {}, "cuda")
    del saved
    selected_checks = checks
    if args.smoke:
        selected_checks = [max(checks, key=lambda r: int(r["channels"])),
                           min(checks, key=lambda r: int(r["channels"]))]
    initial, _ = evaluate(model, config, fitting, selected_checks, scales, maps, args.output, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    cells, augmented, augmented_cells = 0, 0, 0
    schedule = hashlib.sha256()
    smoke_rows = [max(training, key=lambda r: int(r["channels"])),
                  min(training, key=lambda r: int(r["channels"]))]
    with (args.output / "history.jsonl").open("w") as history, (args.output / "compositions.jsonl").open("w") as compositions:
        for update in range(1, endpoint + 1):
            row = smoke_rows[update - 1] if args.smoke else training[rng.randrange(len(training))]
            raw = load(row)
            schedule.update(f"{update}:{row['sample_path']}:{raw.size}\n".encode())
            if args.method == "mixed" and update % 2 == 0:
                raw, metadata = compose(raw, float(row["dt_s"]), float(row["dx_m"]),
                                        scales[row["group_id"]], augmentation_rng)
                compositions.write(json.dumps({"update": update, "sample_path": row["sample_path"], **metadata}) + "\n")
                augmented += 1
                augmented_cells += raw.size
            packet = encode_robust(raw)
            x = torch.from_numpy(packet.values)[None, None].cuda()
            mask = torch.from_numpy(packet.valid)[None, None].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, quantizer_loss, _, _, perplexity = exact_branch_forward(model, x)
            error = (prediction - x)[mask]
            distortion = .5 * error.square().mean() + .5 * error.abs().mean()
            loss = distortion + quantizer_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite semisynthetic training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            cells += raw.size
            if update % 32 == 0 or update == endpoint:
                history.write(json.dumps({"update": update, "cells": cells, "augmented_updates": augmented,
                    "distortion": float(distortion.detach()), "gradient_norm": float(norm),
                    "perplexity": float(perplexity)}) + "\n")
                history.flush()
            if update % 2048 == 0 or update == endpoint:
                atomic_save({"config": config, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "update": update,
                    "cells": cells, "sampling_rng": rng.getstate(),
                    "augmentation_rng": augmentation_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                    args.output / "last.pt")
                print(json.dumps({"method": args.method, "update": update, "cells": cells,
                                  "augmented_updates": augmented}), flush=True)
    final, _ = evaluate(model, config, fitting, selected_checks, scales, maps, args.output, endpoint)
    if sha256(SOURCE) != protocol["source_sha256"] or any(sha256(Path(p)) != h for p, h in hashes.items()):
        raise AssertionError("Frozen source changed during semisynthetic run")
    write_json(args.output / "results.json", {
        "method": args.method, "updates": endpoint, "cells": cells, "augmented_updates": augmented,
        "augmented_cells": augmented_cells, "schedule_sha256": schedule.hexdigest(),
        "initial_nrmse_star": initial["macro_families"], "endpoint_nrmse_star": final["macro_families"],
        "initial_bpc": initial["stream_bpc"], "endpoint_bpc": final["stream_bpc"],
        "endpoint_model_bytes": final["full_shared_model_bytes"],
        "check_windows": len(selected_checks), "reserved_waveform_access": False})
    (args.output / ("SMOKE_COMPLETE" if args.smoke else "COMPLETE")).touch()


if __name__ == "__main__":
    main()
