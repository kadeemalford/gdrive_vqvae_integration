"""Equal-cell regime-diversity curve starting from verified pre-FEBUS OOI weights."""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.dense_product_vq import initialize
from finetuning.full_tile_bottleneck import select_probes
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.regime_diversity import crop, nested_groups
from finetuning.robust_escape import encode_robust
from finetuning.run_dense_product_study import CORPUS, SCALES, evaluate, load
from finetuning.run_joint_anchor_study import atomic_save

ORIGINAL = Path("models/febus_finetune_base/best.pt")
EXPECTED = "7d2da0d1cbd1d3dae579d7fd900850afc9ef15f645ea844168c0f78f18b0a27c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, choices=(4, 6, 10), required=True)
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
    torch.manual_seed(2117)
    rng = random.Random(2117)
    crop_rng = np.random.default_rng(2117)
    if sha256(ORIGINAL) != EXPECTED:
        raise ValueError("Expected original OOI source, never a FEBUS-adapted checkpoint")
    fitting, probes = select_probes(CORPUS)
    checks = [r for r in probes if r["role"] == "check"]
    excluded = {r["drive_id"] for r in checks}
    rows = [r for r in csv.DictReader((CORPUS / "samples_manifest.csv").open())
            if r["split"] == "train" and r["drive_id"] not in excluded]
    subsets = nested_groups(rows)
    selected = subsets[args.groups]
    calibration = [r for r in fitting if r["group_id"] in subsets[4]]
    training = [r for r in rows if r["group_id"] in selected
                and int(r["channels"]) >= 8 and int(r["time_samples"]) >= 1000]
    pool = defaultdict(lambda: defaultdict(list))
    for row in sorted(training, key=lambda r: r["sample_path"]):
        pool[row["group_id"]][row["drive_id"]].append(row)
    families = sorted({g.split("/")[0] for g in selected})
    if len(checks) != 136 or len(excluded) != 43 or set(pool) != set(selected) or not calibration:
        raise ValueError("Incomplete diversity population or calibration")
    if {r["drive_id"] for r in calibration} & excluded:
        raise ValueError("Calibration/check file overlap")
    endpoint = 2 if args.smoke else 8192
    scales = json.loads(SCALES.read_text())["scales"]
    maps = {r["channel_map_id"]: r for r in map(json.loads, (CORPUS / "channel_maps.jsonl").open())}
    sources = [Path(__file__).resolve().relative_to(Path.cwd().resolve()),
               Path("finetuning/regime_diversity.py"), Path("finetuning/run_dense_product_study.py"),
               Path("finetuning/dense_product_vq.py"), Path("finetuning/model_factory.py"),
               Path("finetuning/latent_geometry.py"), Path("finetuning/joint_anchor_vq.py"),
               Path("compression/full_tile_vq_stream.py"), Path("compression/alphabet_stream.py"),
               Path("finetuning/streaming_physical_summary.py"),
               Path("finetuning/prospective_metrics.py"), Path("finetuning/prospective_spectral.py"),
               Path("finetuning/reversible_normalization.py"), Path("finetuning/robust_escape.py")]
    hashes = {str(p): sha256(p) for p in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {
        "kind": "equal-cell-regime-diversity-v1", "group_count": args.groups,
        "subsets": subsets, "selected_groups": selected, "updates": endpoint, "smoke": args.smoke,
        "source_sha256": EXPECTED, "sources": hashes,
        "seed": 2117, "crop_shape": [8, 1000], "scales_sha256": sha256(SCALES),
        "training_files": sorted({r["drive_id"] for r in training}),
        "calibration": calibration, "checks": checks,
        "input_sha256": {r["sample_path"]: sha256(CORPUS / "train" / Path(r["sample_path"]).name)
                         for r in {r["sample_path"]: r for r in training + calibration + checks}.values()},
        "reserved_waveform_access": False,
        "limits": "Configuration diversity, not independent-session scaling; fixed endpoint, not convergence."})
    for path in sources:
        target = args.output / "SOURCE_SNAPSHOT" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    source = torch.load(ORIGINAL, map_location="cpu", weights_only=False)
    source = {**source, "config": {**source["config"], "latent_geometry": "spatial4-temporal4"}}
    model, config = initialize(source, "flat", {}, "cuda")
    del source
    eval_rows = checks if not args.smoke else [checks[0], checks[-1]]
    initial, _ = evaluate(model, config, calibration, eval_rows, scales, maps, args.output, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    family_updates = defaultdict(int)
    group_updates = defaultdict(int)
    with (args.output / "history.jsonl").open("w") as history:
        for update in range(1, endpoint + 1):
            family = families[(update - 1) % len(families)]
            group = rng.choice([g for g in selected if g.split("/")[0] == family])
            file_id = rng.choice(sorted(pool[group]))
            row = rng.choice(pool[group][file_id])
            raw, _, _ = crop(load(row), crop_rng)
            packet = encode_robust(raw)
            x = torch.from_numpy(packet.values)[None, None].cuda()
            mask = torch.from_numpy(packet.valid)[None, None].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predicted, qloss, _, _, perplexity = exact_branch_forward(model, x)
            error = (predicted - x)[mask]
            distortion = .5 * error.square().mean() + .5 * error.abs().mean()
            loss = distortion + qloss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite regime-diversity loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            family_updates[family] += 1
            group_updates[group] += 1
            if update % 32 == 0 or update == endpoint:
                history.write(json.dumps({"update": update, "cells": update * 8000,
                    "distortion": float(distortion.detach()), "perplexity": float(perplexity),
                    "gradient_norm": float(norm)}) + "\n")
                history.flush()
            if update % 2048 == 0 or update == endpoint:
                atomic_save({"config": config, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "update": update},
                    args.output / "last.pt")
                print(json.dumps({"groups": args.groups, "update": update, "cells": update * 8000}), flush=True)
    final, _ = evaluate(model, config, calibration, eval_rows, scales, maps, args.output, endpoint)
    if sha256(ORIGINAL) != EXPECTED or any(sha256(Path(p)) != digest for p, digest in hashes.items()):
        raise AssertionError("Diversity sources changed")
    write_json(args.output / "results.json", {
        "groups": args.groups, "updates": endpoint, "cells": endpoint * 8000,
        "family_updates": dict(family_updates), "group_updates": dict(group_updates),
        "initial_nrmse_star": initial["macro_families"], "initial_bpc": initial["stream_bpc"],
        "endpoint_nrmse_star": final["macro_families"], "endpoint_bpc": final["stream_bpc"],
        "check_windows": len(eval_rows), "reserved_waveform_access": False})
    (args.output / ("SMOKE_COMPLETE" if args.smoke else "COMPLETE")).touch()


if __name__ == "__main__":
    main()
