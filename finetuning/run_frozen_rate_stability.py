#!/usr/bin/env python3
"""Frozen retained-checkpoint and fixed midpoint controls; no optimization."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

import torch

try:
    from .finetune_vqvae import save_checkpoint
    from .run_sampling_experiment import ROOT, run
    from .run_temporal_density_long import digest, require_cuda
except ImportError:
    from finetune_vqvae import save_checkpoint
    from run_sampling_experiment import ROOT, run
    from run_temporal_density_long import digest, require_cuda


def midpoint_state(before, after):
    if before.keys() != after.keys():
        raise ValueError("State keys differ")
    result = {}
    for key, a in before.items():
        b = after[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("State tensor shape/dtype differs")
        if a.is_floating_point():
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Nonfinite source weights")
            result[key] = (a.double() * .5 + b.double() * .5).to(a.dtype)
            if not torch.isfinite(result[key]).all():
                raise ValueError("Nonfinite averaged weights")
        else:
            if not torch.equal(a, b):
                raise ValueError("Nonfloating state differs")
            result[key] = a.clone()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    initial = ROOT / "results/experiments/high_rate_fold0_robust_20261017"
    continued = ROOT / "results/experiments/robust_rate_continuation_20261018"
    audit = ROOT / "results/experiments/robust_rate_paired_audit_20261018"
    if not all((p / "COMPLETE").is_file() for p in (initial, continued, audit)):
        raise ValueError("Completed source studies and paired audit required")
    sources = {}
    for arm in ("2x", "4x"):
        for label, path in (("initial", initial / f"rate_{arm}/best_macro.pt"),
                            ("continued", continued / f"rate_{arm}/best_macro.pt"),
                            ("loss_selected", continued / f"rate_{arm}/best.pt")):
            sources[f"{label}_{arm}"] = path
    code = [ROOT / f"finetuning/{n}.py" for n in (
        "run_frozen_rate_stability", "finetune_vqvae", "model_factory", "latent_geometry",
        "evaluate_physical_candidates", "physical_validation", "robust_escape", "reversible_normalization",
        "channel_normalization", "reconstruction_objective", "run_sampling_experiment", "run_temporal_density_long")]
    code += [Path(args.repo_root) / f"vqvae/{n}.py" for n in ("train_vqvae", "evaluate_vqvae")]
    manifest = ROOT / "data/finetune/fold_0/samples_manifest.csv"
    protocol = {"kind": "frozen_rate_stability_v1", "training_updates": 0, "fold": 0, "split": "val",
                "midpoint_weight": .5, "repo_root": str(Path(args.repo_root).resolve()),
                "sources": {label: {"path": str(p), "sha256": digest(p)} for label, p in sources.items()},
                "manifest_sha256": digest(manifest), "audit_sha256": digest(audit / "configurations.csv"),
                "code_sha256": {str(p): digest(p) for p in code},
                "note": "Fixed 50/50 state midpoint includes VQ embeddings with unchanged code indices, but nonlinear benefits are not assumed. Evaluate endpoints, retained loss-selected epoch38, and midpoint on full reused validation. No ensemble at inference, extra parameters, training, test access, alpha sweep or automatic promotion. Validation-guided exploratory diagnostics only."}
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[frozen stability] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("New directory or --resume required")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 2 * 1024 ** 3:
        raise RuntimeError("2 GiB free required; no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        candidates = dict(sources)
        for arm, geometry in (("2x", "spatial4-temporal2"), ("4x", "spatial4-temporal4")):
            parents = [torch.load(sources[f"{label}_{arm}"], map_location="cpu", weights_only=False)
                       for label in ("initial", "continued")]
            if parents[0]["config"] != parents[1]["config"] or [p["epoch"] for p in parents] != [24, 40]:
                raise ValueError("Source config/epoch mismatch")
            for p in parents:
                state = p["finetune"]
                if (state["normalization"] != "robust-escape-v1" or state["latent_geometry"] != geometry
                        or Path(state["samples_dir"]).resolve() != manifest.parent):
                    raise ValueError("Source representation/geometry/fold mismatch")
            state = midpoint_state(*(p["model_state_dict"] for p in parents))
            path = output / f"midpoint_{arm}.pt"
            saved = {"config": parents[0]["config"], "model_state_dict": state, "epoch": 0,
                     "finetune": {"normalization": "robust-escape-v1", "latent_geometry": geometry,
                                  "samples_dir": str(manifest.parent), "physical_validation": True},
                     "frozen_midpoint": {"weight": .5, "parent_epochs": [24, 40], "training_updates": 0,
                                         "parent_sha256": [protocol["sources"][f"{label}_{arm}"]["sha256"]
                                                           for label in ("initial", "continued")]}}
            if path.exists():
                existing = torch.load(path, map_location="cpu", weights_only=False)
                if (existing.get("frozen_midpoint") != saved["frozen_midpoint"]
                        or existing["config"] != saved["config"] or existing["finetune"] != saved["finetune"]
                        or existing["model_state_dict"].keys() != state.keys()
                        or any(not torch.equal(existing["model_state_dict"][k], v) for k, v in state.items())):
                    raise ValueError("Existing midpoint provenance/weights mismatch")
                del existing
            else:
                save_checkpoint(saved, path)
            candidates[f"midpoint_{arm}"] = path
            del parents, state, saved
        full = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root,
                   "--fold", "0", "--output-dir", str(full)]
        for label, path in candidates.items():
            command += ["--candidate", f"{label}=robust-escape-v1={path}"]
        if full.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=args.resume, quiet=True)
        (output / "COMPLETE").write_text("Frozen rate stability full validation complete; no training or promotion.\n")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
