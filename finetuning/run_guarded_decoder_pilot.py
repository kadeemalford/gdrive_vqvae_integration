#!/usr/bin/env python3
"""Prospective full/decoder-only low-LR pilot with full-validation snapshots."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil

import torch

try:
    from .finetune_vqvae import save_checkpoint
    from .generalist_guard import assess, GUARD_METRICS
    from .run_sampling_experiment import ROOT, check_splits, run
    from .run_temporal_density_long import digest, require_cuda
    from .reconstruction_objective import path_fingerprint
except ImportError:
    from finetune_vqvae import save_checkpoint
    from generalist_guard import assess, GUARD_METRICS
    from run_sampling_experiment import ROOT, check_splits, run
    from run_temporal_density_long import digest, require_cuda
    from reconstruction_objective import path_fingerprint


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def weights_only(saved):
    return {"config": saved["config"], "model_state_dict": saved["model_state_dict"], "epoch": saved["epoch"],
            "finetune": {k: v for k, v in saved["finetune"].items() if k != "rng_state"},
            "snapshot_note": "Frozen inference snapshot; resume training from branch last.pt only."}


def trim_uncommitted_history(target, epoch):
    for name in ("metrics.csv", "validation_metrics.csv", "validation_groups.csv"):
        path = target / name
        if not path.exists():
            continue
        history = rows(path)
        retained = [r for r in history if int(r["epoch"]) <= epoch]
        if retained == history:
            continue
        # Preserve interrupted diagnostic records rather than silently discarding them.
        suffix = len(list(target.glob(f"{name}.uncommitted_*"))) + 1
        shutil.copy2(path, target / f"{name}.uncommitted_{suffix:03d}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with path.open() as stream:
            fields = next(csv.reader(stream))
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained)
        temporary.replace(path)


def verify_phase0(baseline, history):
    initial = [r for r in history if int(r["epoch"]) == 0]
    assess(baseline, initial)  # Coverage/count/finite checks; not candidate admission.
    key = lambda r: (r["family"], r["configuration"])
    if any(abs(float(a[m]) - float(b[m])) > 5e-7
           for a, b in zip(sorted(baseline, key=key), sorted(initial, key=key)) for m in GUARD_METRICS):
        raise ValueError("Source/phase0 physical validation mismatch")
    return initial


def validate_training_source(saved, samples, normalization):
    state = saved["finetune"]
    if (saved["epoch"] != 24 or state["normalization"] != normalization
            or state["latent_geometry"] != "spatial4-temporal4"
            or Path(state["samples_dir"]).resolve() != samples.resolve()):
        raise ValueError("Source epoch/normalization/geometry/fold mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--channel-tail-arm-only", action="store_true", help="Run the fixed 0.05 channel-MSE arm against the previously completed full control.")
    parser.add_argument("--fold1-channel-confirmation", action="store_true", help="Paired fresh full/channel-tail arms on fold 1, fixed auxiliary term and new seed.")
    args = parser.parse_args()
    if args.channel_tail_arm_only and args.fold1_channel_confirmation:
        raise ValueError("Choose one channel experiment mode")
    output = args.output_dir.resolve()
    fold = 1 if args.fold1_channel_confirmation else 0
    normalization = "escape-v1" if fold == 1 else "robust-escape-v1"
    seed = 2030 if fold == 1 else 2029
    prior = ROOT / ("results/experiments/channel_tail_pilot_20261019" if fold == 1
                    else "results/experiments/norm_matched_quantization_20261019")
    if not (prior / "COMPLETE").is_file():
        raise ValueError("Completed latent control required")
    source = ROOT / ("results/experiments/neural_rate_sweep_20261014/rate_4x/best_macro.pt" if fold == 1
                     else "results/experiments/high_rate_fold0_robust_20261017/rate_4x/best_macro.pt")
    smoke = ROOT / ("results/experiments/channel_fold1_smoke_20261019" if fold == 1
                    else "results/experiments/guarded_decoder_smoke_20261019")
    if not (smoke / "COMPLETE").is_file():
        raise ValueError("Full/decoder train-gradient smoke required")
    if json.loads((smoke / "summary.json").read_text())["source_sha256"] != digest(source):
        raise ValueError("Smoke/source mismatch")
    samples = ROOT / f"data/finetune/fold_{fold}"
    check_splits(samples, fold)
    manifest = samples / "samples_manifest.csv"
    manifest_rows = rows(manifest)
    if any(not (ROOT / r["sample_path"]).is_file() for r in manifest_rows if r["split"] in ("train", "val")):
        raise ValueError("Complete train/validation arrays required")
    validation_hash = path_fingerprint(sorted((samples / "val").glob("*.npz")))
    reference = ROOT / ("results/experiments/high_rate_continuation_20261015/full_validation/initial_rate4.groups.csv" if fold == 1
                        else "results/experiments/frozen_quantization_probe_20261019/rate4_vq.groups.csv")
    reference_protocol = reference.parent / "protocol.json"
    previous = json.loads(reference_protocol.read_text())
    reference_source_hashes = ([c["sha256"] for c in previous["candidates"] if c["label"] == "initial_rate4"] if fold == 1
                               else [previous["sources"]["rate4"]["sha256"]])
    if previous["validation_hash"] != validation_hash or reference_source_hashes != [digest(source)]:
        raise ValueError("Reference source/validation coordinates mismatch")
    baseline = rows(reference)
    code = [ROOT / f"finetuning/{n}.py" for n in (
        "run_guarded_decoder_pilot", "generalist_guard", "channel_objective", "finetune_vqvae", "model_factory", "latent_geometry",
        "physical_validation", "robust_escape", "reversible_normalization", "channel_normalization",
        "reconstruction_objective", "sampling", "codebook_refresh", "domain_routing", "run_sampling_experiment",
        "run_temporal_density_long", "evaluate_physical_candidates")]
    code += [Path(args.repo_root) / f"vqvae/{n}.py" for n in ("train_vqvae", "evaluate_vqvae")]
    protocol = {"kind": "guarded_full_decoder_low_lr_v1", "fold": fold, "arms": ["full", "decoder"],
                "source": str(source), "source_sha256": digest(source), "source_epoch": 24,
                "repo_root": str(Path(args.repo_root).resolve()), "seed": seed, "epochs": 8, "steps_per_epoch": 4096,
                "learning_rate": 3e-6, "weight_decay": 1e-5, "normalization": normalization,
                "latent_geometry": "spatial4-temporal4", "sampling": "window", "optimizer": "fresh AdamW per arm",
                "validation_hash": validation_hash, "manifest_sha256": digest(manifest),
                "reference_groups_sha256": digest(reference), "smoke_sha256": digest(smoke / "summary.json"),
                "reference_protocol_sha256": digest(reference_protocol),
                "guard_metrics": GUARD_METRICS, "guard_epsilon": 1e-9,
                "code_sha256": {str(p): digest(p) for p in code},
                "note": "Prospective pilot, not independent acquisition confirmation. Same source/LR/updates/objective/rate and full validation every epoch; only trainable scope differs, so compute/optimized parameter count differs. Retain all eight weights-only snapshots plus macro control and fixed-baseline guarded choice. No VQ bypass, aux loss, averaging, calibration or codebook refresh. Guard checks all configs RMSE/correlation/channel p90 versus fixed phase0; numerical epsilon is not physical tolerance. No test access or final promotion."}
    if args.channel_tail_arm_only:
        control = ROOT / "results/experiments/guarded_decoder_pilot_20261019"
        if not (control / "COMPLETE").is_file() or not (control / "full_validation/COMPLETE").is_file():
            raise ValueError("Completed matched full control required; do not repeat it")
        old = json.loads((control / "protocol.json").read_text())
        for key in ("source_sha256", "seed", "epochs", "steps_per_epoch", "learning_rate", "weight_decay",
                    "normalization", "latent_geometry", "sampling", "validation_hash", "manifest_sha256", "guard_metrics", "guard_epsilon"):
            if old[key] != protocol[key]:
                raise ValueError(f"Matched-control policy differs: {key}")
        tail_smoke = ROOT / "results/experiments/channel_tail_smoke_20261019"
        if not (tail_smoke / "COMPLETE").is_file():
            raise ValueError("Channel-tail finite-gradient smoke required")
        smoke_state = json.loads((tail_smoke / "summary.json").read_text())
        if (smoke_state["source_sha256"] != digest(source)
                or smoke_state["objective_sha256"] != digest(ROOT / "finetuning/channel_objective.py")):
            raise ValueError("Channel-tail smoke source/objective mismatch")
        protocol.update(kind="guarded_bounded_channel_mse_v1", arms=["channel_tail"], channel_mse_weight=.05,
                        channel_weight_cap=4., matched_control=str(control), control_protocol_sha256=digest(control / "protocol.json"),
                        control_history_sha256=digest(control / "full/validation_groups.csv"),
                        control_summary_sha256=digest(control / "full_validation/summary.csv"),
                        channel_smoke_sha256=digest(tail_smoke / "summary.json"),
                        note="Reuse immutable completed full control; fresh full-network arm differs only by fixed 0.05 bounded channel MSE auxiliary training term. Same source/LR/seed/update budget/normalization/validation/guard. Training-only weights in [0.25,4], no inverse-std infinity, metadata routing or hyperparameter search. Historical validation objective stays unchanged and excludes the auxiliary term. No bitrate/parameter change, test access or promotion.")
    if args.fold1_channel_confirmation:
        origin = json.loads((prior / "protocol.json").read_text())
        choice = json.loads((prior / "selection.json").read_text())["channel_tail"]
        objective = ROOT / "finetuning/channel_objective.py"
        if (not (prior / "full_validation/COMPLETE").is_file() or choice["guarded_epoch"] == 0
                or origin["channel_mse_weight"] != .05 or origin["channel_weight_cap"] != 4.
                or origin["code_sha256"][str(objective)] != digest(objective)):
            raise ValueError("Fixed positive fold-0 method/source evidence required")
        smoke_state = json.loads((smoke / "summary.json").read_text())
        if smoke_state["objective_sha256"] != digest(objective) or smoke_state["seed"] != seed:
            raise ValueError("Fold-1 smoke objective/seed mismatch")
        protocol.update(kind="guarded_channel_mse_fold_confirmation_v1", arms=["full", "channel_tail"],
                        channel_mse_weights={"full": 0., "channel_tail": .05}, channel_weight_cap=4.,
                        origin_protocol_sha256=digest(prior / "protocol.json"),
                        origin_selection_sha256=digest(prior / "selection.json"),
                        note="Fixed-weight/cap confirmation across configurations and seed; both arms fresh AdamW from fold-1-only source epoch24, 32,768 updates each. Fold-0 weights never transferred. Same within-fold escape-v1 reference as historical fold-1 source; contrast within fold, not absolute pooling with robust fold0 scores. Full validation every epoch, fixed-baseline config/correlation/channel-p90 guard. No auxiliary retuning, extra epochs, test access or final promotion; reused development groups, not independent acquisition evidence.")
    if args.resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Resume protocol/source/code mismatch")
        if (output / "COMPLETE").is_file():
            print(f"[guarded decoder] already complete: {output}")
            return
    elif output.exists():
        raise FileExistsError("New directory or --resume required")
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    require_cuda()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 6 * 1024 ** 3:
        raise RuntimeError("6 GiB free required; no automatic deletions")
    lock = (output.parent / f".{output.name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            output.mkdir()
            atomic_json(output / "protocol.json", protocol)
        original = torch.load(source, map_location="cpu", weights_only=False)
        validate_training_source(original, samples, normalization)
        original_weights = original["model_state_dict"]
        baseline_payload = weights_only(original)
        baseline_payload["epoch"] = 0
        baseline_payload["source_epoch"] = 24
        if not (output / "baseline.pt").exists():
            save_checkpoint(baseline_payload, output / "baseline.pt")
        del baseline_payload, original
        selections = {}
        for arm in protocol["arms"]:
            trainable = "full" if arm == "channel_tail" else arm
            channel_mse_weight = protocol.get("channel_mse_weights", {}).get(arm, protocol.get("channel_mse_weight", 0.))
            target = output / arm
            target.mkdir(exist_ok=True)
            snapshots = target / "snapshots"
            snapshots.mkdir(exist_ok=True)
            last = target / "last.pt"
            completed = 0
            if last.exists():
                saved = torch.load(last, map_location="cpu", weights_only=False)
                completed = int(saved["epoch"])
                state = saved["finetune"]
                if (not 0 <= completed <= 8 or state["validation_hash"] != validation_hash
                        or state["learning_rate"] != 3e-6 or state["trainable"] != trainable
                        or state.get("channel_mse_weight", 0.) != channel_mse_weight
                        or state["normalization"] != normalization):
                    raise ValueError("Branch checkpoint policy mismatch")
                trim_uncommitted_history(target, completed)
                for name, key in (("best_macro.pt", "best_macro_epoch"), ("best.pt", "best_epoch")):
                    chosen_epoch = int(state[key])
                    selected_path = target / name
                    selected_epoch = -1
                    if selected_path.exists():
                        selected = torch.load(selected_path, map_location="cpu", weights_only=False)
                        selected_epoch = int(selected["epoch"])
                        del selected
                    if selected_epoch != chosen_epoch:
                        if chosen_epoch != completed:
                            raise ValueError("Selected checkpoint/history mismatch; inspect before recovery")
                        save_checkpoint(saved, selected_path)
                if completed and not (snapshots / f"epoch_{completed:02d}.pt").exists():
                    save_checkpoint(weights_only(saved), snapshots / f"epoch_{completed:02d}.pt")
                del saved
            elif any((target / n).exists() for n in ("metrics.csv", "validation_metrics.csv", "best.pt")):
                raise ValueError("Partial initialization without last.pt: preserve and inspect before recovery")
            for epoch in range(1, 9):
                snapshot = snapshots / f"epoch_{epoch:02d}.pt"
                if epoch <= completed:
                    if not snapshot.exists():
                        raise ValueError("Missing committed epoch snapshot")
                    continue
                command = ["finetuning/finetune_vqvae.py", "--repo-root", args.repo_root, "--checkpoint", str(source),
                           "--samples-dir", str(samples), "--output-dir", str(target), "--epochs", "1",
                           "--steps-per-epoch", "4096", "--learning-rate", "3e-6", "--weight-decay", "1e-5",
                           "--seed", str(seed), "--device", "cuda", "--sampling", "window", "--normalization", normalization,
                           "--latent-geometry", "spatial4-temporal4", "--trainable", trainable, "--physical-validation",
                           "--validation-details", "--evaluate-initial", "--val-samples-per-file", "0",
                           "--codebook-diagnostics", "--channel-diagnostics", "--early-stopping-metric", "macro-rmse",
                           "--early-stopping-patience", "9", "--early-stopping-min-delta", "1e-4"]
                if channel_mse_weight:
                    command += ["--channel-mse-weight", str(channel_mse_weight)]
                if last.exists():
                    command += ["--resume", str(last)]
                run(command, output / f"{arm}.log", resume=(output / f"{arm}.log").exists(), quiet=True)
                saved = torch.load(last, map_location="cpu", weights_only=False)
                if saved["epoch"] != epoch or saved["finetune"]["validation_hash"] != validation_hash:
                    raise ValueError("Unexpected completed epoch/validation")
                if arm == "decoder" and any(not torch.equal(value, original_weights[k])
                                            for k, value in saved["model_state_dict"].items() if not k.startswith("decoder.")):
                    raise ValueError("Decoder-only branch changed frozen encoder/quantizer")
                save_checkpoint(weights_only(saved), snapshot)
                del saved
                verify_phase0(baseline, rows(target / "validation_groups.csv"))
            history = rows(target / "validation_groups.csv")
            phase0 = verify_phase0(baseline, history)
            decisions = []
            best_epoch, best_score = 0, assess(phase0, phase0)["macro_relative_rmse"]
            for epoch in range(1, 9):
                decision = {"epoch": epoch, **assess(phase0, [r for r in history if int(r["epoch"]) == epoch])}
                decisions.append(decision)
                if decision["eligible"] and decision["macro_relative_rmse"] < best_score:
                    best_epoch, best_score = epoch, decision["macro_relative_rmse"]
            chosen = output / "baseline.pt" if best_epoch == 0 else snapshots / f"epoch_{best_epoch:02d}.pt"
            guarded = target / "best_guarded.pt"
            if not guarded.exists():
                temporary = guarded.with_suffix(".pt.tmp")
                shutil.copy2(chosen, temporary)
                temporary.replace(guarded)
            elif digest(guarded) != digest(chosen):
                raise ValueError("Guarded checkpoint disagrees with frozen decisions")
            selections[arm] = {"guarded_epoch": best_epoch, "guarded_macro_rmse": best_score, "decisions": decisions}
            atomic_json(output / "selection.json", selections)
        full = output / "full_validation"
        command = ["finetuning/evaluate_physical_candidates.py", "--repo-root", args.repo_root, "--fold", str(fold), "--output-dir", str(full)]
        for arm in protocol["arms"]:
            for label, name in (("macro", "best_macro.pt"), ("guarded", "best_guarded.pt")):
                command += ["--candidate", f"{arm}_{label}={normalization}={output / arm / name}"]
        if full.exists():
            command += ["--resume"]
        run(command, output / "full_validation.log", resume=(output / "full_validation.log").exists(), quiet=True)
        (output / "COMPLETE").write_text("Bounded full/decoder pilot and frozen final measurements complete; exploratory only.\n")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
