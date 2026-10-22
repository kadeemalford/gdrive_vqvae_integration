#!/usr/bin/env python3
"""Explicitly scoped pruning of abandoned checkpoints and regenerable NPZ copies.

Default is dry-run. Raw data, fold_1, current temporal-density checkpoints and
selected upstream baselines are never candidates. Audit is persisted before unlink.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRUNED_DATASETS = ("fold_0", "fold_2", "fold_3", "domain_doped_holdout")
ABANDONED_RUNS = ("sampling_pilot_20261005", "spectral_pilot_20261007", "capacity_probe_20261007",
                  "shared_capacity_probe_20261007", "numeric_adaptation_pilot_20261007")
CHECKPOINT_NAMES = {"best.pt", "best_macro.pt", "last.pt", "probe_best.pt", "probe_last.pt",
                    "probe_best_train.pt", "probe_best_val.pt"}


def targets(root):
    planned = {}
    experiments = root / "results/experiments"
    def add(path, kind, group):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink artifact: {path}")
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Artifact escapes repository")
            planned[path] = (kind, group)
    for name in ABANDONED_RUNS:
        folder = experiments / name
        if folder.exists() and not (folder / "COMPLETE").is_file():
            raise ValueError(f"Experiment not complete: {folder}")
        for path in folder.rglob("*.pt"):
            if path.name in CHECKPOINT_NAMES:
                add(path, "checkpoint", str(folder.relative_to(root)))
    sampling = experiments / "sampling_full_20261005"
    if sampling.exists() and not (sampling / "COMPLETE").is_file():
        raise ValueError("Full sampling comparison must be complete")
    for fold in range(4):
        for name in ("best.pt", "last.pt"):
            add(sampling / f"fold_{fold}/family-group/{name}", "checkpoint", str(sampling.relative_to(root)))
        add(sampling / f"fold_{fold}/window/last.pt", "checkpoint", str(sampling.relative_to(root)))
    for name, branch, files in (
        ("preservation_long_20261007", "decoder", ("best.pt", "best_macro.pt", "last.pt")),
        ("channel_representation_pilot_20261008", "global", ("best.pt", "last.pt")),
        ("channel_representation_pilot_20261008", "channel", ("best.pt", "last.pt")),
    ):
        folder = experiments / name
        if folder.exists() and not (folder / "COMPLETE").is_file():
            raise ValueError(f"Upstream comparison not complete: {folder}")
        for file in files:
            add(folder / branch / file, "checkpoint", str(folder.relative_to(root)))
    for fold in range(4):
        add(root / f"results/fine_tune_results/fold_{fold}/last.pt", "checkpoint", "results/fine_tune_results")
    for name in PRUNED_DATASETS:
        folder = root / "data/finetune" / name
        if folder.is_symlink():
            raise ValueError("Prepared directory may not be a symlink")
        manifest = folder / "samples_manifest.csv"
        if not manifest.is_file():
            continue
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            raw = root / "data/finetune/raw" / f"{row['drive_id']}.h5"
            if not raw.resolve().is_relative_to((root / "data/finetune/raw").resolve()) or not raw.is_file() or raw.stat().st_size == 0:
                raise ValueError(f"Cannot regenerate sample without raw source: {raw}")
            path = root / row["sample_path"]
            if not path.resolve().is_relative_to(folder.resolve()) or path.suffix != ".npz":
                raise ValueError("Prepared manifest points outside its prunable directory")
            add(path, "prepared_npz", str(folder.relative_to(root)))
    return [(path, *planned[path]) for path in sorted(planned)]


def assert_no_repo_jobs(root):
    names = ("finetune_vqvae.py", "run_sampling_experiment.py", "run_preservation_experiment.py",
             "run_channel_experiment.py", "run_temporal_density_experiment.py", "evaluate_physical_candidates.py",
             "prepare_finetune_samples.py", "restore_pruned_samples.py")
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            command = (directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            cwd = (directory / "cwd").resolve()
        except (OSError, RuntimeError):
            continue
        if (cwd == root.resolve() or str(root.resolve()) in command) and any(name in command for name in names):
            raise RuntimeError(f"Repository training/preparation/evaluation job active: PID {directory.name}; postpone pruning")


def prune(root, audit):
    plan = targets(root)
    if audit.exists():
        raise FileExistsError("Use a new audit directory")
    audit.mkdir(parents=True)
    free_before = shutil.disk_usage(root).free
    counts, reclaimed, groups = Counter(), 0, {}
    journal = audit / "removed_files.jsonl"
    # Inventory all original records before deleting anything. This also makes
    # recovery possible if the process is interrupted halfway through cleanup.
    with journal.open("x") as log:
        for path, kind, group in plan:
            state = path.stat()
            row = {"path": str(path.relative_to(root)), "kind": kind, "group": group,
                   "bytes": state.st_size, "allocated_bytes": state.st_blocks * 512,
                   "inode": state.st_ino, "mtime_ns": state.st_mtime_ns, "links_before_unlink": state.st_nlink}
            if kind == "prepared_npz":
                with zipfile.ZipFile(path) as archive:
                    row["zip_records"] = {entry.filename: {"crc32": entry.CRC, "uncompressed_bytes": entry.file_size}
                                          for entry in archive.infolist() if not entry.is_dir()}
                if set(row["zip_records"]) != {"data.npy", "metadata_json.npy"}:
                    raise ValueError("Unexpected sample archive; refusing deletion")
            log.write(json.dumps(row) + "\n")
        log.flush()
        os.fsync(log.fileno())
    assert_no_repo_jobs(root)
    # Mark datasets incomplete BEFORE unlinking, so interrupted cleanup cannot
    # leave a partial dataset looking ready for training.
    for name in PRUNED_DATASETS:
        folder = root / "data/finetune" / name
        manifest = folder / "samples_manifest.csv"
        if manifest.is_file():
            marker = folder / "PRUNED_ARRAYS.json"
            previous = json.loads(marker.read_text()) if marker.is_file() else {}
            old_logs = previous.get("audit_logs", [previous["audit_log"]] if "audit_log" in previous else [])
            marker.write_text(json.dumps({"status": "pruning", "audit_log": str(journal.resolve()),
                "audit_logs": list(dict.fromkeys([*old_logs, str(journal.resolve())])),
                "recovery": f".venv/bin/python finetuning/restore_pruned_samples.py --samples-dir data/finetune/{name}"}, indent=2) + "\n")
    with journal.open() as inventory, (audit / "applied_actions.jsonl").open("x") as actions:
        for line in inventory:
            row = json.loads(line)
            path, kind, group = root / row["path"], row["kind"], row["group"]
            current = path.stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (row["inode"], row["bytes"], row["mtime_ns"]):
                raise RuntimeError("Artifact changed during pruning")
            path.unlink()
            freed = current.st_blocks * 512 if current.st_nlink == 1 else 0
            reclaimed += freed
            counts[kind] += 1
            groups[group] = groups.get(group, 0) + 1
            actions.write(json.dumps({"path": row["path"], "removed": True, "reclaimed_file_blocks_bytes": freed}) + "\n")
            actions.flush()
        os.fsync(actions.fileno())
    for name in PRUNED_DATASETS:
        folder = root / "data/finetune" / name
        manifest = folder / "samples_manifest.csv"
        if not manifest.is_file():
            continue
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        missing = sum(not (root / row["sample_path"]).is_file() for row in rows)
        marker = folder / "PRUNED_ARRAYS.json"
        previous = json.loads(marker.read_text()) if marker.is_file() else {}
        old_logs = previous.get("audit_logs", [previous["audit_log"]] if "audit_log" in previous else [])
        marker.write_text(json.dumps({"status": "pruned" if missing else "restored",
            "missing_manifest_samples": missing, "expected_manifest_samples": len(rows),
            "audit_log": str((audit / "removed_files.jsonl").resolve()),
            "audit_logs": list(dict.fromkeys([*old_logs, str(journal.resolve())])),
            "recovery": f".venv/bin/python finetuning/restore_pruned_samples.py --samples-dir data/finetune/{name}"}, indent=2) + "\n")
    for group in groups:
        if group.startswith("results/"):
            (root / group / "PRUNED_CHECKPOINTS.json").write_text(json.dumps({"removed_paths": groups[group],
                "audit_log": str((audit / "removed_files.jsonl").resolve()),
                "note": "Execution metrics/reports preserved; removed checkpoints are unavailable for old resume/inference. Retained selected upstream baselines remain usable."}, indent=2) + "\n")
    summary = {"removed_files": dict(counts), "removed_by_group": groups, "reclaimed_file_blocks_bytes": reclaimed,
               "free_bytes_before": free_before, "free_bytes_after": shutil.disk_usage(root).free,
               "preserved": ["all raw HDF5", "all manifests/metrics/reports", "entire prepared fold_1",
                             "all temporal-density checkpoints", "all models/ base checkpoints",
                             "uniform best.pt per fold", "preservation full checkpoints", "global/channel representation best_macro.pt",
                             "original fine-tuning best.pt per fold"]}
    (audit / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args()
    assert_no_repo_jobs(ROOT)
    if args.apply:
        if args.audit_dir is None:
            raise ValueError("Apply requires a new audit directory")
        print(json.dumps(prune(ROOT, args.audit_dir.resolve()), indent=2))
    else:
        plan = targets(ROOT)
        print(json.dumps({"dry_run": True, "candidate_counts": dict(Counter(kind for _, kind, _ in plan)),
                          "logical_candidate_bytes": sum(path.stat().st_size for path, _, _ in plan),
                          "groups": dict(Counter(group for _, _, group in plan))}, indent=2))


if __name__ == "__main__":
    main()
