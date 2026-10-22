#!/usr/bin/env python3
"""Conservative disk cleanup: hardlink equivalent completed checkpoint archives.

All ZIP records must match (including optimizer/RNG/metadata), ignoring only the
outer archive directory name. No checkpoints are dropped; active runs/data excluded.
Future writes to shared paths must use atomic replacement, never in-place mutation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    "sampling_pilot_20261005", "sampling_full_20261005", "spectral_pilot_20261007",
    "capacity_probe_20261007", "shared_capacity_probe_20261007", "numeric_adaptation_pilot_20261007",
)


def records(archive):
    entries = {}
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        parts = entry.filename.split("/", 1)
        if len(parts) != 2 or parts[1] in entries:
            raise ValueError("Unexpected or duplicate checkpoint ZIP record")
        entries[parts[1]] = entry
    if "data.pkl" not in entries:
        raise ValueError("Not a PyTorch checkpoint archive")
    return entries


def signature(path):
    with zipfile.ZipFile(path) as archive:
        return tuple(sorted((name, entry.file_size, entry.CRC) for name, entry in records(archive).items()))


def payload_digest(path):
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for name, entry in sorted(records(archive).items()):
            digest.update(json.dumps([name, entry.file_size]).encode() + b"\n")
            with archive.open(entry) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def snapshot(path):
    state = path.stat()
    return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)


def link_equivalent(source, target):
    if source.is_symlink() or target.is_symlink():
        raise ValueError("Symlink checkpoints excluded")
    source_state, target_state = snapshot(source), snapshot(target)
    if source_state[:2] == target_state[:2]:
        return None
    if source_state[0] != target_state[0]:
        raise ValueError("Hardlinks require same filesystem")
    source_digest, target_digest = payload_digest(source), payload_digest(target)
    if source_digest != target_digest:
        raise ValueError("Checkpoint payloads differ; refusing to replace")
    if snapshot(source) != source_state or snapshot(target) != target_state:
        raise RuntimeError("Checkpoint changed during verification")
    target_stat = target.stat()
    reclaimed = target_stat.st_blocks * 512 if target_stat.st_nlink == 1 else 0
    temporary = target.with_name(target.name + ".dedup-" + uuid.uuid4().hex)
    try:
        os.link(source, temporary)
        # Creating the hardlink changes source ctime/nlink, not its payload.
        if snapshot(target) != target_state:
            raise RuntimeError("Target changed before atomic replacement")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not os.path.samefile(source, target):
        raise AssertionError("Hardlink verification failed")
    return {"source": str(source), "target": str(target), "verified_payload_sha256": source_digest,
            "reclaimed_allocated_bytes": reclaimed, "target_previous_bytes": target_state[2],
            "target_previous_inode": target_state[1], "action": "atomic hardlink replacement; both checkpoint names retained"}


def candidates(root):
    groups = defaultdict(list)
    for name in EXPERIMENTS:
        folder = root / name
        if not (folder / "COMPLETE").is_file():
            continue
        for path in sorted(folder.rglob("*.pt")):
            if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                continue
            groups[signature(path)].append(path)
    return [paths for paths in groups.values() if len(paths) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, default=ROOT / "results/experiments")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--log", type=Path, help="New JSONL audit log, required for apply")
    args = parser.parse_args()
    if args.apply and not args.log:
        raise ValueError("Apply requires a new audit log")
    groups = candidates(args.experiments_dir.resolve())
    actions, reclaimed = 0, 0
    log = None
    try:
        if args.apply:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = args.log.open("x")
        for paths in groups:
            source = paths[0]  # Sorted best checkpoints precede last where equivalent.
            for target in paths[1:]:
                if os.path.samefile(source, target):
                    continue
                if args.apply:
                    result = link_equivalent(source, target)
                    if result:
                        log.write(json.dumps(result) + "\n")
                        log.flush()
                        os.fsync(log.fileno())
                        reclaimed += result["reclaimed_allocated_bytes"]
                        actions += 1
                        print(json.dumps(result), flush=True)
                else:
                    print(json.dumps({"source": str(source), "target": str(target), "candidate_only": True}))
        print(json.dumps({"applied": args.apply, "hardlinked_checkpoints": actions, "reclaimed_allocated_bytes": reclaimed,
                          "scope": list(EXPERIMENTS), "no_data_or_active_run_changes": True}), flush=True)
    finally:
        if log:
            log.close()


if __name__ == "__main__":
    main()
