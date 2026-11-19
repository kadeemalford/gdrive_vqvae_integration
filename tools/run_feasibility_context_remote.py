"""Synchronize a preserved F1 snapshot and run on exactly one remote GPU."""
import argparse
import hashlib
import json
import shlex
import stat
import time
from pathlib import Path

from tools.sync_research_remote import REMOTE, command, connect


def digest(content):
    return hashlib.sha256(content).hexdigest()


def mkdirs(sftp, path):
    """Create missing parents without masking non-directory errors."""
    try:
        info = sftp.stat(path)
    except FileNotFoundError:
        parent = str(Path(path).parent)
        if parent != path:
            mkdirs(sftp, parent)
        sftp.mkdir(path)
    else:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Remote parent is not a directory: {path}")


def upload(sftp, relative, backup):
    """Preserve differing remote content and verify the new snapshot."""
    content = Path(relative).read_bytes()
    target = f"{REMOTE}/{relative}"
    try:
        with sftp.open(target, "rb") as stream:
            previous = stream.read()
    except FileNotFoundError:
        previous = None
    if previous == content:
        return None
    if previous is not None:
        saved = f"{backup}/{relative}"
        mkdirs(sftp, str(Path(saved).parent))
        with sftp.open(saved, "xb") as stream:
            stream.write(previous)
    mkdirs(sftp, str(Path(target).parent))
    temporary = target + ".feasibility_upload"
    with sftp.open(temporary, "wb") as stream:
        stream.write(content)
    sftp.posix_rename(temporary, target)
    with sftp.open(target, "rb") as stream:
        if digest(stream.read()) != digest(content):
            raise AssertionError(f"Remote snapshot differs: {relative}")
    return {"path": str(relative), "sha256": digest(content),
            "previous_sha256": digest(previous) if previous is not None else None}


def retrieve(sftp, remote, local):
    """Copy only the new diagnostic output; refuse divergent local results."""
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote):
        path = f"{remote}/{entry.filename}"
        target = local / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            retrieve(sftp, path, target)
        elif stat.S_ISREG(entry.st_mode):
            with sftp.open(path, "rb") as stream:
                content = stream.read()
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite local result: {target}")
            if not target.exists():
                target.write_bytes(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path(
        "results/experiments/feasibility_definition_20261117/metadata_freeze.json"))
    args = parser.parse_args()
    if args.output.is_absolute() or ".." in args.output.parts or not str(args.output).startswith("results/experiments/"):
        raise ValueError("Expected a new repository-relative experiment destination")
    if args.output.exists():
        raise ValueError("The local experiment destination must be new")
    files = {Path("config/feasibility_protocol_v1.json"), args.metadata, args.weights_manifest,
             Path("tools/freeze_feasibility_metadata.py")}
    for folder in ("finetuning", "compression"):
        files.update(path for path in Path(folder).glob("*.py")
                     if "feasibility" not in path.name or path.name in {
                         "feasibility_context.py", "feasibility_metrics.py",
                         "run_feasibility_context.py"})
    files.update(Path("vendor/smartgrids/vqvae").glob("*.py"))
    # No prepared or raw signal files are uploaded or downloaded.
    for relative in (
        "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json",
        "results/experiments/prospective_p0_v2_fold0_20261114/protocol.json",
        "results/experiments/train_clock_copy_audit_fold0_20261115/results.json",
        "results/experiments/physical_conditioning_20261115/scientific/constant/protocol.json",
        "results/experiments/regime_diversity_curve_20261115/scientific/10/protocol.json",
    ):
        files.add(Path(relative))
    client = connect()
    sftp = client.open_sftp()
    backup = f"{REMOTE}/results/remote_setup/feasibility_sync_{time.time_ns()}"
    try:
        command(client, f"test ! -e {shlex.quote(str(Path(REMOTE) / args.output))}")
        # Do not launch into a GPU already used by another task.
        processes = command(client, "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader")
        gpu = command(client, "nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader").strip()
        if any(gpu in line for line in processes.splitlines()):
            raise RuntimeError("GPU 0 has an active compute process; no job launched")
        records = [record for path in sorted(files)
                   if (record := upload(sftp, path, backup)) is not None]
        mkdirs(sftp, backup)
        audit = {"kind": "feasibility-f1-preserved-sync-v1", "backup": backup,
                 "updated": records, "gpu_visible_devices": "0", "parameter_updates": 0,
                 "signal_transfer": False}
        encoded = (json.dumps(audit, indent=2) + "\n").encode()
        with sftp.open(backup + "/sync.json", "wb") as stream:
            stream.write(encoded)
        destination = Path("results/remote_setup") / (Path(backup).name + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        invocation = ["--output", str(args.output), "--weights-manifest", str(args.weights_manifest),
                      "--metadata", str(args.metadata), "--device", "cuda:0"]
        script = (
            f"cd {shlex.quote(REMOTE)} && "
            "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            ".venv/bin/python -m finetuning.run_feasibility_context "
            + shlex.join(invocation))
        # This blocks on the one SSH job, not a polling supervisor.
        stdin, stdout, stderr = client.exec_command(script)
        for line in stdout:
            print(line.rstrip(), flush=True)
        errors = stderr.read().decode()
        status = stdout.channel.recv_exit_status()
        stdin.close()
        stdout.close()
        stderr.close()
        retrieve(sftp, f"{REMOTE}/{args.output}", args.output)
        if status:
            raise RuntimeError(f"F1 exited {status}; preserved output retrieved. {errors}")
        if errors:
            print(errors, flush=True)
        print(f"F1 results retrieved: {args.output}", flush=True)
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
