"""Choose exactly one GPU just before launching a report-111 training arm."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from tools.choose_feasibility_training_gpu import GPU_QUERY, PROCESS_QUERY, available, local_inventory, snapshot
from tools.run_feasibility_context_remote import mkdirs, upload
from tools.sync_research_remote import REMOTE, command, connect


REPORT = Path("progress_reports/archive/feasibility_protocols_20261118/111-FEBUS_TRAIN_INFERENCE_CONTEXT_ALIGNMENT.md")
PROTOCOL = Path("config/feasibility_context_alignment_v1.json")
RUNNER = "finetuning.run_feasibility_context_alignment"
CLOCK = Path("results/experiments/feasibility_clock_envelopes_20261118/results.json")
FREEZE = Path("results/experiments/feasibility_definition_20261117/metadata_freeze.json")
VENDOR = Path("results/experiments/feasibility_context_replay_20261117/code_snapshot/vendor/smartgrids")
SOURCES = tuple(map(Path, (
    "finetuning/run_feasibility_context_alignment.py",
    "finetuning/run_feasibility_full_coverage.py",
    "finetuning/run_feasibility_context.py",
    "finetuning/feasibility_metrics.py",
    "finetuning/feasibility_context.py",
    "tools/freeze_feasibility_metadata.py",
    "config/feasibility_protocol_v1.json",
    "compression/partition_bundle.py",
    "compression/full_tile_vq_stream.py",
    "compression/alphabet_stream.py",
    "finetuning/streaming_physical_summary.py",
    "tests/test_feasibility_context_alignment.py",
)))


def retrieve_metadata(sftp, remote_path, local_path):
    """Copy reports/logs, leaving checkpoints and waveform-bearing frames remote."""
    if local_path.exists():
        raise ValueError("Local result directory already exists")
    local_path.mkdir(parents=True)
    for entry in sftp.listdir_attr(remote_path):
        name = entry.filename
        if name in ("A", "B", "mature", "endpoint"):
            retrieve_metadata(sftp, f"{remote_path}/{name}", local_path / name)
            continue
        if Path(name).name != name or not (name.endswith((".json", ".jsonl", ".csv"))
                                           or name in ("COMPLETE", "BUDGET_STOP")):
            continue
        with sftp.open(f"{remote_path}/{name}", "rb") as remote, (local_path / name).open("xb") as local:
            local.write(remote.read())


def archive_local(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("part8", "full", "mature"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output
    if (out.is_absolute() or ".." in out.parts or not str(out).startswith("results/experiments/")
            or out.exists() or args.min_free_mib <= 0):
        raise ValueError("Use a new repository-relative experiment path and positive memory floor")
    stamp = time.time_ns()
    audit_path = Path(f"results/remote_setup/feasibility_alignment_device_{stamp}.json")
    arm = {"part8": "A", "full": "B", "mature": "mature"}[args.arm]
    run_args = ["-m", RUNNER, "--arm", arm, "--output", str(out), "--protocol",
                str(PROTOCOL), "--device", "cuda:0", "--max-seconds", "7200"]
    if args.smoke:
        run_args.append("--smoke")
    audit = {"timestamp_ns": stamp, "arm": args.arm, "smoke": args.smoke,
             "output": str(out), "memory_floor_mib": args.min_free_mib,
             "remote_inventory": None, "selected": None, "command": run_args,
             "inventory_is_not_reservation": True}
    client = None
    try:
        try:
            client = connect()
            audit["remote_inventory"] = snapshot(command(client, GPU_QUERY), command(client, PROCESS_QUERY))
        except Exception as error:
            audit["remote_check_error"] = type(error).__name__
        remote = available(audit["remote_inventory"] or [], args.min_free_mib)
        if remote:
            backup = f"{REMOTE}/results/remote_setup/feasibility_alignment_sync_{stamp}"
            command(client, "test ! -e " + shlex.quote(str(Path(REMOTE) / out)))
            with client.open_sftp() as sftp:
                mkdirs(sftp, backup)
                audit["snapshot_backup"] = backup
                uploads = [upload(sftp, path, backup) for path in
                           (*SOURCES, CLOCK, FREEZE, PROTOCOL, REPORT)]
                audit["updated"] = [r for r in uploads if r is not None]
                # Recheck the selected UUID immediately before launch; no
                # implicit use of another device if the machine got busy.
                latest = snapshot(command(client, GPU_QUERY), command(client, PROCESS_QUERY))
                fresh = available([d for d in latest if d["uuid"] == remote["uuid"]], args.min_free_mib)
                audit["remote_launch_inventory"] = latest
                if fresh:
                    audit["selected"] = {"host": "aumo-gpu-001", **fresh}
                    archive_local(audit_path, audit)
                    with sftp.open(backup + "/device.json", "xb") as stream:
                        stream.write(audit_path.read_bytes())
                    invocation = (
                        f"cd {shlex.quote(REMOTE)} && "
                        f"CUDA_VISIBLE_DEVICES={shlex.quote(fresh['uuid'])} "
                        "CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                        + shlex.join([".venv/bin/python", *run_args]))
                    # Combine stdout/stderr so the foreground SSH channel always drains.
                    stdin, stdout, stderr = client.exec_command(invocation + " 2>&1")
                    stdin.close()
                    for line in stdout:
                        print(line.rstrip(), flush=True)
                    status = stdout.channel.recv_exit_status()
                    try:
                        retrieve_metadata(sftp, str(Path(REMOTE) / out), out)
                    except FileNotFoundError:
                        if not status:
                            raise
                    if status:
                        raise SystemExit(status)
                    print(f"Completed on {REMOTE}/{out}; device record {audit_path}", flush=True)
                    return
        local, desktop = local_inventory()
        audit["local_inventory"] = local
        audit["local_desktop_contexts"] = desktop
        selected = next((d for d in local if "RTX 4060" in d["name"]
                         and not d["has_compute_process"] and d["free_mib"] >= args.min_free_mib
                         and d["utilization_percent"] <= 10), None)
        if selected:
            audit["selected"] = {"host": "local", **selected}
        archive_local(audit_path, audit)
        if not selected:
            raise RuntimeError(f"No eligible GPU. See {audit_path}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selected["uuid"], CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   PYTHONPATH=str(VENDOR.resolve()))
        print(f"Selected RTX 4060; device record {audit_path}", flush=True)
        subprocess.run([sys.executable, *run_args], check=True, env=env)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
