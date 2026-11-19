"""Report a conservative, read-only device recommendation before a training job."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from tools.sync_research_remote import command, connect


GPU_QUERY = (
    "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,"
    "utilization.gpu --format=csv,noheader,nounits"
)
PROCESS_QUERY = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader"
LOCAL_PROCESS_QUERY = "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits"
DESKTOP_EXECUTABLES = {
    "/usr/libexec/gnome-remote-desktop-daemon", "/usr/share/kiro/kiro", "/usr/share/code/code",
}


def snapshot(gpu_text, processes_text):
    devices = []
    busy = set()
    for entry in csv.reader(processes_text.splitlines(), skipinitialspace=True):
        if len(entry) != 2 or not entry[0].startswith("GPU-") or not entry[1].strip().isdigit():
            raise ValueError("Unrecognized compute-process inventory")
        busy.add(entry[0])
    for index, uuid, name, total, free, utilization in csv.reader(
            gpu_text.splitlines(), skipinitialspace=True):
        device = {"index": int(index), "uuid": uuid, "name": name,
                  "total_mib": int(total), "free_mib": int(free),
                  "utilization_percent": int(utilization), "has_compute_process": uuid in busy}
        if not 0 <= device["free_mib"] <= device["total_mib"] or not uuid.startswith("GPU-"):
            raise ValueError("Invalid GPU inventory")
        devices.append(device)
    if not devices or len({item["uuid"] for item in devices}) != len(devices):
        raise ValueError("Empty or duplicate GPU inventory")
    return devices


def available(devices, min_free_mib):
    """Only consider devices with no known compute process and low utilization."""
    eligible = [d for d in devices if not d["has_compute_process"]
                and d["free_mib"] >= min_free_mib and d["utilization_percent"] <= 5]
    return min(eligible, key=lambda d: (-d["free_mib"], d["index"]), default=None)


def local_inventory():
    """Ignore only identified small desktop contexts, never unknown local jobs."""
    gpu = subprocess.run(GPU_QUERY.split(), check=True, text=True, capture_output=True).stdout
    processes = subprocess.run(LOCAL_PROCESS_QUERY.split(), check=True, text=True, capture_output=True).stdout
    busy, desktop = [], []
    for uuid, pid, memory in csv.reader(processes.splitlines(), skipinitialspace=True):
        if not pid.isdigit():
            raise ValueError("Unexpected local GPU process")
        try:
            executable = str(Path(f"/proc/{pid}/exe").resolve(strict=True))
        except OSError:
            executable = None
        if executable in DESKTOP_EXECUTABLES and memory.isdigit() and int(memory) <= 256:
            desktop.append({"uuid": uuid, "pid": int(pid), "executable": executable,
                            "memory_mib": int(memory)})
        else:
            busy.append(f"{uuid},{pid}")
    return snapshot(gpu, "\n".join(busy)), desktop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-free-mib", type=int, required=True)
    args = parser.parse_args()
    if args.min_free_mib <= 0:
        parser.error("--min-free-mib must be positive")
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "min_free_mib": args.min_free_mib, "remote_inventory": None,
              "reservation": False, "selected": None}
    try:
        with connect() as client:
            remote = snapshot(command(client, GPU_QUERY), command(client, PROCESS_QUERY))
            result["remote_inventory"] = remote
            selected = available(remote, args.min_free_mib)
            if selected:
                result["selected"] = {"host": "aumo-gpu-001", **selected}
    except Exception as error:
        # Record the class only: authentication or configuration messages may be sensitive.
        result["remote_check_error"] = type(error).__name__
    if result["selected"] is None:
        local, desktop = local_inventory()
        result["local_inventory"] = local
        result["local_desktop_contexts"] = desktop
        selected = next((d for d in local if "RTX 4060" in d["name"]
                         and d["free_mib"] >= args.min_free_mib
                         and not d["has_compute_process"] and d["utilization_percent"] <= 10), None)
        if selected:
            result["selected"] = {"host": "local", **selected}
    print(json.dumps(result, indent=2))
    if result["selected"] is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
