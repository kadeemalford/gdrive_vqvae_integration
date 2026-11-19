"""Two-GPU frozen diagnostic validation; no training, reselection or final-test access."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    integrity = Path("results/experiments/development_validation_integrity_20261115")
    if not json.loads((integrity / "REMOTE_VERIFIED.json").read_text())["passed"]:
        raise RuntimeError("Frozen validation bytes/weights must be verified first")
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("Diagnostic validation requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = [(0, ["mature_reference", "independent", "encoder_local"]),
                (1, ["h2_joint", "encoder_full"])]
    (args.output / "SCHEDULE.json").write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "parameter_updates": 0,
        "final_test_waveform_access": False, "schedule": schedule,
        "selection": "Previously registered frozen selected checkpoints only",
        "scope": "Historically consumed development validation, not P3 or final acceptance"}, indent=2) + "\n")

    def run(item):
        gpu, names = item
        results = []
        for name in names:
            command = [sys.executable, "-m", "finetuning.development_validation",
                       "--candidate", name, "--vendor", "vendor/smartgrids",
                       "--integrity", str(integrity), "--output", str(args.output / name)]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
            with (args.output / f"{name}.log").open("w") as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, stdin=subprocess.DEVNULL)
                (args.output / f"LAUNCH_{name}.json").write_text(json.dumps({
                    "candidate": name, "gpu": gpu, "pid": process.pid, "command": command,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
                code = process.wait()
            results.append({"candidate": name, "gpu": gpu, "returncode": code})
            if code:
                break
        return results
    with ThreadPoolExecutor(2) as pool:
        results = [result for group in pool.map(run, schedule) for result in group]
    (args.output / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
    if len(results) != 5 or any(r["returncode"] for r in results):
        raise RuntimeError("Development validation interrupted; no window was silently dropped")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"output": str(args.output), "complete": True,
                      "parameter_updates": 0, "final_test_waveform_access": False}), flush=True)


if __name__ == "__main__":
    main()
