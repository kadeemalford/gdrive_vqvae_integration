"""Smoke both arms, check pairing, then run the registered augmentation contrast."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def verify_pair(output):
    results = [json.loads((output / method / "results.json").read_text()) for method in ("real", "mixed")]
    if any(results[0][key] != results[1][key] for key in
           ("updates", "cells", "schedule_sha256", "initial_nrmse_star", "initial_bpc")):
        raise AssertionError("Real/mixed schedule, initial metric or cell exposure differs")
    left = output / "real/check_000000"
    right = output / "mixed/check_000000"
    frames = list(left.glob("*.frame"))
    if not frames or {p.name for p in frames} != {p.name for p in right.glob("*.frame")}:
        raise AssertionError("Missing initial physical frames in paired comparison")
    for frame in frames:
        if frame.read_bytes() != (right / frame.name).read_bytes():
            raise AssertionError("Initial physical stream differs across the paired GPUs")
    if results[1]["augmented_updates"] != results[1]["updates"] // 2:
        raise AssertionError("Exactly half of mixed updates must use composition")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Semisynthetic study requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    for smoke in (True, False):
        phase = args.output / ("smoke" if smoke else "scientific")
        phase.mkdir()

        def run(item):
            gpu, method = item
            command = [sys.executable, "-m", "finetuning.run_semisynthetic_study",
                       "--method", method, "--output", str(phase / method)]
            if smoke:
                command.append("--smoke")
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
            with (phase / f"{method}.log").open("w") as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                (phase / f"LAUNCH_{method}.json").write_text(json.dumps({
                    "gpu": gpu, "pid": process.pid, "command": command}, indent=2) + "\n")
                return {"method": method, "returncode": process.wait()}
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, [(0, "real"), (1, "mixed")]))
        (phase / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
        if any(row["returncode"] for row in results):
            raise RuntimeError("Semisynthetic phase failed; preserve logs, no automatic repeat")
        verify_pair(phase)
        (phase / "COMPLETE").touch()
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
