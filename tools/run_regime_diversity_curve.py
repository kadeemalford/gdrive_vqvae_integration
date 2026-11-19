"""Run registered nested data regimes, two concurrent arms at most."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def verify(output, smoke):
    results = [json.loads((output / str(count) / "results.json").read_text()) for count in (4, 6, 10)]
    reference = results[0]
    for result in results[1:]:
        for key in ("updates", "cells", "family_updates", "initial_nrmse_star", "initial_bpc"):
            if result[key] != reference[key]:
                raise AssertionError(f"Regime curve has unmatched {key}")
    folders = [output / str(count) / "check_000000" for count in (4, 6, 10)]
    frames = list(folders[0].glob("*.frame"))
    if len(frames) != (2 if smoke else 136):
        raise AssertionError("Missing initial physical frames")
    for frame in frames:
        if any((folder / frame.name).read_bytes() != frame.read_bytes() for folder in folders[1:]):
            raise AssertionError("Initial source differs across regimes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Diversity curve requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    for smoke in (True, False):
        phase = args.output / ("smoke" if smoke else "scientific")
        phase.mkdir()

        def run(item):
            gpu, count = item
            command = [sys.executable, "-m", "finetuning.run_regime_diversity",
                       "--groups", str(count), "--output", str(phase / str(count))]
            if smoke:
                command.append("--smoke")
            with (phase / f"{count}.log").open("w") as log:
                process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"},
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                (phase / f"LAUNCH_{count}.json").write_text(json.dumps({
                    "pid": process.pid, "gpu": gpu, "command": command}, indent=2) + "\n")
                return {"groups": count, "returncode": process.wait()}
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, [(0, 4), (1, 6)]))
        if all(row["returncode"] == 0 for row in results):
            results.append(run((0, 10)))
        (phase / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
        if len(results) != 3 or any(row["returncode"] for row in results):
            raise RuntimeError("Regime curve interrupted; preserve artifacts, no automatic repeat")
        verify(phase, smoke)
        (phase / "COMPLETE").touch()
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
