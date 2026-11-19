"""Supervise paired causal residual models on two otherwise idle GPUs."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Residual study requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)

    def run(item):
        gpu, method = item
        command = [sys.executable, "-m", "finetuning.neural_bounded_residual",
                   "--method", method, "--output", str(args.output / method)]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
        with (args.output / f"{method}.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            (args.output / f"LAUNCH_{method}.json").write_text(json.dumps({
                "gpu": gpu, "pid": process.pid, "command": command}, indent=2) + "\n")
            return {"method": method, "returncode": process.wait()}
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, [(0, "causal"), (1, "base_only")]))
    (args.output / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(row["returncode"] for row in results):
        raise RuntimeError("Causal study interrupted; preserve logs")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
