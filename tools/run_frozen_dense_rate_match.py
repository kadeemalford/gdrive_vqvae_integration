"""Two-GPU inference-only frozen rate curves; preserve every measured point."""
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
        raise RuntimeError("Frozen rate curves require free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)

    def run(item):
        gpu, variant = item
        command = [sys.executable, "-m", "finetuning.frozen_dense_rate_match",
                   "--variant", variant, "--output", str(args.output / variant)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
        with (args.output / f"{variant}.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            (args.output / f"LAUNCH_{variant}.json").write_text(json.dumps({
                "gpu": gpu, "pid": process.pid, "command": command}, indent=2) + "\n")
            return {"variant": variant, "returncode": process.wait()}
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, [(0, "flat"), (1, "product")]))
    (args.output / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(row["returncode"] for row in results):
        raise RuntimeError("Frozen rate diagnostic failed; inspect preserved logs")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "parameter_updates": 0, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
