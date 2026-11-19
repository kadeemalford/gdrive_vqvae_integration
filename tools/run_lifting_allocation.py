"""Remote inference-only frozen-network allocation on two free GPUs."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("Allocation requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    children, logs = [], []
    for gpu, arm in enumerate(("fixed", "learned")):
        log = (args.output / f"{arm}.log").open("w")
        logs.append(log)
        command = [sys.executable, "-m", "finetuning.fit_lifting_allocation",
                   "--corpus", "data/finetune/spatial_band_v1_fold0_20261024",
                   "--calibration", "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json",
                   "--study", f"results/experiments/lifting_p2_dense_fold0_20261114/{arm}",
                   "--output", str(args.output / arm)]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, stdin=subprocess.DEVNULL)
        children.append((gpu, arm, process, command))
    (args.output / "LAUNCH.json").write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "parameter_updates": 0,
        "children": [{"gpu": gpu, "arm": arm, "pid": process.pid, "command": command}
                     for gpu, arm, process, command in children]}, indent=2) + "\n")
    results = [{"arm": arm, "gpu": gpu, "pid": process.pid, "returncode": process.wait()}
               for gpu, arm, process, _ in children]
    for log in logs:
        log.close()
    (args.output / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r["returncode"] != 0 for r in results):
        raise RuntimeError("Allocation stopped; inspect preserved logs")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "output": str(args.output), "parameter_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
