"""Two-GPU frozen-anchor H2 preflight; no parameter updates."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("Anchor preflight requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    children, logs = [], []
    for gpu, label, filename in ((0, "compact", "compact_transfer_fold0_weights.pt"),
                                  (1, "dense", "dense_transfer_fold0_weights_p93.pt")):
        log = (args.output / f"{label}.log").open("w")
        logs.append(log)
        command = [sys.executable, "-m", "finetuning.run_anchor_innovation_probe",
                   "--corpus", "data/finetune/spatial_band_v1_fold0_20261024",
                   "--calibration", "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json",
                   "--checkpoint", f"results/remote_setup/{filename}", "--vendor", "vendor/smartgrids",
                   "--output", str(args.output / label)]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        children.append((gpu, label, process, command))
    (args.output / "LAUNCH.json").write_text(json.dumps({
        "parameter_updates": 0, "children": [
            {"gpu": gpu, "label": label, "pid": process.pid, "command": command}
            for gpu, label, process, command in children]}, indent=2) + "\n")
    results = [{"label": label, "gpu": gpu, "pid": process.pid, "returncode": process.wait()}
               for gpu, label, process, _ in children]
    for log in logs:
        log.close()
    (args.output / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r["returncode"] != 0 for r in results):
        raise RuntimeError("H2 preflight stopped; inspect preserved logs")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"output": str(args.output), "complete": True, "parameter_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
