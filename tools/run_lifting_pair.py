"""Remote-only paired launch with bounded budget, no automatic extension."""
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
    parser.add_argument("--target-bpc", type=float, required=True)
    parser.add_argument("--updates", type=int, default=32768)
    args = parser.parse_args()
    integration = Path("results/experiments/lifting_integration_v3_fold0_20261114/INTEGRATION.json")
    if not json.loads(integration.read_text())["passed"]:
        raise RuntimeError("Integration must pass before scientific launch")
    if args.updates < 1 or args.updates > 32768:
        raise ValueError("This launcher permits at most the registered initial budget")
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("GPUs have compute processes; refusing to overlap existing work")
    args.output.mkdir(parents=True, exist_ok=False)
    children = []
    logs = []
    for gpu, arm in enumerate(("fixed", "learned")):
        log = (args.output / f"{arm}.log").open("w")
        logs.append(log)
        command = [sys.executable, "-m", "finetuning.run_lifting_study",
                   "--corpus", "data/finetune/spatial_band_v1_fold0_20261024",
                   "--calibration", "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json",
                   "--output", str(args.output / arm), "--arm", arm,
                   "--target-bpc", str(args.target_bpc), "--updates", str(args.updates),
                   "--check-every", "2048"]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        children.append((arm, gpu, process, command))
    launch = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "target_proxy_bpc": args.target_bpc, "updates_per_arm": args.updates,
              "children": [{"arm": arm, "gpu": gpu, "pid": process.pid, "command": command}
                           for arm, gpu, process, command in children],
              "automatic_extension": False, "next_rate_automatic": False}
    (args.output / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n")
    results = [{"arm": arm, "gpu": gpu, "pid": process.pid, "returncode": process.wait()}
               for arm, gpu, process, _ in children]
    for log in logs:
        log.close()
    passed = all(result["returncode"] == 0 for result in results)
    (args.output / "EXIT.json").write_text(json.dumps({"success": passed, "results": results}, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Paired run stopped; inspect arm logs, preserve all artifacts")
    (args.output / "PAIR_BUDGET_COMPLETE").touch()
    print(json.dumps({"output": str(args.output), "complete": True, "results": results}), flush=True)


if __name__ == "__main__":
    main()
