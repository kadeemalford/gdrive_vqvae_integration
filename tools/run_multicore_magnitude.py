"""Supervise the fixed-budget conditional-magnitude controls on two GPUs."""
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
        raise RuntimeError("Magnitude study requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)

    def run(item):
        gpu, mode = item
        command = [sys.executable, "-m", "finetuning.run_multicore_magnitude",
                   "--mode", mode, "--output", str(args.output / mode)]
        with (args.output / f"{mode}.log").open("w") as log:
            process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"},
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            (args.output / f"LAUNCH_{mode}.json").write_text(json.dumps({
                "pid": process.pid, "gpu": gpu, "command": command}, indent=2) + "\n")
            return {"mode": mode, "returncode": process.wait()}
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, [(0, "aligned"), (1, "shifted")]))
    if all(row["returncode"] == 0 for row in results):
        results.append(run((0, "none")))
    (args.output / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
    if len(results) != 3 or any(row["returncode"] for row in results):
        raise RuntimeError("Magnitude study failed; preserve artifacts, no automatic repeat")
    summaries = {mode: json.loads((args.output / mode / "results.json").read_text())
                 for mode in ("aligned", "shifted", "none")}
    if len({r["schedule_sha256"] for r in summaries.values()}) != 1:
        raise AssertionError("Magnitude training schedules differ")
    first = args.output / "aligned/baseline"
    for frame in first.glob("*.frame"):
        if any(frame.read_bytes() != (args.output / mode / "baseline" / frame.name).read_bytes()
               for mode in ("shifted", "none")):
            raise AssertionError("Magnitude independent baselines differ")
    aligned = summaries["aligned"]
    group_totals = {}
    for row in aligned["records"]:
        group_totals.setdefault(row["group_id"], [0, 0])
        group_totals[row["group_id"]][0] += row["baseline_bytes"]
        group_totals[row["group_id"]][1] += row["candidate_bytes"]
    gate = (aligned["relative_joint_byte_saving"] >= .05
            and all(aligned["candidate_complete_bytes"] < summaries[mode]["candidate_complete_bytes"]
                    for mode in ("none", "shifted"))
            and all(b <= a for a, b in group_totals.values()))
    (args.output / "COMPARISON.json").write_text(json.dumps({
        "opportunity_gate": gate, "aligned_relative_joint_byte_saving": aligned["relative_joint_byte_saving"],
        "complete_bytes": {mode: r["candidate_complete_bytes"] for mode, r in summaries.items()},
        "group_frame_bytes_baseline_candidate": group_totals,
        "automatic_promotion": False, "physical_bits_equal": True}, indent=2) + "\n")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "output": str(args.output), "opportunity_gate": gate}), flush=True)


if __name__ == "__main__":
    main()
