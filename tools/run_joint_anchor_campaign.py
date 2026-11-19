"""Bounded autonomous H2 campaign: smoke, pilot, registered conditional extension."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from finetuning.joint_anchor_gate import extension_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("joint", "encoder-context"), default="joint")
    parser.add_argument("--source", type=Path, default=Path("results/remote_setup/dense_transfer_fold0_weights_p93.pt"))
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--seed", type=int, default=2115)
    args = parser.parse_args()
    torch.set_num_threads(2)
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("Joint H2 campaign requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    labels = ("independent", "closed-loop") if args.kind == "joint" else ("local-analysis", "full-observation")
    reference_source = args.reference_source or args.source
    common = [sys.executable, "-m", "finetuning.run_joint_anchor_study",
              "--corpus", "data/finetune/spatial_band_v1_fold0_20261024",
              "--calibration", "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json",
              "--source", str(args.source), "--reference-source", str(reference_source),
              "--seed", str(args.seed), "--vendor", "vendor/smartgrids", "--maximum-updates", "32768"]
    launches = []

    def pair(stage, extra):
        def run(item):
            gpu, mode = item
            container = "smoke_resume" if stage.startswith("smoke_resume") else (
                "smoke" if stage == "smoke" else "arms")
            output = args.output / container / mode
            output.parent.mkdir(exist_ok=True)
            actual_mode = mode if args.kind == "joint" else "closed-loop"
            observation = [] if args.kind == "joint" else ["--encoder-context", "local" if gpu == 0 else "full"]
            command = common + ["--output", str(output), "--mode", actual_mode] + observation + extra
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
            with (args.output / f"{stage}_{mode}.log").open("w") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                record = {"stage": stage, "mode": mode, "gpu": gpu, "pid": process.pid, "command": command}
                launches.append(record)
                # Each process gets its own launch record; no shared-file write race.
                (args.output / f"LAUNCH_{stage}_{mode}.json").write_text(json.dumps(record, indent=2) + "\n")
                return {"mode": mode, "gpu": gpu, "stage": stage, "returncode": process.wait()}
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, list(enumerate(labels))))
        (args.output / f"EXIT_{stage}.json").write_text(json.dumps(results, indent=2) + "\n")
        if any(result["returncode"] != 0 for result in results):
            raise RuntimeError(f"Joint H2 {stage} stopped; preserve artifacts and inspect logs")

    pair("smoke", ["--smoke"])
    pair("smoke_resume1", ["--smoke", "--stop-after", "1"])
    pair("smoke_resume2", ["--smoke", "--resume"])
    resume_checks = []
    for mode in labels:
        direct = torch.load(args.output / "smoke" / mode / "last.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(args.output / "smoke_resume" / mode / "last.pt", map_location="cpu", weights_only=False)
        identical = all(torch.equal(value, resumed["model_state_dict"][key])
                        for key, value in direct["model_state_dict"].items())
        moments_identical = all(torch.equal(value, resumed["optimizer_state_dict"]["state"][key][name])
                                for key, state in direct["optimizer_state_dict"]["state"].items()
                                for name, value in state.items() if isinstance(value, torch.Tensor))
        rng_identical = (direct["python_rng"] == resumed["python_rng"]
                         and torch.equal(direct["torch_rng"], resumed["torch_rng"])
                         and torch.equal(direct["cuda_rng"], resumed["cuda_rng"]))
        resume_checks.append({"mode": mode, "parameters_identical": identical,
                              "optimizer_moments_identical": moments_identical, "rng_identical": rng_identical})
        del direct, resumed
    (args.output / "RESUME_INTEGRATION.json").write_text(json.dumps(resume_checks, indent=2) + "\n")
    if not all(all(r[key] for key in ("parameters_identical", "optimizer_moments_identical", "rng_identical"))
               for r in resume_checks):
        raise RuntimeError("Joint H2 deterministic resume did not pass; no scientific pilot launched")
    pair("pilot", ["--stop-after", "8192"])
    independent = args.output / "arms" / labels[0]
    closed = args.output / "arms" / labels[1]

    def result(folder, update):
        return json.loads((folder / f"check_{update:06d}/results.json").read_text())

    selection = json.loads((closed / "selection.json").read_text())["update"]
    control_selection = json.loads((independent / "selection.json").read_text())["update"]
    history = [json.loads(line) for line in (closed / "history.jsonl").read_text().splitlines()][-64:]
    anchor = float(np.mean([r["anchor_perplexity_mean"] for r in history]))
    innovation_values = [r["innovation_perplexity_mean"] for r in history
                         if r["innovation_perplexity_mean"] is not None]
    innovation = float(np.mean(innovation_values)) if innovation_values else None
    gate = extension_gate(result(closed, 0), result(closed, 6144), result(closed, 8192),
                          result(closed, selection), result(independent, control_selection), anchor, innovation)
    gate.update(pilot_updates_per_arm=8192, maximum_updates_per_arm=32768,
                selected_update=selection, independent_selected_update=control_selection,
                anchor_perplexity=anchor, innovation_perplexity=innovation,
                control_name=labels[0], candidate_name=labels[1], kind=args.kind)
    (args.output / "EXTENSION_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"pilot_complete": True, "extension_gate": gate}), flush=True)
    if gate["eligible"]:
        pair("extension", ["--resume", "--stop-after", "32768"])
    (args.output / "TRAINING_COMPLETE").touch()
    def expanded(item):
        gpu, mode = item
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
        command = [sys.executable, "-m", "finetuning.evaluate_joint_anchor_study",
                   "--study", str(args.output / "arms" / mode),
                   "--corpus", "data/finetune/spatial_band_v1_fold0_20261024",
                   "--calibration", "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json",
                   "--source", str(args.source), "--reference-source", str(reference_source),
                   "--vendor", "vendor/smartgrids", "--output", str(args.output / "expanded" / mode)]
        (args.output / "expanded").mkdir(exist_ok=True)
        with (args.output / f"expanded_{mode}.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        return {"mode": mode, "returncode": result.returncode, "parameter_updates": 0}
    with ThreadPoolExecutor(2) as pool:
        expanded_results = list(pool.map(expanded, list(enumerate(labels))))
    (args.output / "EXIT_expanded.json").write_text(json.dumps(expanded_results, indent=2) + "\n")
    if any(result["returncode"] != 0 for result in expanded_results):
        raise RuntimeError("Expanded H2 evaluation stopped; preserve artifacts")
    (args.output / "LAUNCHES.json").write_text(json.dumps(launches, indent=2) + "\n")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"training_complete": True, "extended": gate["eligible"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
