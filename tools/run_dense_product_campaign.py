"""Bounded paired pilot with deterministic resume integration and expansion."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from finetuning.factorized_vq import long_training_gate


def identical(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(identical(a, b) for a, b in zip(left, right))
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--kind", choices=("product", "local-norm"), default="product")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Two free GPUs required")
    args.output.mkdir(parents=True, exist_ok=False)
    base_env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    if args.kind == "product" and args.initialization is None:
        with (args.output / "preparation.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "finetuning.prepare_dense_product",
                            "--output", str(args.output / "initialization")],
                           env={**base_env, "CUDA_VISIBLE_DEVICES": "0"}, stdout=log, stderr=subprocess.STDOUT, check=True)
    assets = (args.initialization or args.output / "initialization") / "assets.pt"
    labels = ("flat", "product") if args.kind == "product" else ("global", "local")

    def pair(stage, extra, container):
        def run(item):
            gpu, variant = item
            destination = args.output / container / variant
            destination.parent.mkdir(exist_ok=True)
            options = (["--variant", variant, "--assets", str(assets)] if args.kind == "product"
                       else ["--variant", "flat", "--normalization", variant])
            command = [sys.executable, "-m", "finetuning.run_dense_product_study",
                       "--output", str(destination)] + options + extra
            with (args.output / f"{stage}_{variant}.log").open("w") as log:
                process = subprocess.Popen(command, env={**base_env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                (args.output / f"LAUNCH_{stage}_{variant}.json").write_text(json.dumps({
                    "gpu": gpu, "pid": process.pid, "command": command}, indent=2) + "\n")
                return {"variant": variant, "stage": stage, "returncode": process.wait()}
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, list(enumerate(labels))))
        (args.output / f"EXIT_{stage}.json").write_text(json.dumps(results, indent=2) + "\n")
        if any(row["returncode"] for row in results):
            raise RuntimeError(f"Dense product stage failed: {stage}; preserve logs and weights")

    pair("smoke", ["--smoke"], "smoke")
    pair("smoke_resume1", ["--smoke", "--stop-after", "1"], "smoke_resume")
    pair("smoke_resume2", ["--smoke", "--resume"], "smoke_resume")
    evidence = []
    for variant in labels:
        direct = torch.load(args.output / "smoke" / variant / "last.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(args.output / "smoke_resume" / variant / "last.pt", map_location="cpu", weights_only=False)
        valid = identical(direct, resumed)
        evidence.append({"variant": variant, "full_checkpoint_exactly_equal": valid})
        del direct, resumed
    (args.output / "RESUME_INTEGRATION.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if not all(row["full_checkpoint_exactly_equal"] for row in evidence):
        raise RuntimeError("Resume integration failed; no scientific pilot authorized")
    pair("pilot", [], "arms")
    pair("expanded", ["--expanded-only"], "arms")

    def result(variant, update):
        return json.loads((args.output / "arms" / variant / f"check_{update:06d}/results.json").read_text())

    selections = {variant: json.loads((args.output / "arms" / variant / "selection.json").read_text())["update"]
                  for variant in labels}
    control_label, candidate_label = labels
    product = result(candidate_label, selections[candidate_label])
    flat = result(control_label, selections[control_label])
    if product["cells"] != flat["cells"] or product["windows"] != flat["windows"]:
        raise RuntimeError("Investment gate requires identical acquired coverage")
    history = [json.loads(line) for line in (args.output / "arms" / candidate_label / "history.jsonl").read_text().splitlines()][-64:]
    evidence = {"initial_product": result(candidate_label, 0)["macro_families"],
                "previous_product": result(candidate_label, 6144)["macro_families"],
                "final_product": result(candidate_label, 8192)["macro_families"],
                "flat_rmse": flat["macro_families"], "product_bytes": product["stream_bytes"],
                "flat_bytes": flat["stream_bytes"], "perplexity": float(np.mean([r["perplexity"] for r in history]))}
    # Gate is recorded, never launches an unregistered long stage.
    write = {"eligible_for_further_investment": long_training_gate(**evidence), "evidence": evidence,
             "selections": selections, "automatic_extension": False, "final_acceptance": False,
             "scientific_updates": 16384, "kind": args.kind,
             "candidate_name": candidate_label, "control_name": control_label}
    (args.output / "INVESTMENT_GATE.json").write_text(json.dumps(write, indent=2) + "\n")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "output": str(args.output), "gate": write}), flush=True)


if __name__ == "__main__":
    main()
