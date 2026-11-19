"""Remote two-GPU inference-only postflight after both registered budgets finish."""
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
    studies = [Path(f"results/experiments/lifting_p2_{rate}_fold0_20261114")
               for rate in ("dense", "compact")]
    for study in studies:
        if not (study / "PAIR_BUDGET_COMPLETE").exists():
            raise RuntimeError(f"Training budget is not complete: {study}")
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("Inference postflight requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    common = ["--corpus", "data/finetune/spatial_band_v1_fold0_20261024",
              "--calibration", "results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json"]
    curve = [sys.executable, "-m", "finetuning.evaluate_lifting_curve", *common,
             "--output", str(args.output / "lifting_curve")]
    for study in studies:
        for arm in ("fixed", "learned"):
            curve.extend(["--study", str(study / arm)])
    references = [
        [sys.executable, "-m", "finetuning.evaluate_prospective_vq_reference", *common,
         "--checkpoint", f"results/remote_setup/{checkpoint}",
         "--vendor", "vendor/smartgrids", "--output", str(args.output / label)]
        for label, checkpoint in (
            ("vq_compact", "compact_transfer_fold0_weights.pt"),
            ("vq_dense", "dense_transfer_fold0_weights_p93.pt"))]
    curve_log = (args.output / "lifting_curve.log").open("w")
    vq_log = (args.output / "vq.log").open("w")
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    curve_process = subprocess.Popen(curve, env={**environment, "CUDA_VISIBLE_DEVICES": "0"},
                                    stdout=curve_log, stderr=subprocess.STDOUT, start_new_session=True)
    codes = []
    for reference in references:
        codes.append(subprocess.run(reference, env={**environment, "CUDA_VISIBLE_DEVICES": "1"},
                                    stdout=vq_log, stderr=subprocess.STDOUT).returncode)
        if codes[-1]:
            break
    curve_code = curve_process.wait()
    curve_log.close()
    vq_log.close()
    status = {"parameter_updates": 0, "curve_returncode": curve_code, "vq_returncodes": codes}
    (args.output / "EXIT.json").write_text(json.dumps(status, indent=2) + "\n")
    if curve_code or len(codes) != 2 or any(codes):
        raise RuntimeError("Postflight interrupted; preserve logs and partial artifacts")
    subprocess.run([sys.executable, "-m", "finetuning.summarize_lifting_frontier",
                    "--curve", str(args.output / "lifting_curve/results.json"),
                    "--vq", str(args.output / "vq_compact/results.json"),
                    "--vq", str(args.output / "vq_dense/results.json"),
                    "--output", str(args.output / "frontier.json")], check=True)
    (args.output / "COMPLETE").touch()
    print(json.dumps({"output": str(args.output), "complete": True, "parameter_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
