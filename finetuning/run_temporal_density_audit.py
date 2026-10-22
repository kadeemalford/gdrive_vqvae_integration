#!/usr/bin/env python3
"""Resume-safe detailed evaluation, followed by validation-only matched DCT control."""
import argparse
from pathlib import Path
import subprocess
import sys

try:
    from .run_sampling_experiment import ROOT
    from .run_temporal_density_long import require_cuda
except ImportError:
    from run_sampling_experiment import ROOT
    from run_temporal_density_long import require_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation, control = args.evaluation_dir.resolve(), args.control_dir.resolve()
    source = ROOT / "results/experiments/temporal_density_long_20261008"
    if not (source / "COMPLETE").is_file():
        raise ValueError("Complete both long training arms before auditing")
    if not (evaluation / "COMPLETE").is_file():
        require_cuda()
    command = [sys.executable, "finetuning/evaluate_physical_candidates.py", "--output-dir", str(evaluation),
               "--candidate", "standard=escape-v1=results/experiments/temporal_density_pilot_20261008/standard/best_macro.pt",
               "--candidate", f"long_2x=escape-v1={source / 'temporal-2x/best_macro.pt'}",
               "--candidate", f"long_4x=escape-v1={source / 'temporal-4x/best_macro.pt'}"]
    if evaluation.exists():
        command += ["--resume"]
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    with evaluation.with_suffix(".pipeline.log").open("a") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not (control / "COMPLETE").is_file():
            subprocess.run([sys.executable, "finetuning/benchmark_matched_temporal_rates.py",
                            "--evaluation-dir", str(evaluation), "--output-dir", str(control)],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f"Detailed evaluation: {evaluation}\nMatched control: {control}")


if __name__ == "__main__":
    main()
