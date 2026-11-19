"""Smoke all capacity-matched conditioning arms before the registered scientific run."""
import argparse
import json
import os
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from compression.alphabet_stream import AlphabetStream
from compression.conditioned_vq_stream import unpack_context

METHODS = ("constant", "hardware", "category")


def verify(output, smoke):
    results = [json.loads((output / method / "results.json").read_text()) for method in METHODS]
    first = results[0]
    for result in results[1:]:
        for key in ("updates", "cells", "family_updates", "group_updates", "schedule_sha256",
                    "missing_updates", "initial_nrmse_star", "seen", "heldout", "norm_layers"):
            if result[key] != first[key]:
                raise AssertionError(f"Conditioning arms disagree on {key}")
    reference = {}
    for method in METHODS:
        folder = output / method / "check_000000"
        metadata, raw_counts = zlib.decompress((folder / "registry.zlib").read_bytes()).split(b"\n", 1)
        stream = AlphabetStream(np.frombuffer(raw_counts, "<i8"), json.loads(metadata)["identifier"])
        frames = list((folder / "normal").glob("*.frame"))
        if len(frames) != (2 if smoke else 136):
            raise AssertionError("Missing initial conditioning frames")
        for path in frames:
            _, inner = unpack_context(path.read_bytes(), method)
            _, absent = unpack_context((folder / "missing" / path.name).read_bytes(), method)
            if inner != absent:
                raise AssertionError("Zero FiLM must not change initial indices when metadata is absent")
            if inner[:4] == b"FAK1":
                indices, shape, latent, side = stream.decode(inner)
                signature = (indices.tobytes(), shape, latent, side)
            else:
                signature = (inner,)
            if method == "constant":
                reference[path.name] = signature
            elif reference[path.name] != signature:
                raise AssertionError("Conditioned source indices differ across the paired arms")
    endpoint = 2 if smoke else 16384
    constant = output / "constant" / f"check_{endpoint:06d}"
    for frame in (constant / "normal").glob("*.frame"):
        if frame.read_bytes() != (constant / "missing" / frame.name).read_bytes():
            raise AssertionError("Constant condition must ignore metadata absence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Conditioning contrast requires free GPUs")
    args.output.mkdir(parents=True, exist_ok=False)
    for smoke in (True, False):
        phase = args.output / ("smoke" if smoke else "scientific")
        phase.mkdir()

        def run(item):
            gpu, method = item
            command = [sys.executable, "-m", "finetuning.run_physical_conditioning",
                       "--method", method, "--output", str(phase / method)]
            if smoke:
                command.append("--smoke")
            with (phase / f"{method}.log").open("w") as log:
                process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"},
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                (phase / f"LAUNCH_{method}.json").write_text(json.dumps({
                    "pid": process.pid, "gpu": gpu, "command": command}, indent=2) + "\n")
                return {"method": method, "returncode": process.wait()}
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, [(0, "constant"), (1, "hardware")]))
        if all(row["returncode"] == 0 for row in results):
            results.append(run((0, "category")))
        (phase / "EXIT.json").write_text(json.dumps(results, indent=2) + "\n")
        if len(results) != 3 or any(row["returncode"] for row in results):
            raise RuntimeError("Conditioning interrupted; preserve logs, no automatic repeat")
        verify(phase, smoke)
        (phase / "COMPLETE").touch()
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
