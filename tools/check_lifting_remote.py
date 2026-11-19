"""Remote dataset hash and deterministic resume integration checks."""
import argparse
import json
import shlex
from pathlib import Path

from tools.sync_research_remote import REMOTE, command, connect

SCRIPT = r'''
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch

root = Path(sys.argv[1])
torch.set_num_threads(2)
audit = Path("results/experiments/prospective_p0_v2_fold0_20261114")
protocol = json.loads((audit / "protocol.json").read_text())
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
corpus = Path("data/finetune/spatial_band_v1_fold0_20261024")
assert digest(corpus / "samples_manifest.csv") == protocol["manifest_sha256"]
assert digest(corpus / "channel_maps.jsonl") == protocol["channel_maps_sha256"]
windows = [json.loads(line) for line in (audit / "windows.jsonl").read_text().splitlines()]
def check_window(row):
    assert digest(corpus / "train" / row["sample"]) == row["sha256"], row["sample"]
with ThreadPoolExecutor(4) as pool:
    list(pool.map(check_window, windows))
def run(pair):
    arm, gpu = pair
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    base = [".venv/bin/python", "-m", "finetuning.run_lifting_study",
            "--corpus", str(corpus), "--calibration", str(audit / "reference_scales.json"),
            "--arm", arm, "--target-bpc", "0.8", "--updates", "64", "--check-every", "32"]
    with (root / f"{arm}.log").open("w") as log:
        for name, extra in [("continuous", []), ("resumed", ["--stop-after", "32"]),
                            ("resumed", ["--resume"])]:
            subprocess.run(base + ["--output", str(root / f"{arm}_{name}")] + extra,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    a = torch.load(root / f"{arm}_continuous/last.pt", map_location="cpu", weights_only=False)
    b = torch.load(root / f"{arm}_resumed/last.pt", map_location="cpu", weights_only=False)
    maximum = max(float((a["model"][key] - b["model"][key]).abs().max()) for key in a["model"])
    return {"arm": arm, "gpu": gpu, "updates": 64,
            "resume_max_parameter_difference": maximum,
            "rng_identical": a["python_rng"] == b["python_rng"],
            "lambda_identical": a["multiplier"] == b["multiplier"]}
with ThreadPoolExecutor(2) as pool:
    results = list(pool.map(run, [("fixed", 0), ("learned", 1)]))
passed = all(r["resume_max_parameter_difference"] == 0 and r["rng_identical"]
             and r["lambda_identical"] for r in results)
result = {"passed": passed, "results": results, "train_npz_hashes_verified": len(windows),
          "manifest_and_maps_hashes_match": True, "heldout_arrays_opened": False}
(root / "INTEGRATION.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="lifting_integration_v3_fold0_20261114")
    args = parser.parse_args()
    root = Path("results/experiments") / args.name
    root.mkdir(parents=True, exist_ok=False)
    client = connect()
    try:
        command(client, f"mkdir {shlex.quote(f'{REMOTE}/{root}')}")
        sftp = client.open_sftp()
        script_path = f"{REMOTE}/{root}/CHECK_SCRIPT.py"
        with sftp.open(script_path, "w") as stream:
            stream.write(SCRIPT)
        print(command(client, f"cd {REMOTE}\n.venv/bin/python {shlex.quote(script_path)} {shlex.quote(str(root))}"))
        sftp.get(f"{REMOTE}/{root}/INTEGRATION.json", str(root / "INTEGRATION.json"))
        sftp.close()
        if not json.loads((root / "INTEGRATION.json").read_text())["passed"]:
            raise RuntimeError("Remote integration did not pass; inspect preserved artifacts")
    finally:
        client.close()


if __name__ == "__main__":
    main()
