"""Byte-only local/remote integrity preflight; no waveform decoding or model fitting."""
import json
import shlex
from pathlib import Path

from finetuning.audit_prospective_preparation import sha256
from finetuning.development_validation import CALIBRATION, CORPUS, candidates, validation_rows
from finetuning.fit_lifting_allocation import fitting_rows
from finetuning.run_spatial_opportunity import select
from tools.sync_research_remote import REMOTE, command, connect

OUTPUT = Path("results/experiments/development_validation_integrity_20261115")

REMOTE_SCRIPT = r'''
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root=Path("results/experiments/development_validation_integrity_20261115")
expected=json.loads((root/"expected_sha256.json").read_text())
corpus=Path("data/finetune/spatial_band_v1_fold0_20261024")
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
assert digest(corpus/"samples_manifest.csv")==expected["manifest_sha256"]
assert digest(corpus/"channel_maps.jsonl")==expected["maps_sha256"]
assert digest(Path("results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json"))==expected["calibration_sha256"]
items=[(corpus/"val"/name,checksum) for name,checksum in expected["validation"].items()]
items += [(corpus/"train"/name,checksum) for name,checksum in expected["fitting"].items()]
items += [(Path(name),checksum) for name,checksum in expected["checkpoints"].items()]
def check(item):
    path,checksum=item
    if digest(path)!=checksum:
        raise ValueError(f"Integrity mismatch: {path}")
with ThreadPoolExecutor(4) as pool:
    list(pool.map(check,items))
result={"passed":True,"validation_npz_verified":len(expected["validation"]),
        "train_calibration_npz_verified":len(expected["fitting"]),
        "frozen_checkpoint_files_verified":len(expected["checkpoints"]),
        "waveforms_decoded":False,"parameter_updates":0,"final_test_waveforms_accessed":False}
(root/"REMOTE_VERIFIED.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
'''


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    rows = validation_rows(CORPUS)
    fitting = fitting_rows(select(CORPUS))
    expected = {
        "manifest_sha256": sha256(CORPUS / "samples_manifest.csv"),
        "maps_sha256": sha256(CORPUS / "channel_maps.jsonl"),
        "calibration_sha256": sha256(CALIBRATION),
        "validation": {Path(r["sample_path"]).name: sha256(CORPUS / "val" / Path(r["sample_path"]).name) for r in rows},
        "fitting": {Path(r["sample_path"]).name: sha256(CORPUS / "train" / Path(r["sample_path"]).name) for r in fitting},
        "checkpoints": {str(s["path"]): sha256(s["path"]) for s in candidates().values()},
        "scope": "Split val bytes and fixed training calibration only; no decoded signals or scores"}
    (OUTPUT / "expected_sha256.json").write_text(json.dumps(expected, indent=2) + "\n")
    client = connect()
    try:
        command(client, f"mkdir {shlex.quote(f'{REMOTE}/{OUTPUT}')}")
        sftp = client.open_sftp()
        sftp.put(str(OUTPUT / "expected_sha256.json"), f"{REMOTE}/{OUTPUT}/expected_sha256.json")
        with sftp.open(f"{REMOTE}/{OUTPUT}/VERIFY.py", "w") as stream:
            stream.write(REMOTE_SCRIPT)
        print(command(client, f"cd {REMOTE}\n.venv/bin/python {OUTPUT}/VERIFY.py"), flush=True)
        sftp.get(f"{REMOTE}/{OUTPUT}/REMOTE_VERIFIED.json", str(OUTPUT / "REMOTE_VERIFIED.json"))
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
