"""Audit delivered bytes and standalone decoding, without accessing acquisition data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

from compression.feasibility_installation_envelope import unpack
from compression.feasibility_portable import decode_bytes
from finetuning.run_feasibility_fit_routing import ROOT, identity, verify, write_json


def audit(output):
    def document(filename):
        return json.loads((output/filename).read_text())
    verify(output/"plan.json", document("plan.freeze.json"))
    verify(output/"decisions.json", document("decisions.freeze.json"))
    results, plan = document("results.json"), document("plan.json")
    verify(output/"decisions.json", results["decisions"])
    for name, meta in plan["sources"].items():
        verify(ROOT/name, meta)
    if (identity(output/"protocol.original.md")["sha256"]
            != plan.get("report_sha256", plan.get("protocol_sha256"))):
        raise ValueError("Prospective report differs")
    dec = document("decisions.json")
    forge = "first_trace_zero_based" in plan
    decisions = dec if forge else dec["ratios"]
    fits = [arm for decision in decisions.values() for fold in decision["folds"]
            for arm in fold["arms"].values()]
    checks = [arm for r in results["rows"] for arm in r["arms"].values()]
    streams = artifacts = 0
    for arm in fits + checks:
        for name, meta in arm["artifacts"].items():
            verify(output/name, meta)
            artifacts += 1
        total = sum(item["bytes"] for item in arm["artifacts"].values()) + arm["source_bytes"]
        if total != arm["total_bytes"]:
            raise ValueError("Complete byte ledger differs")
        def file(suffix):
            name, = [name for name in arm["artifacts"] if name.endswith(suffix)]
            return output/name
        if forge:
            geo = zlib.decompress(file(".geometry").read_bytes())
            if len(geo) != 3600 + 240 * plan["traces"]:
                raise ValueError("Incomplete SEG-Y geometry")
        with zipfile.ZipFile(io.BytesIO(unpack(file(".fie").read_bytes()))) as install, \
                zipfile.ZipFile(file(".transport.zip")) as transport:
            bindings = json.loads(transport.read("bindings.json"))
            if len(bindings) != len(arm["records"]):
                raise ValueError("Incorrect transport bindings")
            for i, record in enumerate(arm["records"]):
                key = f"{i:03d}"
                if bindings[key]["sample_path"] != record["sample_path"]:
                    raise ValueError("Source binding changed")
                table = install.read(bindings[key]["installation_sha256"]+".ftb")
                q, x = decode_bytes(transport.read(key+".fqs"), table)
                if (len(x) != 4*record["cells"] or len(q) != len(x)
                        or hashlib.sha256(q).hexdigest() != record["q_sha256"]
                        or hashlib.sha256(x).hexdigest() != record["waveform_sha256"]
                        or record["bound_violations"] != 0):
                    raise ValueError("Stored delivery did not reproduce the scored waveform")
                streams += 1
    return dict(fit_arms=len(fits), check_arms=len(checks), delivered_files_verified=artifacts,
                streams_decoded=streams, result_identity=identity(output/"results.json"),
                report_original=identity(output/"protocol.original.md"), source_reads=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    write_json(args.directory/"delivery_audit.json", result)
    print(json.dumps(result))
