"""Frozen development-validation evaluation; never opens final-test waveforms."""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

from compression.full_tile_vq_stream import FullTileVQWire
from compression.joint_anchor_vq_stream import JointAnchorWire, HEADER as JOINT_HEADER
from compression.alphabet_stream import HEADER as ALPHABET_HEADER
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.fit_lifting_allocation import fitting_rows, transport
from finetuning.joint_anchor_evaluation import calibrate, load_export
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.model_factory import build_model
from finetuning.robust_escape import encode_robust
from finetuning.reversible_normalization import parse_side_stream
from finetuning.run_spatial_opportunity import load, select
from finetuning.streaming_physical_summary import StreamingPhysicalSummary

CORPUS = Path("data/finetune/spatial_band_v1_fold0_20261024")
CALIBRATION = Path("results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json")


def candidates():
    first = Path("results/experiments/joint_anchor_vq_fold0_20261115/arms")
    second = Path("results/experiments/joint_encoder_observation_fold0_20261115/arms")
    return {
        "mature_reference": {"kind": "native", "path": Path("results/remote_setup/dense_transfer_fold0_weights_p93.pt"),
                             "selection": "historical selected epoch23"},
        "independent": {"kind": "joint", "path": first / "independent/best.pt", "selection": 30720},
        "h2_joint": {"kind": "joint", "path": first / "closed-loop/best.pt", "selection": 16384},
        "encoder_local": {"kind": "joint", "path": second / "local-analysis/best.pt", "selection": 2048},
        "encoder_full": {"kind": "joint", "path": second / "full-observation/best.pt", "selection": 2048},
    }


def validation_rows(corpus):
    all_rows = list(csv.DictReader((corpus / "samples_manifest.csv").open()))
    rows = sorted([r for r in all_rows if r["split"] == "val"], key=lambda r: r["sample_path"])
    training_groups = {r["group_id"] for r in all_rows if r["split"] == "train"}
    if not rows or training_groups & {r["group_id"] for r in rows}:
        raise ValueError("Validation must be nonempty and configuration-disjoint from training")
    return rows


def scales_for_rows(rows, calibration):
    result, provenance = {}, {}
    for row in rows:
        group = row["group_id"]
        if group in calibration["scales"]:
            result[group] = calibration["scales"][group]
            provenance[group] = "frozen training configuration"
        else:
            result[group] = calibration["family_scales_for_unseen_configurations"][row["family"]]
            provenance[group] = "frozen training family median"
        if not np.isfinite(result[group]) or result[group] <= 0:
            raise ValueError("No positive training-only physical scale")
    return result, provenance


def load_validation(corpus, row, expected_sha256=None):
    if row.get("split") != "val":
        raise ValueError("Development evaluator forbids train/test waveform access through its validation reader")
    path = corpus / "val" / Path(row["sample_path"]).name
    content = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Validation window bytes changed")
    with np.load(io.BytesIO(content), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        for key in ("split", "drive_id", "channel_map_id", "block_index"):
            if key in row and str(metadata.get(key)) != str(row[key]):
                raise ValueError("Validation embedded provenance mismatch")
        raw = payload["data"]
    if (raw.dtype != np.float32 or raw.shape != (int(row["channels"]), int(row["time_samples"]))
            or not np.isfinite(raw).all()):
        raise ValueError("Validation target differs from the finite native prepared contract")
    return raw


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(candidates()), required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--integrity", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor.resolve()))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = candidates()[args.candidate]
    expected = json.loads((args.integrity / "expected_sha256.json").read_text())
    verified = json.loads((args.integrity / "REMOTE_VERIFIED.json").read_text())
    if not verified["passed"] or sha256(CORPUS / "samples_manifest.csv") != expected["manifest_sha256"]:
        raise ValueError("Validation integrity audit did not pass")
    if (sha256(CORPUS / "channel_maps.jsonl") != expected["maps_sha256"]
            or sha256(CALIBRATION) != expected["calibration_sha256"]):
        raise ValueError("Frozen physical preparation/calibration changed")
    if sha256(spec["path"]) != expected["checkpoints"][str(spec["path"])]:
        raise ValueError("Frozen checkpoint changed")
    if spec["kind"] == "joint":
        if json.loads((spec["path"].parent / "selection.json").read_text())["update"] != spec["selection"]:
            raise ValueError("Checkpoint reselection after registered validation protocol")
    rows = validation_rows(CORPUS)
    fitting = fitting_rows(select(CORPUS))
    if any(r["split"] != "train" for r in fitting):
        raise ValueError("Registry calibration is train-only")
    if {r["drive_id"] for r in rows} & {r["drive_id"] for r in fitting}:
        raise ValueError("Calibration/validation file leakage")
    calibration = json.loads(CALIBRATION.read_text())
    scales, provenance = scales_for_rows(rows, calibration)
    maps = {m["channel_map_id"]: m for m in map(json.loads, (CORPUS / "channel_maps.jsonl").open())}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {
        "kind": "frozen_development_validation_v1", "candidate": args.candidate, "split": "val",
        "model": str(spec["path"]), "model_sha256": sha256(spec["path"]), "selection": spec["selection"],
        "parameter_updates": 0, "checkpoint_reselection": False, "final_test_access": False,
        "windows": len(rows), "files": len({r["drive_id"] for r in rows}),
        "sample_paths": [r["sample_path"] for r in rows], "calibration_train_paths": [r["sample_path"] for r in fitting],
        "manifest_sha256": expected["manifest_sha256"], "calibration_sha256": sha256(CALIBRATION),
        "script_sha256": sha256(__file__), "integrity_sha256": sha256(args.integrity / "expected_sha256.json"),
        "sources": {str(path): sha256(path) for path in [
            Path("compression/full_tile_vq_stream.py"), Path("compression/joint_anchor_vq_stream.py"),
            Path("compression/alphabet_stream.py"), Path("finetuning/streaming_physical_summary.py"),
            Path("finetuning/joint_anchor_vq.py"), Path("finetuning/encoder_observation_vq.py"),
            Path("finetuning/prospective_metrics.py"), Path("finetuning/prospective_spectral.py"),
            Path("finetuning/reversible_normalization.py"), Path("finetuning/robust_escape.py")]},
        "vendor_sources": {name: sha256(args.vendor / "vqvae" / name)
                           for name in ("train_vqvae.py", "evaluate_vqvae.py")},
        "scales": scales, "scale_provenance": provenance,
        "limits": "Historically consumed development validation, not independent confirmation or operational acceptance"})
    source_data = spec["path"].read_bytes()
    if spec["kind"] == "native":
        saved = torch.load(spec["path"], map_location="cpu", weights_only=True)
        config = {**saved["config"], "exact_eval_quantization": True}
        model = build_model(config, saved, "cuda").eval().requires_grad_(False)
        receiver_model = build_model(config, saved, "cuda").eval().requires_grad_(False)
        counts = np.zeros(2048, np.int64)
        for row in fitting:
            raw = load(CORPUS, row)
            if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
                continue
            packet = encode_robust(raw)
            _, _, ids, _, _ = exact_branch_forward(model, torch.from_numpy(packet.values)[None, None].cuda())
            counts += np.bincount(ids.cpu().numpy(), minlength=2048)
        sender = FullTileVQWire(model, counts, sha256(spec["path"]))
        registry = sender.setup_bytes()
        receiver = FullTileVQWire.from_setup(receiver_model, registry)
    else:
        model = load_export(source_data, "cuda").eval().requires_grad_(False)
        receiver_model = load_export(source_data, "cuda").eval().requires_grad_(False)
        counts = calibrate(model, fitting, CORPUS)
        sender = JointAnchorWire(model, counts, sha256(spec["path"]))
        registry = sender.setup_bytes()
        receiver = JointAnchorWire.from_setup(receiver_model, registry)
    del source_data
    (args.output / "registry.zlib").write_bytes(registry)
    geometry = transport(rows, maps)
    (args.output / "transport.zlib").write_bytes(geometry)
    summary = StreamingPhysicalSummary(scales, maps)
    coverage = {"literal_windows": 0, "literal_cells": 0, "neural_ordinary_cells": 0, "exact_escape_cells": 0}
    frame_directory = args.output / "frames"
    frame_directory.mkdir()
    with (args.output / "frames_manifest.jsonl").open("w") as output:
        for index, row in enumerate(rows):
            raw = load_validation(CORPUS, row, expected["validation"][Path(row["sample_path"]).name])
            frame = sender.encode(raw)
            restored = receiver.decode(frame)
            if frame[:4] in (b"FRC1", b"FRC2"):
                if raw.tobytes() != restored.tobytes():
                    raise AssertionError("Validation literal bits changed")
                coverage["literal_windows"] += 1
                coverage["literal_cells"] += raw.size
            else:
                if frame[:4] == b"FJH1":
                    _, nm, ns, _, _, _ = JOINT_HEADER.unpack_from(frame)
                    side = frame[JOINT_HEADER.size + nm:JOINT_HEADER.size + nm + ns]
                elif frame[:4] == b"FAK1":
                    ns = ALPHABET_HEADER.unpack_from(frame)[8]
                    side = frame[ALPHABET_HEADER.size:ALPHABET_HEADER.size + ns]
                else:
                    raise AssertionError("Unknown validation frame semantics")
                shape, _, _, escaped, _ = parse_side_stream(side)
                if shape != raw.shape or raw.ravel()[escaped].tobytes() != restored.ravel()[escaped].tobytes():
                    raise AssertionError("Validation exact escape geometry/bits changed")
                coverage["exact_escape_cells"] += len(escaped)
                coverage["neural_ordinary_cells"] += raw.size - len(escaped)
            summary.update(row, raw, restored, len(frame))
            name = Path(row["sample_path"]).stem + ".frame"
            (frame_directory / name).write_bytes(frame)
            output.write(json.dumps({"frame": name, "sample_path": row["sample_path"],
                                     "bytes": len(frame), "group_id": row["group_id"]}) + "\n")
            if (index + 1) % 256 == 0:
                output.flush()
                print(json.dumps({"candidate": args.candidate, "windows": index + 1, "total": len(rows)}), flush=True)
    result = summary.result(len(registry) + len(geometry))
    if result["windows"] != len(rows) or result["cells"] != sum(int(r["channels"]) * int(r["time_samples"]) for r in rows):
        raise AssertionError("Incomplete acquired coverage")
    if sum(coverage[key] for key in ("literal_cells", "neural_ordinary_cells", "exact_escape_cells")) != result["cells"]:
        raise AssertionError("Physical coverage classes do not sum to acquired cells")
    if sha256(spec["path"]) != expected["checkpoints"][str(spec["path"])]:
        raise AssertionError("Frozen waveform weights changed")
    result.update(candidate=args.candidate, shared_full_model_bytes=spec["path"].stat().st_size,
                  registry_bytes=len(registry), parameter_updates=0, final_test_access=False,
                  checkpoint_reselection=False, frozen_model_sha256=sha256(spec["path"]),
                  coverage_classes=coverage)
    write_json(args.output / "results.json", result)
    (args.output / "COMPLETE").touch()
    print(json.dumps({k: result[k] for k in ("candidate", "macro_families", "stream_bpc", "windows")}), flush=True)


if __name__ == "__main__":
    main()
