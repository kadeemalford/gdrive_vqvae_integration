"""Train-only frozen VQ diagnostics, real wire replay and bounded covariance."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression.alphabet_stream import HEADER
from compression.full_tile_vq_stream import FullTileVQWire
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.fit_lifting_allocation import fitting_rows, transport
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.model_factory import build_model
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust
from finetuning.run_spatial_opportunity import select
from finetuning.streaming_physical_summary import StreamingPhysicalSummary


def select_probes(corpus):
    selected = select(corpus)
    fitting = fitting_rows(selected)
    probes = {r["sample_path"]: r for r in fitting + [r for r in selected if r["role"] == "check"]}
    roles = {r["drive_id"]: r["role"] for r in selected}
    files = defaultdict(list)
    for row in csv.DictReader((corpus / "samples_manifest.csv").open()):
        if row["split"] == "train":
            files[row["drive_id"]].append(row)
    for file_id, rows in files.items():
        row = min(rows, key=lambda r: (int(r["channels"]), int(r["time_samples"]), r["sample_path"]))
        probes.setdefault(row["sample_path"], {**row, "role": roles[file_id]})
    return fitting, sorted(probes.values(), key=lambda r: r["sample_path"])


class Moments:
    def __init__(self, dimension):
        self.n = 0
        self.sum = np.zeros(dimension)
        self.gram = np.zeros((dimension, dimension))

    def update(self, values):
        values = np.asarray(values, np.float64)
        self.n += len(values)
        self.sum += values.sum(0)
        self.gram += values.T @ values

    def result(self):
        if not self.n:
            return {"positions": 0, "effective_rank": None}
        covariance = self.gram / self.n - np.outer(self.sum / self.n, self.sum / self.n)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
        energy = eigenvalues.sum()
        probabilities = eigenvalues[eigenvalues > 0] / energy if energy else np.array([])
        return {"positions": self.n, "trace": float(energy), "eigenvalues": eigenvalues.tolist(),
                "top16_variance_fraction": float(eigenvalues[:16].sum() / energy) if energy else None,
                "effective_rank": float(np.exp(-(probabilities * np.log(probabilities)).sum())) if energy else None}


def pair_moments(values, valid, axis):
    left = [slice(None)] * 2
    right = [slice(None)] * 2
    left[axis], right[axis] = slice(None, -1), slice(1, None)
    mask = valid[tuple(left)] & valid[tuple(right)]
    a, b = values[tuple(left)][mask], values[tuple(right)][mask]
    return np.array([len(a), a.sum(), b.sum(), a @ a, b @ b, a @ b], np.float64)


def pair_result(moments):
    n, a, b, aa, bb, ab = moments
    if not n:
        return {"pairs": 0, "correlation": None}
    va, vb = aa - a * a / n, bb - b * b / n
    denominator = np.sqrt(max(va, 0) * max(vb, 0))
    return {"pairs": int(n), "correlation": float((ab - a * b / n) / denominator) if denominator > 0 else None}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/finetune/spatial_band_v1_fold0_20261024"))
    parser.add_argument("--calibration", type=Path, default=Path("results/experiments/prospective_p0_v2_fold0_20261114/reference_scales.json"))
    parser.add_argument("--source", type=Path, default=Path("results/remote_setup/dense_transfer_fold0_weights_p93.pt"))
    parser.add_argument("--vendor", type=Path, default=Path("vendor/smartgrids"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor.resolve()))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    fitting, probes = select_probes(args.corpus)
    if any(row["split"] != "train" for row in fitting + probes):
        raise ValueError("Only training waveforms authorized")
    args.output.mkdir(parents=True, exist_ok=False)
    identifier = sha256(args.source)
    expected = {row["sample_path"]: sha256(args.corpus / "train" / Path(row["sample_path"]).name) for row in probes}
    write_json(args.output / "protocol.json", {
        "kind": "full-tile-bottleneck-v1", "parameter_updates": 0, "waveform_split": "train",
        "source_sha256": identifier, "calibration_sha256": sha256(args.calibration),
        "manifest_sha256": sha256(args.corpus / "samples_manifest.csv"),
        "probes": probes, "fitting": [row["sample_path"] for row in fitting],
        "input_sha256": expected,
        "sources": {str(p): sha256(p) for p in [Path(__file__), Path("compression/full_tile_vq_stream.py"),
                    Path("finetuning/joint_anchor_vq.py"), Path("finetuning/model_factory.py"),
                    Path("finetuning/streaming_physical_summary.py")]},
        "limits": "Descriptive correlations and covariance, not a continuous ceiling or achievable bit saving"})

    def load(row):
        path = args.corpus / "train" / Path(row["sample_path"]).name
        if sha256(path) != expected[row["sample_path"]]:
            raise ValueError("Probe input changed")
        with np.load(path, allow_pickle=False) as data:
            raw = data["data"]
        if raw.dtype != np.float32 or raw.shape != (int(row["channels"]), int(row["time_samples"])) or not np.isfinite(raw).all():
            raise ValueError("Invalid native training target")
        return raw

    saved = torch.load(args.source, map_location="cpu", weights_only=False)
    config = {**saved["config"], "exact_eval_quantization": True}
    model = build_model(config, saved, "cuda").eval().requires_grad_(False)
    receiver_model = build_model(config, saved, "cuda").eval().requires_grad_(False)
    counts = np.zeros(2048, np.int64)
    for row in fitting:
        raw = load(row)
        if np.all(raw.view(np.uint32) == raw.view(np.uint32).flat[0]):
            continue
        packet = encode_robust(raw)
        _, _, ids, _, _ = exact_branch_forward(model, torch.from_numpy(packet.values)[None, None].cuda())
        counts += np.bincount(ids.cpu().numpy(), minlength=2048)
    sender = FullTileVQWire(model, counts, identifier)
    registry = sender.setup_bytes()
    receiver = FullTileVQWire.from_setup(receiver_model, registry)
    (args.output / "registry.zlib").write_bytes(registry)
    maps = {r["channel_map_id"]: r for r in map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    scales = json.loads(args.calibration.read_text())["scales"]
    summaries = {role: StreamingPhysicalSummary(scales, maps) for role in ("fit", "check")}
    covariance = {role: {key: Moments(128) for key in ("z", "z_minus_q")} for role in summaries}
    pairs = defaultdict(lambda: np.zeros(6))
    layouts = defaultdict(lambda: defaultdict(int))
    usage = {role: np.zeros(2048, np.int64) for role in summaries}
    (args.output / "frames").mkdir()
    for row in probes:
        raw = load(row)
        role = row["role"]
        packet = encode_robust(raw)
        x = torch.from_numpy(packet.values)[None, None].cuda()
        direct, _, ids, latent, _ = exact_branch_forward(model, x)
        frame = sender.encode(raw)
        restored = receiver.decode(frame)
        literal = frame[:4] in (b"FRC1", b"FRC2")
        expected_raw = raw if literal else decode(direct[0, 0].cpu().numpy(), packet.side_stream)
        if expected_raw.tobytes() != restored.tobytes():
            raise AssertionError("Independent physical replay differs")
        summaries[role].update(row, raw, restored, len(frame))
        (args.output / "frames" / (Path(row["sample_path"]).stem + ".frame")).write_bytes(frame)
        h, w = raw.shape
        bucket = layouts[f"{role}/{h}x{w}"]
        bucket["windows"] += 1
        bucket["cells"] += raw.size
        bucket["external_padding_cells"] += (h + (-h) % 16) * (w + (-w) % 16) - raw.size
        bucket["frame_bytes"] += len(frame)
        if literal:
            bucket["literal_windows"] += 1
            continue
        recovered, _, _, _ = receiver.stream.decode(frame)
        np.testing.assert_array_equal(ids.cpu().numpy(), recovered)
        fields = HEADER.unpack_from(frame)
        bucket["positions"] += int(np.prod(latent))
        bucket["payload_bytes_including_index_crc"] += fields[-2]
        usage[role] += np.bincount(recovered, minlength=2048)
        z = model.encoder(F.pad(x, (0, (-w) % 16, 0, (-h) % 16)))
        q = model.quantizer.embedding(ids).view(1, *latent, 128).permute(0, 3, 1, 2).contiguous()
        for key, values in (("z", z), ("z_minus_q", z - q)):
            covariance[role][key].update(values.permute(0, 2, 3, 1).reshape(-1, 128).cpu().numpy())
        normalized_error = direct[0, 0].cpu().numpy().astype(np.float64) - packet.values
        for key, values in (("target", packet.values.astype(np.float64)), ("error", normalized_error)):
            # Center separately per physical channel/window; never concatenate gapped windows.
            values = values.copy()
            for channel in range(h):
                mask = packet.valid[channel]
                if mask.any():
                    values[channel] -= values[channel, mask].mean()
            for axis, label in ((0, "space"), (1, "time")):
                pairs[f"{role}/{key}/{label}"] += pair_moments(values, packet.valid, axis)
    results = {}
    for role, summary in summaries.items():
        rows = [r for r in probes if r["role"] == role]
        geometry = transport(rows, maps)
        (args.output / f"transport_{role}.zlib").write_bytes(geometry)
        results[role] = summary.result(len(registry) + len(geometry))
        results[role]["latent_covariance"] = {key: value.result() for key, value in covariance[role].items()}
        frequency = usage[role][usage[role] > 0] / usage[role].sum()
        results[role]["index_usage"] = {"active": int((usage[role] > 0).sum()),
            "empirical_entropy_bits_no_tables": float(-(frequency * np.log2(frequency)).sum())}
    write_json(args.output / "results.json", {"roles": results, "layouts": dict(layouts),
        "lag1": {key: pair_result(value) for key, value in pairs.items()},
        "source_unchanged": sha256(args.source) == identifier, "parameter_updates": 0})
    if sha256(args.source) != identifier:
        raise AssertionError("Frozen source changed")
    (args.output / "COMPLETE").touch()
    print(json.dumps({"complete": True, "probes": len(probes),
        "roles": {role: {key: value[key] for key in ("macro_families", "stream_bpc", "windows")}
                  for role, value in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
