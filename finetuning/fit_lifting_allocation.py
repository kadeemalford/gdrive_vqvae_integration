"""Frozen-network coordinate allocation using transmitted hard coefficient costs."""
import argparse
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from compression.adaptive_coefficient_stream import encode_band, pack, pack_best, unpack_any
from compression.adaptive_lifting_stream import decode_frame, descriptor, encode_frame
from compression.lifting_stream import load_receiver, model_digest
from compression.research_coefficient_stream import HEADER, quantize
from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.evaluate_lifting_curve import select_receiver, summarize
from finetuning.reversible_normalization import parse_side_stream
from finetuning.robust_escape import encode_robust
from finetuning.run_spatial_opportunity import load, select

GRID = [.25, .5, .75, 1., 1.5, 2., 3., 4.]
LAMBDAS = [.01, .03, .1, .3, 1., 3.]


def coordinate_search(evaluate, dimensions, choices, multiplier, passes=3, initial=3):
    current = [initial] * dimensions
    trace = []
    for iteration in range(passes):
        changed = False
        for band in range(dimensions):
            best = current[band]
            error, rate = evaluate(tuple(current))
            objective = error ** 2 + multiplier * rate
            for choice in range(choices):
                proposal = current.copy()
                proposal[band] = choice
                trial_error, trial_rate = evaluate(tuple(proposal))
                trial = trial_error ** 2 + multiplier * trial_rate
                if trial < objective - 1e-10:
                    objective, best = trial, choice
            if best != current[band]:
                current[band] = best
                changed = True
                error, rate = evaluate(tuple(current))
                trace.append({"pass": iteration, "band": band, "choices": current.copy(),
                              "approximate_physical_nrmse": error, "hard_fit_bpc": rate})
        if not changed:
            break
    error, rate = evaluate(tuple(current))
    return current, error, rate, trace


def fitting_rows(selected):
    files = defaultdict(list)
    for row in selected:
        if row["role"] == "fit":
            files[row["drive_id"]].append(row)
    return [sorted(rows, key=lambda r: r["sample_path"])[len(rows) // 2]
            for _, rows in sorted(files.items())]


def file_weights(rows):
    files_per_group = Counter(r["group_id"] for r in rows)
    groups_per_family = Counter(group.split("/")[0] for group in files_per_group)
    families = len(groups_per_family)
    return [1 / (families * groups_per_family[r["family"]] * files_per_group[r["group_id"]])
            for r in rows]


def transport(rows, maps):
    data = [{"drive_id": r["drive_id"], "dataset": r["dataset"],
             "block": int(r["block_index"]), "start": int(r["time_start"]),
             "dt_s": float(r["dt_s"]), "dx_m": float(r["dx_m"]),
             "source_channel_indices": maps[r["channel_map_id"]]["source_channel_indices"]}
            for r in rows]
    return zlib.compress(json.dumps(data, sort_keys=True, separators=(",", ":")).encode(), 6)


@torch.no_grad()
def prepare_fit(model, rows, corpus, scales):
    weights = file_weights(rows)
    grouped = defaultdict(list)
    metadata = []
    cells = 0
    fixed_literal_bytes = 0
    for row, weight in zip(rows, weights):
        raw = load(corpus, row)
        packet = encode_robust(raw)
        if not packet.valid.all():
            raise ValueError("This allocation calibration requires the audited no-escape training subset")
        cells += raw.size
        bits = raw.view(np.uint32)
        literal = bool(np.all(bits == bits.flat[0]))
        if literal:
            fixed_literal_bytes += len(pack_best([np.array([bits.flat[0]], np.uint32).view(np.int32)],
                                                {"kind": "literal-float32-v1", "shape": list(raw.shape)}, b""))
            continue
        _, _, scale, _, _ = parse_side_stream(packet.side_stream)
        denominator = np.maximum(raw.astype(np.float64).std(axis=1), scales[row["group_id"]])
        grouped[raw.shape].append((packet.values, scale / denominator, weight, len(packet.side_stream)))
    band_costs = np.zeros((len(model.log_step), len(GRID)), np.int64)
    batches = []
    base_steps = model.steps.detach().cpu().numpy().astype(np.float64)
    device = next(model.parameters()).device
    for shape, entries in sorted(grouped.items()):
        tensor = torch.from_numpy(np.stack([entry[0] for entry in entries]))[:, None].to(device)
        coefficients = model.analysis(tensor)
        indices = model.band_indices(shape)
        options = []
        shapes = [[1, 1, *band.shape[-2:]] for band in coefficients]
        for band, index in zip(coefficients, indices):
            choices = []
            values = band.cpu().numpy()
            for choice, factor in enumerate(GRID):
                step = float(base_steps[index] * factor)
                symbols = quantize(values, step)
                for sample in symbols:
                    chunk, _ = encode_band(sample[None])
                    band_costs[index, choice] += len(chunk)
                choices.append(torch.from_numpy(symbols.astype(np.float32)).to(device) * step)
            options.append(choices)
        batches.append({"target": tensor, "options": options, "indices": indices, "shape": shape,
                        "ratios": torch.from_numpy(np.stack([e[1] for e in entries])).float().to(device),
                        "weights": torch.tensor([e[2] for e in entries], device=device)})
        metadata.extend({"shape": shape, "band_shapes": shapes, "indices": indices, "side_bytes": e[3]}
                        for e in entries)
    return batches, band_costs, metadata, cells, fixed_literal_bytes, base_steps


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    source, update = select_receiver(args.study)
    data = source.read_bytes()
    model = load_receiver(data, args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original_digest = model_digest(model)
    selected = select(args.corpus)
    fitting = fitting_rows(selected)
    checks = [r for r in selected if r["role"] == "check"]
    scales = json.loads(args.calibration.read_text())["scales"]
    maps = {m["channel_map_id"]: m for m in map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    write_json(args.output / "protocol.json", {
        "kind": "hard_frozen_subband_coordinate_allocation_v1", "study": str(args.study),
        "receiver": str(source), "receiver_sha256": sha256(source), "selected_update": update,
        "grid": GRID, "lambdas": LAMBDAS, "max_passes": 3, "parameter_updates": 0,
        "fit": [r["sample_path"] for r in fitting], "check": [r["sample_path"] for r in checks],
        "calibration_sha256": sha256(args.calibration),
        "manifest_sha256": sha256(args.corpus / "samples_manifest.csv"),
        "sources": {str(p): sha256(p) for p in [
            Path(__file__), Path("compression/adaptive_coefficient_stream.py"),
            Path("compression/adaptive_lifting_stream.py"), Path("finetuning/prospective_metrics.py")]},
        "fit_objective": "approximate physical macro NRMSE squared + lambda*actual FRC2 bpc including per-frame histogram counts and setup",
        "evaluation": "actual float32 physical restoration, independent receiver, FRC1 whole-frame fallback allowed",
        "limits": "reused training configurations, posthoc calibration; not independent validation"})
    fit_setup = transport(fitting, maps)
    check_setup = transport(checks, maps)
    (args.output / "fit_setup.zlib").write_bytes(fit_setup)
    (args.output / "check_setup.zlib").write_bytes(check_setup)
    batches, costs, metadata, cells, literal_bytes, base_steps = prepare_fit(model, fitting, args.corpus, scales)
    identifier = model_digest(model)
    cache = {}

    def evaluate(vector):
        if vector not in cache:
            error = 0.
            for batch in batches:
                bands = [options[vector[index]] for options, index in zip(batch["options"], batch["indices"])]
                prediction = model.synthesis(bands, batch["shape"])
                channel_rms = (prediction - batch["target"]).square().mean(-1).sqrt()[:, 0]
                row_scores = (channel_rms * batch["ratios"]).mean(-1)
                error += float((row_scores * batch["weights"]).sum())
            steps = [float(base_steps[index] * GRID[choice]) for index, choice in enumerate(vector)]
            byte_count = int(sum(costs[index, choice] for index, choice in enumerate(vector)))
            byte_count += len(fit_setup) + literal_bytes
            for entry in metadata:
                header = {"shapes": entry["band_shapes"],
                          "codec": {"kind": model.VERSION, "shape": list(entry["shape"]),
                                    "steps": [steps[i] for i in entry["indices"]], "model_sha256": identifier}}
                byte_count += HEADER.size + entry["side_bytes"] + len(json.dumps(
                    header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            cache[vector] = error, 8 * byte_count / cells
        return cache[vector]

    profiles = []
    for multiplier in LAMBDAS:
        choices, error, rate, trace = coordinate_search(
            evaluate, len(base_steps), len(GRID), multiplier)
        steps = [float(base_steps[i] * GRID[c]) for i, c in enumerate(choices)]
        profiles.append({"lambda": multiplier, "choices": choices, "steps": steps,
                         "fit_approximate_physical_nrmse": error, "fit_hard_frc2_bpc": rate, "trace": trace})
        print(json.dumps({"lambda": multiplier, "fit_error": error, "fit_bpc": rate, "choices": choices}), flush=True)
    if model_digest(model) != original_digest:
        raise AssertionError("Frozen waveform weights changed")
    write_json(args.output / "profiles.json", {"profiles": profiles, "unique_fit_evaluations": len(cache)})
    (args.output / "FIT_COMPLETE").touch()
    del batches
    torch.cuda.empty_cache() if str(args.device).startswith("cuda") else None
    results = []
    receiver = load_receiver(data, args.device)
    (args.output / "receiver.pt").write_bytes(data)
    for index, profile in enumerate(profiles):
        directory = args.output / f"profile_{index}"
        directory.mkdir()
        fit_frames, fit_reconstructed = [], []
        adaptive_fit_bytes = len(fit_setup)
        for row in fitting:
            raw = load(args.corpus, row)
            frame = encode_frame(raw, model, profile["steps"])
            predicted = decode_frame(frame, receiver)
            arrays, metadata, side = unpack_any(frame)
            if metadata["kind"] == "literal-float32-v1":
                adaptive_fit_bytes += len(frame)
            else:
                adaptive_fit_bytes += len(pack(arrays, metadata, side))
            fit_frames.append(frame)
            fit_reconstructed.append((raw, predicted))
        measured_frc2_bpc = 8 * adaptive_fit_bytes / cells
        if abs(measured_frc2_bpc - profile["fit_hard_frc2_bpc"]) > 1e-10:
            raise AssertionError("Precomputed cost differs from fully serialized FRC2")
        fit_result = summarize(fitting, fit_reconstructed, fit_frames, scales, maps, len(fit_setup))
        write_json(directory / "fit_verification.json", {
            "precomputed_frc2_bpc": profile["fit_hard_frc2_bpc"],
            "serialized_frc2_bpc": measured_frc2_bpc,
            "serialized_with_legacy_fallback_bpc": fit_result["stream_bpc"],
            "approximate_physical_nrmse": profile["fit_approximate_physical_nrmse"],
            "actual_float32_physical_nrmse": fit_result["macro_families"]})
        frames, reconstructed = [], []
        for row in checks:
            raw = load(args.corpus, row)
            frame = encode_frame(raw, model, profile["steps"])
            predicted = decode_frame(frame, receiver)
            (directory / (Path(row["sample_path"]).stem + ".frc")).write_bytes(frame)
            frames.append(frame)
            reconstructed.append((raw, predicted))
        result = summarize(checks, reconstructed, frames, scales, maps, len(check_setup))
        result.update({k: v for k, v in profile.items() if k != "trace"})
        result.update(method=args.study.name, label=f"allocation/{args.study.name}/lambda{profile['lambda']:g}",
                      study=str(args.study), selected_update=update, receiver_bytes=len(data), shared_bytes=len(data))
        write_json(directory / "results.json", result)
        results.append({k: v for k, v in result.items() if k != "records"})
        print(json.dumps({k: result[k] for k in ("label", "macro_families", "stream_bpc")}), flush=True)
    write_json(args.output / "results.json", results)
    (args.output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
