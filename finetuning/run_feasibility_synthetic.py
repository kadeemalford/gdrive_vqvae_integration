"""CPU-only report-107 D2b controls. Never reads FEBUS or starts training.

Run preflight first. Outputs are append-only: existing destinations are refused.
Full controls use five prescribed seeds and independent check seeds, with
8 x 131072 cells per realization. Smaller runs must be labelled smoke.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import sys
import time
import zipfile

import numpy as np
from scipy.signal import lfilter

from compression.feasibility_tables import (
    MAX_CELLS, MAX_TABLE_BYTES, MODES, RESET, TOTAL, fit_tables,
    quantize, reconstruct, score,
)
from compression.feasibility_stream import decode, encode, portable_object, select_support
from compression.feasibility_portable import decode_bytes


PREFLIGHT = Path("results/experiments/feasibility_coder_preflight_20261117")
SEEDS = (1701, 1702, 1703, 1704, 1705)
CONTROLS = ("gaussian_iid", "gaussian_ar1", "conditional_variance", "constant", "specials")
SOURCES = (
    "compression/feasibility_tables.py", "compression/feasibility_stream.py",
    "compression/feasibility_portable.py",
    "finetuning/run_feasibility_synthetic.py", "tests/test_feasibility_coder.py",
    "config/feasibility_protocol_v1.json",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def generate(kind, seed, samples):
    rng = np.random.default_rng(seed)
    shape = (samples, 8)
    if kind == "gaussian_iid":
        return rng.normal(size=shape).astype(np.float32)
    if kind == "gaussian_ar1":
        innovations = rng.normal(size=shape) * np.sqrt(1 - .95 ** 2)
        # Exact stationary first sample; channels remain independent.
        innovations[0] = rng.normal(size=8)
        return lfilter([1.], [1., -.95], innovations, axis=0).astype(np.float32)
    if kind == "conditional_variance":
        switches = rng.random(shape) >= .99
        states = (np.cumsum(switches, axis=0) + rng.integers(0, 2, size=8)) % 2
        return (rng.normal(size=shape) * np.where(states == 0, .25, 4.)).astype(np.float32)
    if kind == "constant":
        return np.zeros(shape, dtype=np.float32)
    if kind == "specials":
        bits = np.array([0, 0x80000000, 0x7fc12345, 0x7fa12345,
                         0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff,
                         1, 0x80000001, 0x3f800000, 0xbf800000], dtype=np.uint32)
        return bits[rng.integers(0, len(bits), size=shape)].view(np.float32)
    raise ValueError("Unknown control")


def calibration_sigma(x):
    """Median positive finite per-channel std of fit windows, float64."""
    deviations = []
    for start in range(0, len(x), RESET):
        block = x[start:start + RESET].astype(np.float64)
        for ch in range(block.shape[1]):
            values = block[:, ch]
            values = values[np.isfinite(values)]
            if len(values):
                std = float(np.std(values))
                if np.isfinite(std) and std > 0:
                    deviations.append(std)
    return float(np.median(deviations)) if deviations else None


def freeze_preflight(root):
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(1701)
    x = rng.normal(size=(1003, 3)).astype(np.float32)
    special = np.array([0x80000000, 0x7fc12345, 0x7fa12345, 0x7f800000,
                        0xff800000, 0x7f7fffff, 0xff7fffff], dtype=np.uint32)
    x.view("u4").reshape(-1)[10:17] = special
    x[900, 1] = 10000
    epsilon = .03
    q, positions, bits = quantize(x, epsilon)
    expected = reconstruct(q, positions, bits, epsilon)
    k, candidates = select_support(q, positions, bits, epsilon, [4, 5, 9])
    buffer = io.BytesIO()
    np.savez(buffer, source_bits=x.view("u4"), q=q, physical_positions=positions,
             physical_bits=bits, reconstruction_bits=expected.view("u4"))
    artifacts = {"fixture.npz": buffer.getvalue()}
    ledgers = {}
    for mode in MODES:
        tables = fit_tables(q, positions, epsilon, [4, 5, 9], k, mode)
        transport, ledgers[mode] = encode(q, positions, bits, tables, forced=2)
        qq, xx = decode(transport, tables.dump())
        if not np.array_equal(q, qq) or not np.array_equal(expected.view("u4"), xx.view("u4")):
            raise RuntimeError("Preflight roundtrip failed")
        artifacts[f"{mode}.ftb"] = tables.dump()
        artifacts[f"{mode}.fqs"] = transport
        artifacts[f"{mode}.pyz"] = portable_object(transport, tables.dump())
        pq, px = decode_bytes(transport, tables.dump())
        if pq != q.astype("<i4").tobytes() or px != expected.astype("<f4").tobytes():
            raise RuntimeError("Independent portable decoder mismatch")
    # Measured source installation, not a claim of an OS/runtime-independent object.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in SOURCES:
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, Path(name).read_bytes())
    artifacts["decoder_sources.zip"] = buffer.getvalue()
    specification = {
        "version": "feasibility-direct-rans-v1", "endianness": "little",
        "quantizer": "float64 division, nearest-even, int32; float64 multiply then float32; verify bound",
        "physical_escape": "q=0 placeholder, uint32 frame-flat position + uint32 original bits; unavailable bucket",
        "alphabet_escape": "ID 2K+1; exact int32 in forward order; context from decoded q",
        "symbols": "IDs q+K for [-K,K], then ESC; physical placeholder encoded normally",
        "buckets": "0 zero; 1..8 positive exponents; 9..16 negative; 17 unavailable",
        "contexts": "temporal bucket; spatiotemporal=18*temporal+spatial; physical adjacency by IDs",
        "reset": "every 1000 time samples, independent frames; final short frame retained",
        "support": "K in 15,63,255,1023,4095; minimum actual marginal fit transport+installation; smaller K tie",
        "fit": "Jeffreys .5; n<256 backoff row0; otherwise weight n/(n+256); no check adaptation",
        "cdf_total": TOTAL,
        "cdf": "1+floor((TOTAL-A)*p); largest remainder; ascending symbol ID tie; transmitted uint32 frequencies",
        "tables": "FTB1,u32 JSON length,canonical UTF8 JSON,zlib6(mapping u32 then rows u32),SHA256",
        "stream": "FQS1,32-byte installation SHA256,u32 times,u32 frames,frames,SHA256",
        "frame": "<BIIIII mode,times,physical_count,alphabet_count,payload_bytes,reserved0; physical pairs; q escapes; payload; CRC32",
        "fallback": "shortest actual frame among raw0,marginal1,contextual2; lower mode ID tie; whole installation charged",
        "rans": "byte rANS L=2^31,precision20; reverse encode,forward decode; final state uint64 followed by reversed emitted bytes",
        "rans_encode": "renorm while x>=((L>>20)<<8)*f; emit low byte; x=((x//f)<<20)+x%f+cdf[s]",
        "rans_decode": "s=bisect_right(cdf,x&(TOTAL-1))-1; x=f*(x>>20)+(x&(TOTAL-1))-cdf[s]; renorm bytes until x>=L",
        "rans_finalization": "8-byte initial decode state per entropy frame; decoded final state exactly L; no unread bytes",
        "integrity": "SHA256 transport+installation,CRC32 each frame; not cryptographic authentication",
        "constriction": "Categorical documents automatic rescaling/approximation; rejected; own integer rANS has no requantization",
        "memory": {"region_cell_cap": MAX_CELLS, "fit_table_working_ceiling_bytes": MAX_TABLE_BYTES,
                   "note": "table ceiling excludes source arrays, CDF Python lists and interpreter; measured process max RSS reported"},
        "costs": "transport exact disjoint ledger; tables+geometry+epsilon+backoff in installation; reset in frame headers",
        "deployment": "real executable ZIP_STORED .pyz includes stdlib-only independent decoder, installation and transport; Python3+stdlib is the declared shared platform; Python interpreter/OS are not bundled",
        "deployment_limit": "table_object_bytes is only transport+tables, not deployment; python_stdlib_object_bytes measures actual .pyz; no OS-independent executable or F4 gate claimed",
        "synthetic": {"seeds": SEEDS, "check_offset": 100000, "channels": 8,
                      "samples": 131072, "ratios": [.03, .1], "sigma": "fit median positive per-channel 1000-sample std; final tail included",
                      "constant_scale": "undefined; explicit 1.0 integrity-only surrogate, never physical calibration"},
    }
    artifacts["protocol.json"] = (json.dumps(specification, indent=2, sort_keys=True) + "\n").encode()
    for name, data in artifacts.items():
        with (root / name).open("xb") as stream:
            stream.write(data)
    manifest = {
        "artifacts": {name: digest(data) for name, data in artifacts.items()},
        "sources": {name: digest(Path(name).read_bytes()) for name in SOURCES},
        "support_fit_candidates": candidates, "fixture_selected_k": k,
        "fixture_ledgers": ledgers, "decoder_source_zip_bytes": len(artifacts["decoder_sources.zip"]),
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def verify_preflight(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        if digest((root / name).read_bytes()) != expected:
            raise RuntimeError(f"Frozen artifact mismatch: {name}")
    for name, expected in manifest["sources"].items():
        if digest(Path(name).read_bytes()) != expected:
            raise RuntimeError(f"Source changed since preflight; new version required: {name}")
    return manifest


def summarize(rows):
    """Monte Carlo uncertainty across prescribed seeds, not across cells."""
    summary = []
    for kind in CONTROLS:
        for ratio in (.03, .1):
            subset = [r for r in rows if r["control"] == kind and r["ratio"] == ratio]
            for mode in ("temporal", "spatiotemporal"):
                if not subset:
                    continue
                deltas = [r["arms"]["marginal"]["transport_bytes"] - r["arms"][mode]["transport_bytes"]
                          for r in subset]
                cdf = [(r["arms"]["marginal"]["integer_cdf_nll_bits"] -
                        r["arms"][mode]["integer_cdf_nll_bits"]) / r["cells"] for r in subset]
                object_deltas = [r["arms"]["marginal"]["python_stdlib_object_bytes"] -
                                 r["arms"][mode]["python_stdlib_object_bytes"] for r in subset]
                def stats(values):
                    return {"values": values, "mean": float(np.mean(values)),
                            "monte_carlo_se": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                            if len(values) > 1 else None}
                summary.append({"control": kind, "ratio": ratio, "mode": mode, "n_seeds": len(subset),
                                "cdf_saving_bpc": stats(cdf), "transport_saving_bytes": stats(deltas),
                                "python_stdlib_object_saving_bytes": stats(object_deltas)})
    return summary


def run(args):
    manifest = verify_preflight(args.preflight)
    if args.samples != 131072 and not args.smoke:
        raise ValueError("Nonstandard sample count requires --smoke")
    if not 1 <= args.samples <= 131072:
        raise ValueError("Samples must be between 1 and 131072")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "invocation.json", {
        "argv": sys.argv, "smoke": args.smoke, "samples": args.samples,
        "preflight_manifest_sha256": digest((args.preflight / "manifest.json").read_bytes()),
        "source_cells_planned": len(CONTROLS) * len(SEEDS) * 2 * 8 * args.samples,
        "deployment_platform": "Python3 standard library; no numpy decoder dependency", "f4_authorized": False,
    })
    start = time.perf_counter()
    rows = []
    cells_read = 0
    for kind in CONTROLS:
        for seed in SEEDS:
            if time.perf_counter() - start > args.max_seconds:
                write_json(args.output / "budget_stop.json", {"completed_rows": len(rows), "cells": cells_read})
                return
            fit = generate(kind, seed, args.samples)
            check = generate(kind, seed + 100000, args.samples)
            cells_read += fit.size + check.size
            sigma = calibration_sigma(fit)
            # Constant has no calibration scale; ratios are integrity-only labels.
            calibration = sigma if sigma is not None else 1.
            ids = list(range(8))
            for ratio in (.03, .1):
                epsilon = ratio * calibration
                fq, fp, fb = quantize(fit, epsilon)
                cq, cp, cb = quantize(check, epsilon)
                expected = reconstruct(cq, cp, cb, epsilon)
                kstart = time.perf_counter()
                k, candidates = select_support(fq, fp, fb, epsilon, ids)
                selection_seconds = time.perf_counter() - kstart
                row = {"control": kind, "seed": seed, "check_seed": seed + 100000,
                       "ratio": ratio, "sigma_cal_fit": sigma, "integrity_scale_surrogate": sigma is None,
                       "epsilon": epsilon, "cells": check.size, "k": k, "fit_support_candidates": candidates,
                       "support_selection_seconds": selection_seconds, "arms": {}}
                reference = check.astype(np.float64)
                finite = np.isfinite(reference)
                error = np.abs(reference[finite] - expected.astype(np.float64)[finite])
                row["max_finite_error"] = float(error.max(initial=0))
                row["bound_violations"] = int(np.count_nonzero(error > epsilon))
                if row["bound_violations"]:
                    raise RuntimeError("Physical bound violation")
                for mode in MODES:
                    before = time.perf_counter()
                    tables = fit_tables(fq, fp, epsilon, ids, k, mode)
                    installation = tables.dump()
                    fit_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    nll = score(tables, cq, cp)
                    score_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    transport, costs = encode(cq, cp, cb, tables)
                    encode_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    decoded, waveform = decode(transport, installation)
                    decode_seconds = time.perf_counter() - before
                    if (not np.array_equal(cq, decoded)
                            or not np.array_equal(expected.view("u4"), waveform.view("u4"))):
                        raise RuntimeError("Roundtrip mismatch")
                    before = time.perf_counter()
                    pq, px = decode_bytes(transport, installation)
                    portable_decode_seconds = time.perf_counter() - before
                    if pq != cq.astype("<i4").tobytes() or px != expected.astype("<f4").tobytes():
                        raise RuntimeError("Independent portable decoder mismatch")
                    object_bytes = portable_object(transport, installation)
                    # Persist actual measured bytes, not a theoretical byte estimate.
                    prefix = f"{kind}_{seed}_{ratio:g}_{mode}"
                    for suffix, data in (("fqs", transport), ("ftb", installation), ("pyz", object_bytes)):
                        with (args.output / f"{prefix}.{suffix}").open("xb") as stream:
                            stream.write(data)
                    row["arms"][mode] = {
                        **nll, **costs, "fit_seconds": fit_seconds, "score_seconds": score_seconds,
                        "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                        "portable_decode_seconds": portable_decode_seconds,
                        "python_stdlib_object_bytes": len(object_bytes),
                        "python_stdlib_object_sha256": digest(object_bytes),
                        "object_decoder_and_zip_overhead_bytes": len(object_bytes) - len(installation) - len(transport),
                        "transport_sha256": digest(transport), "installation_sha256": digest(installation),
                        "transport_minus_integer_nll_bits": 8 * len(transport) - nll["integer_cdf_nll_bits"],
                        "source_installation_zip_bytes": manifest["decoder_source_zip_bytes"],
                        "roundtrip_q_and_waveform_bit_exact": True,
                    }
                row["process_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                write_json(args.output / f"{kind}_{seed}_{ratio:g}.json", row)
                rows.append(row)
                print(json.dumps({"control": kind, "seed": seed, "ratio": ratio,
                                  "elapsed_seconds": time.perf_counter() - start}), flush=True)
    write_json(args.output / "results.json", {
        "smoke": args.smoke, "rows": rows, "summary": summarize(rows),
        "source_cells": cells_read, "elapsed_seconds": time.perf_counter() - start,
        "self_contained_platform": "Python3 standard library; actual executable object measured",
        "limits": ["No FEBUS score", "No F4 decision", "Python interpreter and OS not bundled",
                   "Reset degradation requires a separately versioned reset-length contrast"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=131072)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.preflight_only:
        freeze_preflight(args.preflight)
    else:
        if args.output is None or not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
            parser.error("--output and a positive finite --max-seconds are required")
        run(args)


if __name__ == "__main__":
    main()
