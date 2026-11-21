"""Stage-held-out transfer in Utah FORGE, using aligned event pairs from two wells."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib
from xml.etree import ElementTree

import numpy as np
import requests

from acquisition.public_hdf5_ranges import PublicRanges
from finetuning.run_external_egs_routing import DEPENDENCIES, calibration, paired
from finetuning.run_feasibility_fit_routing import (
    CODEC, MODES, RATIOS, ROOT, fit_arm, identity, select_mode, verify, write_json,
)
from finetuning.run_feasibility_npz_probe import choose_support, fit_windows, quantized_window


BUCKET = "https://gdr-data-lake.s3.amazonaws.com/"
BASE = "FORGE/2024_04_stimulation/GDR1824_FOGMORE_triggered_DAS_events/"
WELLS = {"16B": ("Carina22002", 3520), "78B": ("iDAS19093", 1344)}
STAGES = {"16A_4": "fit", "16A_5": "fit", "16A_8": "check", "16A_9": "check"}
FIRST, WIDTH, NS, DT_US = 500, 256, 3000, 1000
STRIDE = 240 + NS * 4
SOURCE = "finetuning/run_external_forge_routing.py"
REPORT = "progress_reports/124-EXTERNAL_DAS_ROUTING_FORGE.md"
OUT = ROOT/"results/experiments/external_forge_routing_20261119"


def event_pair(stage):
    rows = []
    for well, (unit, ntraces) in WELLS.items():
        prefix = f"{BASE}{stage}/{well}/v1.0.0/SILIXA_{unit}_"
        response = requests.get(BUCKET, params={"list-type": 2, "prefix": prefix, "max-keys": 1},
                                timeout=30)
        response.raise_for_status()
        entry = ElementTree.fromstring(response.content).find("{*}Contents")
        key = entry.findtext("{*}Key")
        size = int(entry.findtext("{*}Size"))
        if not key.startswith(prefix) or not key.endswith(".sgy") or size != 3600+ntraces*STRIDE:
            raise ValueError("Unexpected first trigger object")
        rows.append(dict(stage=stage, well=well, role=STAGES[stage], key=key, size=size,
                         tag=entry.findtext("{*}ETag"), stamp=key[len(prefix):-4]))
    if rows[0]["stamp"] != rows[1]["stamp"]:
        raise ValueError("First triggers in the two wells are not paired")
    return rows


def read_header(src, row):
    b = src._get(0, 3599)
    text = b[:3200].decode("cp500")
    cards = [text[i:i+80].strip() for i in range(0, 3200, 80)]
    clock = datetime.strptime(cards[12].split("stamp: ")[1], "%Y/%m/%d %H:%M:%S.%f (UTC)")
    if (clock.strftime("%Y%m%d%H%M%S.%f")[:-3] != row["stamp"]
            or "SC= X11600 TO NM/M/S" not in cards[13]
            or "STRAINRATE" not in cards[13]
            or struct.unpack_from(">H", b, 3212)[0] != WELLS[row["well"]][1]
            or struct.unpack_from(">H", b, 3216)[0] != DT_US
            or struct.unpack_from(">H", b, 3220)[0] != NS
            or struct.unpack_from(">H", b, 3224)[0] != 5
            or struct.unpack_from(">h", b, 3504)[0] != 0 or src.size != row["size"]):
        raise ValueError("Unsupported SEG-Y format, units or event time")
    return b, dict(gps_utc=clock.replace(tzinfo=timezone.utc).isoformat(),
                   parent=cards[5], calibration=cards[13], text_dt=cards[10],
                   header_sha256=hashlib.sha256(b).hexdigest())


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for stage in STAGES:
        pair = event_pair(stage)
        for row in pair:
            with PublicRanges(BUCKET+row["key"], etag=row["tag"]) as src:
                _, row["header"] = read_header(src, row)
        if pair[0]["header"]["gps_utc"] != pair[1]["header"]["gps_utc"]:
            raise ValueError("Event timestamps differ")
        records.extend(pair)
    if (len({r["header"]["parent"] for r in records}) != len(records)
            or len({r["stamp"][:8] for r in records}) != len(STAGES)):
        raise ValueError("Repeated parent export or stage dates are not distinct")
    deps = (*DEPENDENCIES, SOURCE)
    plan = dict(rows=records, ratios=RATIOS, first_trace_zero_based=FIRST, traces=WIDTH,
                samples=NS, sources={p: identity(ROOT/p) for p in deps},
                protocol_sha256=identity(ROOT/REPORT)["sha256"],
                attribution="Ajo-Franklin et al., Utah FORGE April 2024, GDR 1824, CC BY 4.0")
    write_json(output/"plan.json", plan)
    write_json(output/"plan.freeze.json", identity(output/"plan.json"))


def parse_block(data, first=FIRST, width=WIDTH):
    if len(data) != STRIDE * width:
        raise ValueError("Unexpected SEG-Y trace block length")
    records = np.frombuffer(data, dtype=[("hdr", "u1", (240,)), ("samples", ">f4", (NS,))])
    headers = records["hdr"].copy()
    def integers(byte, dtype):
        return headers[:, byte:byte+np.dtype(dtype).itemsize].copy().view(dtype).reshape(-1)
    if (not np.array_equal(integers(0, ">i4"), np.arange(first+1, first+width+1))
            or not np.all(integers(114, ">u2") == NS)
            or not np.all(integers(116, ">u2") == DT_US)
            or not np.all(np.diff(integers(196, ">i4")) > 0)):
        raise ValueError("Discontinuous geometry or inconsistent trace headers")
    # Endian conversion by bit operations preserves float32 payloads, even NaNs.
    raw = records["samples"].view(">u4").T.astype("<u4", order="C").view("<f4")
    return raw, headers.tobytes()


def load(row):
    with PublicRanges(BUCKET+row["key"], etag=row["tag"]) as src:
        header, meta = read_header(src, row)
        if meta != row["header"]:
            raise ValueError("Frozen source header changed")
        start = 3600 + FIRST*STRIDE
        raw, traces = parse_block(src._get(start, start+WIDTH*STRIDE-1))
    return dict(role=row["role"], drive_id=row["key"], group_id=row["well"],
                sample_path=f"{row['stage']}/{row['well']}", stage=row["stage"], dataset="SEG-Y",
                block_index=0, region_index=0, source_time_start=0, time_samples=NS,
                channel_ids=list(range(FIRST, FIRST+WIDTH)), raw=raw, geometry=zlib.compress(header+traces, 6))


def exposure(w, model, eps, path, table_hash):
    arm = fit_arm([w], model, eps,
                  "Python 3 stdlib; SEG-Y IEEE32 native amplitudes; x11600 nm/m/s; "
                  "source SEG-Y headers zlib-6 in .geometry; source indices zero-based",
                  sum((ROOT/p).stat().st_size for p in (SOURCE, CODEC)), path)
    if table_hash != arm["records"][0]["installation_sha256"]:
        raise ValueError("Table changed at evaluation")
    geom, route = Path(str(path)+".geometry"), Path(str(path)+".route")
    with geom.open("xb") as f:
        f.write(w["geometry"])
    with route.open("xb") as f:
        f.write(bytes([MODES.index(model.mode)]))
    arm["artifacts"].update({a.name: identity(a) for a in (geom, route)})
    arm["geometry_bytes"] = len(w["geometry"])
    arm["total_bytes"] += len(w["geometry"])
    return arm


def execute(output):
    started = time.monotonic()
    verify(output/"plan.json", json.loads((output/"plan.freeze.json").read_text()))
    plan = json.loads((output/"plan.json").read_text())
    for p, meta in plan["sources"].items():
        verify(ROOT/p, meta)
    write_json(output/"started.json", identity(output/"plan.json"))
    fit = [load(row) for row in plan["rows"] if row["role"] == "fit"]
    models, decisions = {}, {}
    for well in WELLS:
        subset = [w for w in fit if w["group_id"] == well]
        for ratio in RATIOS:
            tag = f"{well}_{ratio}"
            folds = []
            for i, held in enumerate(subset):
                train = [w for w in subset if w["stage"] != held["stage"]]
                eps = ratio * calibration(train)
                train = [quantized_window(w, eps) for w in train]
                k, _ = choose_support(train, eps)
                arms = {}
                for mode in MODES:
                    model = fit_windows(train, eps, k, mode)
                    arms[mode] = exposure(quantized_window(held, eps), model, eps,
                                          output/f"fold_{tag}_{i}_{mode}", hashlib.sha256(model.dump()).hexdigest())
                paired(*arms.values())
                folds.append(dict(held_out_drive_id=held["stage"], epsilon=eps, k=k, arms=arms,
                                  **{m+"_total_bytes": arms[m]["total_bytes"] for m in MODES}))
            eps = ratio * calibration(subset)
            train = [quantized_window(w, eps) for w in subset]
            k, _ = choose_support(train, eps)
            tables = {}
            for mode in MODES:
                model = fit_windows(train, eps, k, mode)
                file = output/f"precheck_{tag}_{mode}.ftb"
                with file.open("xb") as stream:
                    stream.write(model.dump())
                tables[mode] = identity(file)
                models[tag, mode] = model
            decisions[tag] = dict(folds=folds, epsilon=eps, k=k, chosen=select_mode(folds, 2), tables=tables)
    write_json(output/"decisions.json", decisions)
    write_json(output/"decisions.freeze.json", identity(output/"decisions.json"))
    hashes = {hashlib.sha256(w["raw"].tobytes()).hexdigest() for w in fit}
    if len(hashes) != len(fit):
        raise ValueError("Duplicate fit waveforms")
    results = []
    for row in plan["rows"]:
        if row["role"] != "check":
            continue
        w = load(row)
        sha = hashlib.sha256(w["raw"].tobytes()).hexdigest()
        if sha in hashes:
            raise ValueError("Repeated source clip")
        hashes.add(sha)
        for ratio in RATIOS:
            tag = f"{row['well']}_{ratio}"
            d = decisions[tag]
            window = quantized_window(w, d["epsilon"])
            arms = {}
            for mode in MODES:
                verify(output/f"precheck_{tag}_{mode}.ftb", d["tables"][mode])
                arms[mode] = exposure(window, models[tag, mode], d["epsilon"],
                                      output/f"check_{row['stage']}_{tag}_{mode}", d["tables"][mode]["sha256"])
            paired(*arms.values())
            results.append(dict(stage=row["stage"], well=row["well"], ratio=ratio, cells=w["raw"].size,
                                chosen=d["chosen"], epsilon_nanostrain_per_s=11600*d["epsilon"],
                                arms=arms, routed_bytes=arms[d["chosen"]]["total_bytes"]))
    write_json(output/"results.json", dict(rows=results, seconds=time.monotonic()-started,
               decisions=identity(output/"decisions.json"), gpu_used=False, f4_authorized=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=("freeze", "execute"))
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    (freeze if args.stage == "freeze" else execute)(args.output)
