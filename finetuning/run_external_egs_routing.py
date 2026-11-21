"""External replication of the fit-only selection method on native EGS velocity."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import h5py
import numpy as np

from acquisition.public_hdf5_ranges import PublicRanges
from finetuning.run_feasibility_fit_routing import (
    CODEC, MODES, RATIOS, ROOT, fit_arm, identity, select_mode, verify, write_json,
)
from finetuning.run_feasibility_npz_probe import (
    choose_support, digest, fit_windows, quantized_window,
)


BASE = "https://gdr-data-lake.s3.amazonaws.com/egs_collab/experiment_2/DAS/Terra15/205505/"
RUNS = (
    ("A", "fit", "20220317-HMS223322.842", "CollabExp2_taps_", "235e699d9104d378c476b1c24c249bb5-240"),
    ("B", "fit", "20220318-HMS155708.284", "CollabExp2_", "954ad655116a811672beecede47d38c8-240"),
    ("C", "check", "20220322-HMS182523.983", "CollabExp2_", "2be78be127dbe9481a4ab9a6648e6ff2-240"),
    ("D", "check", "20220517-HMS060211.867", "CollabExp2_", "890f4bc0f9a709f1a48c95a2329400e5-240"),
)
STARTS = (57600, 576000)
LENGTH = 2000
WIDTH = 268
DATASET = "data_product/data"
GROUP = "EGS2/Terra15-T216125"
REPORT = "progress_reports/123-EXTERNAL_DAS_ROUTING_EGS.md"
SOURCE = "finetuning/run_external_egs_routing.py"
DEPENDENCIES = (
    SOURCE, "acquisition/public_hdf5_ranges.py", CODEC,
    "finetuning/run_feasibility_fit_routing.py", "finetuning/run_feasibility_npz_probe.py",
    "compression/feasibility_stream.py", "compression/feasibility_tables.py",
    "compression/feasibility_portable.py",
)
SUBSETS = {"Aonly": (0, 1), "Bonly": (2, 3), "A0B1": (0, 3), "B0A1": (2, 1)}
OUT = ROOT / "results/experiments/external_egs_routing_20261119"


def rows():
    records = []
    for sid, role, stamp, prefix, tag in RUNS:
        run = "UTC-YMD" + stamp
        key = f"{run}/{prefix}velocity_{run}_seq_00000000000.hdf5"
        records.append(dict(session=sid, role=role, stamp=stamp, key=key, etag=f'"{tag}"'))
    return records


def inspect(handle, row):
    attrs = handle.attrs
    ds = handle[DATASET]
    meta = {k: str(attrs[k]) for k in (
        "data_product", "data_product_units", "serial_number", "saving_start_computer_time_string",
        "file_start_computer_time_string", "gauge_length", "dx", "dt_computer", "sensing_range_start",
        "acoustic_bandwidth_end", "gps_locked", "nframes_occupied", "frame_size",
    )}
    start = datetime.fromisoformat(meta["file_start_computer_time_string"])
    if (start.strftime("%Y%m%d-HMS%H%M%S.%f")[:-3] != row["stamp"]
            or meta["saving_start_computer_time_string"] != meta["file_start_computer_time_string"]
            or tuple(meta[k] for k in ("data_product", "data_product_units", "serial_number"))
            != ("velocity", "m/s", "T216125")
            or ds.shape != (1843200, WIDTH) or ds.dtype != np.dtype("<f4")
            or ds.chunks != (5760, WIDTH) or float(meta["dt_computer"]) != 1 / 36000
            or float(meta["dx"]) != 2.4502857212913773 or float(meta["gauge_length"]) != 4.9
            or float(meta["sensing_range_start"]) != 1.8
            or int(meta["nframes_occupied"]) * int(meta["frame_size"]) < max(STARTS) + LENGTH):
        raise ValueError("Recording or physical geometry does not match preflight")
    clocks = []
    for offset in STARTS:
        ticks = handle["data_product/posix_time"][offset:offset + LENGTH]
        steps = np.diff(ticks)
        if (not np.all(np.isfinite(ticks)) or np.any(steps <= 0)
                or not np.allclose(steps, 1 / 36000, rtol=.05, atol=1e-6)
                or abs(ticks[0] - start.timestamp() - offset / 36000) > 1):
            raise ValueError("Invalid readout interval")
        clocks.append([float(ticks[0]), float(ticks[-1])])
    return {**meta, "clock_intervals": clocks}


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    records = rows()
    for row in records:
        with PublicRanges(BASE + row["key"], etag=row["etag"]) as src, h5py.File(src, "r") as f:
            row.update(metadata=inspect(f, row), object_bytes=src.size)
    plan = dict(rows=records, dataset=DATASET, starts=STARTS, length=LENGTH, width=WIDTH,
                ratios=RATIOS, subsets=SUBSETS, sources={n: identity(ROOT/n) for n in DEPENDENCIES},
                source_credit="Rodriguez Tribaldos & Hopp, EGS Collab Experiment 2 DAS, GDR 1606, CC BY 4.0",
                report_sha256=identity(ROOT/REPORT)["sha256"])
    write_json(output/"plan.json", plan)
    write_json(output/"plan.freeze.json", identity(output/"plan.json"))


def load(row):
    windows = []
    with PublicRanges(BASE + row["key"], etag=row["etag"]) as src, h5py.File(src, "r") as f:
        if src.size != row["object_bytes"] or inspect(f, row) != row["metadata"]:
            raise ValueError("Frozen header changed")
        for start in STARTS:
            values = f[DATASET][start:start+LENGTH, :]
            if values.shape != (LENGTH, WIDTH) or values.dtype != np.dtype("<f4"):
                raise ValueError("Unexpected waveform")
            windows.append(dict(role=row["role"], drive_id=row["key"], group_id=GROUP,
                                session=row["session"], channel_ids=list(range(WIDTH)), dataset=DATASET,
                                block_index=0, region_index=0, source_time_start=start, time_samples=LENGTH,
                                sample_path=f"{row['session']}/t{start}", raw=values))
    return windows


def calibration(train):
    if not train or any(w["role"] != "fit" for w in train):
        raise ValueError("Calibration requires fit sessions only")
    # Positive finite channel standard deviations; check or held-out run has no vote.
    stds = [np.std(x[np.isfinite(x)], dtype=np.float64) for w in train
            for x in w["raw"].T if np.any(np.isfinite(x))]
    return float(np.median([v for v in stds if v > 0 and np.isfinite(v)]))


def paired(left, right):
    keys = ("sample_path", "q_sha256", "original_float32_sha256", "waveform_sha256")
    a, b = [[tuple(r[k] for k in keys) for r in arm["records"]] for arm in (left, right)]
    if not a or a != b:
        raise ValueError("Unpaired reconstruction")


def measure(windows, model, epsilon, prefix, table_sha):
    # Include the acquisition contract in the delivered installation.
    platform = dict(runtime="Python 3 stdlib with lzma", product="velocity", units="m/s",
                    dx_m=2.4502857212913773, dt_s=1/36000, gauge_m=4.9,
                    range_start_m=1.8, data_source=BASE, coordinate="native source-channel order",
                    session_horizon="two separate 2000-sample regions, no continuity across them")
    source_bytes = sum((ROOT/p).stat().st_size for p in (CODEC, SOURCE))
    arm = fit_arm(windows, model, epsilon, platform, source_bytes, prefix)
    # Materialize the one-byte ID counted by fit_arm on either route.
    route = Path(str(prefix) + ".route")
    with route.open("xb") as stream:
        stream.write(bytes([MODES.index(model.mode)]))
    arm["artifacts"][route.name] = identity(route)
    if any(r["installation_sha256"] != table_sha for r in arm["records"]):
        raise ValueError("Delivered probability tables changed")
    return arm


def execute(output):
    start = time.monotonic()
    verify(output/"plan.json", json.loads((output/"plan.freeze.json").read_text()))
    plan = json.loads((output/"plan.json").read_text())
    for name, expected in plan["sources"].items():
        verify(ROOT/name, expected)
    with (output/"started.json").open("x") as f:
        json.dump({"plan": identity(output/"plan.json")}, f)
    train = [w for row in plan["rows"] if row["role"] == "fit" for w in load(row)]
    if [(w["session"], w["source_time_start"]) for w in train] != [
            (s, t) for s in ("A", "B") for t in STARTS]:
        raise ValueError("Unexpected fit assignments")
    origins = {w["sample_path"]: digest(w["raw"].tobytes()) for w in train}
    models, decisions = {}, {}
    for ratio in RATIOS:
        tag = str(ratio)
        folds = []
        for sid in ("A", "B"):
            other = [w for w in train if w["session"] != sid]
            epsilon = ratio * calibration(other)
            fit = [quantized_window(w, epsilon) for w in other]
            held = [quantized_window(w, epsilon) for w in train if w["session"] == sid]
            k, _ = choose_support(fit, epsilon)
            arms = {}
            for mode in MODES:
                model = fit_windows(fit, epsilon, k, mode)
                arms[mode] = measure(held, model, epsilon, output/f"fold_{sid}_{tag}_{mode}",
                                     digest(model.dump()))
            paired(*arms.values())
            folds.append({"held_out_drive_id": sid, "epsilon": epsilon, "k": k,
                          **{m+"_total_bytes": arm["total_bytes"] for m, arm in arms.items()},
                          "arms": arms})
        chosen = select_mode(folds, 2)
        epsilon = ratio * calibration(train)
        fit = [quantized_window(w, epsilon) for w in train]
        k, _ = choose_support(fit, epsilon)
        specs = {m: (fit, m) for m in MODES}
        specs.update({n: ([fit[i] for i in ids], "spatiotemporal") for n, ids in SUBSETS.items()})
        hashes = {}
        for label, (sample, mode) in specs.items():
            model = fit_windows(sample, epsilon, k, mode)
            models[tag, label] = model
            table = output/f"precheck_{tag}_{label}.ftb"
            with table.open("xb") as stream:
                stream.write(model.dump())
            hashes[label] = identity(table)
        decisions[tag] = dict(chosen=chosen, epsilon=epsilon, k=k, folds=folds, tables=hashes)
    write_json(output/"decisions.json", {"fit_sha256": origins, "ratios": decisions})
    write_json(output/"decisions.freeze.json", identity(output/"decisions.json"))
    verify(output/"decisions.json", json.loads((output/"decisions.freeze.json").read_text()))
    results = []
    for row in plan["rows"]:
        if row["role"] != "check":
            continue
        check = load(row)
        for w in check:
            sha = digest(w["raw"].tobytes())
            if sha in origins.values():
                raise ValueError("Duplicate waveform across source clips")
            origins[w["sample_path"]] = sha
        for ratio in RATIOS:
            tag = str(ratio)
            d = decisions[tag]
            windows = [quantized_window(w, d["epsilon"]) for w in check]
            arms = {}
            for label, meta in d["tables"].items():
                verify(output/f"precheck_{tag}_{label}.ftb", meta)
                arms[label] = measure(windows, models[tag, label], d["epsilon"],
                                      output/f"check_{row['session']}_{tag}_{label}", meta["sha256"])
                paired(arms["marginal"], arms[label])
            b = {label: arm["total_bytes"] for label, arm in arms.items()}
            single = (b["Aonly"] + b["Bonly"])/2
            mixed = (b["A0B1"] + b["B0A1"])/2
            results.append(dict(session=row["session"], ratio=ratio, epsilon=d["epsilon"], bytes=b,
                                routed=b[d["chosen"]], chosen=d["chosen"], cells=sum(w["raw"].size for w in check),
                                mixed_vs_single_saving=1-mixed/single, arms=arms))
    write_json(output/"results.json", dict(rows=results, sha256=origins,
               decisions=identity(output/"decisions.json"), seconds=time.monotonic()-start,
               gpu_used=False, f4_authorized=False, independent_febus_confirmation=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=("freeze", "execute"))
    cli.add_argument("--output", type=Path, default=OUT)
    args = cli.parse_args()
    (freeze if args.stage == "freeze" else execute)(args.output)
