#!/usr/bin/env python3
"""Verify all prepared spatial windows against raw HDF5 before model selection."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.prepare_finetune_samples import blocks, choose_dataset
from finetuning.run_sampling_experiment import check_splits
from finetuning.spatial_band_filter import POLICY, DEFAULTS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, default=ROOT/"data/finetune/raw")
    p.add_argument("--test-fold",type=int,choices=range(4),default=0)
    args = p.parse_args()
    root = args.samples_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    check_splits(root, args.test_fold)
    preparation = json.loads((root/"preprocessing.json").read_text())
    assert preparation["spatial_band_policy"]==POLICY and preparation["parameters"]==DEFAULTS
    with (root/"samples_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    maps = {m["channel_map_id"]: m for m in map(json.loads,(root/"channel_maps.jsonl").read_text().splitlines())}
    by_file = defaultdict(list)
    for row in rows:
        by_file[row["drive_id"]].append(row)
    groups = defaultdict(lambda: {"windows":0,"cells":0,"regions":set(),"files":set(),"amplitude_removed_channels":0})
    shapes = Counter()
    array_fingerprint = hashlib.sha256()
    source_hashes = {}
    for file_id, samples in sorted(by_file.items()):
        source = args.raw_dir/f"{file_id}.h5"
        source_hashes[file_id] = digest(source)
        with h5py.File(source,"r") as handle:
            _, dataset = choose_dataset(handle)
            raw = blocks(dataset)
        key = (samples[0]["split"], samples[0]["family"], samples[0]["configuration"])
        group = groups[key]
        group["files"].add(file_id)
        decision = preparation["file_decisions"][file_id]
        removed = set(decision["removed_source_channel_indices"])
        group["amplitude_removed_channels"] += len(removed)
        per_block = defaultdict(list)
        seen = set()
        for row in samples:
            path = Path(row["sample_path"])
            if not path.is_absolute():
                path = ROOT/path
            mapping = maps[row["channel_map_id"]]
            indices = np.asarray(mapping["source_channel_indices"],dtype=int)
            assert len(indices)==int(row["channels"]) and np.all(np.diff(indices)==1)
            assert not removed.intersection(indices)
            assert mapping["spatial_filter_id"]==decision["filter_id"]
            map_id = row["channel_map_id"]
            if map_id not in seen:
                per_block[int(row["block_index"])].extend(indices.tolist())
                seen.add(map_id)
            with np.load(path,allow_pickle=False) as packet:
                data = packet["data"]
                meta = json.loads(str(packet["metadata_json"].item()))
                assert data.dtype==np.float32 and np.isfinite(data).all()
                assert meta["spatial_band_policy"]==POLICY
                assert data.shape==(len(indices),int(row["time_samples"]))
                start = int(row["time_start"])
                expected = raw[int(row["block_index"]),indices,start:start+data.shape[1]]
                assert data.tobytes()==expected.tobytes(), path
                array_fingerprint.update(str(path.relative_to(ROOT)).encode())
                array_fingerprint.update(data.tobytes())
            group["windows"] += 1
            group["cells"] += data.size
            group["regions"].add(map_id)
            shapes[len(indices)] += 1
        for block, indices in per_block.items():
            expected = set(np.flatnonzero(np.isfinite(raw[block]).all(axis=1)))-removed
            assert len(indices)==len(set(indices)), "Overlapping spatial tiles"
            assert set(indices)==expected, "Missing retained spatial coverage"
        assert digest(source)==source_hashes[file_id]
        print(f"[audit] {file_id}: {len(samples)} exact windows verified",flush=True)
    output_rows = [{"split":key[0],"family":key[1],"configuration":key[2],
                    **{k:(len(v) if isinstance(v,set) else v) for k,v in group.items()}}
                   for key,group in sorted(groups.items())]
    with (args.output_dir/"coverage.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(output_rows[0]))
        writer.writeheader();writer.writerows(output_rows)
    summary = {"samples_dir":str(root),"test_fold":args.test_fold,"verified_windows":len(rows),"source_files":len(by_file),
               "splits":dict(Counter(r["split"] for r in rows)),"channels_histogram":dict(sorted(shapes.items())),
               "policy_status_counts":dict(Counter(d["status"] for d in preparation["file_decisions"].values())),
               "array_sha256":array_fingerprint.hexdigest(),"raw_sha256":source_hashes,
               "prepared_metadata_sha256":{name:digest(root/name) for name in ("samples_manifest.csv","channel_maps.jsonl","preprocessing.json")},
               "no_test_inference":True,"raw_mutation":False,
               "checks":"All window values bit-identical to source; finite; disjoint/complete contiguous spatial tiling; configured splits; filter/mask provenance."}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.output_dir/"COMPLETE").write_text("All prepared arrays and spatial maps verified; no model inference.\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("raw_sha256","channels_histogram")},indent=2),flush=True)


if __name__=="__main__":
    main()
