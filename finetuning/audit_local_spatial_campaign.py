#!/usr/bin/env python3
"""Read-only integrity/guard/budget audit for the bounded local campaign."""
import argparse
import csv
import json
from pathlib import Path
import sys

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.spatial_correction_stream import unpack_frame
from finetuning.paired_search_diagnostics import guarded_admit
from finetuning.representation_study import digest, write_json
from finetuning.run_physical_guard_search import physical_admit
from finetuning.run_index_search import FrozenIndexStream


def verify_snapshots(directory):
    assert (directory / 'COMPLETE').exists(), directory
    protocol = json.loads((directory / 'protocol.json').read_text())
    for path, expected in protocol['code_sha256'].items():
        assert digest(directory / 'code_snapshot' / path) == expected, (directory, path)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-root', type=Path, default=ROOT / 'results/experiments')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/finetune/spatial_band_v1_fold1_20261026/samples_manifest.csv')
    args = parser.parse_args()
    root = args.experiments_root
    directories = {name: root / path for name, path in {
        'correction': 'spatial_output_local_20261102',
        'boundary_coding': 'spatial_boundary_coding_local_20261102',
        'joint_vq': 'joint_vq_rate_local_20261102',
        'physical_search': 'physical_guard_search_local_20261102'}.items()}
    source_hash = digest(args.source)
    snapshot_count, framed_count, guard_checks = 0, 0, 0
    for name, directory in directories.items():
        protocol = verify_snapshots(directory)
        assert protocol['source_sha256'] == source_hash
        assert '4060' in protocol['gpu']
        snapshot_count += len(protocol['code_sha256'])
        if name != 'physical_search':
            for frame in directory.rglob('*.fsc'):
                unpack_frame(frame.read_bytes())
                framed_count += 1
    correction = directories['correction']
    r = json.loads((correction / 'results.json').read_text())
    assert r['parameter_updates'] == 2048
    protocol = json.loads((correction / 'protocol.json').read_text())
    assert not set(protocol['fit']) & set(protocol['internal_holdout'])
    with args.manifest.open() as manifest:
        metadata = {row['sample_path']: row for row in csv.DictReader(manifest)}
    fit_ids = {metadata[p]['drive_id'] for p in protocol['fit']}
    check_ids = {metadata[p]['drive_id'] for p in protocol['internal_holdout']}
    assert not fit_ids & check_ids
    for record in json.loads((correction / 'validation_records.json').read_text()):
        for arm in ('pointwise', 'spatial'):
            assert not record[arm + '_mode'] or guarded_admit(record['source'], record[arm + '_guarded'])
            guard_checks += 1
    boundary = directories['boundary_coding']
    for split in ('train', 'val'):
        for record in json.loads((boundary / split / 'records.json').read_text()):
            for alternative in record['alternatives'].values():
                assert not alternative['admitted'] or (
                    alternative['payload_bytes'] <= record['source_payload_bytes'] and guarded_admit(record['source'], alternative['metrics']))
                guard_checks += 1
    joint = directories['joint_vq']
    r = json.loads((joint / 'results.json').read_text())
    assert r['parameter_updates'] == 2048
    joint_protocol = json.loads((joint / 'protocol.json').read_text())
    assert len(joint_protocol['sequence']) == 1024
    assert not check_ids & {metadata[p]['drive_id'] for p in joint_protocol['sequence']}
    assert not check_ids & {metadata[p]['drive_id'] for p in joint_protocol['rate_calibration_samples']}
    selected = {}
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    for arm in ('control', 'rate'):
        item = torch.load(joint / f'{arm}_selected.pt', map_location='cpu', weights_only=False)
        registry = json.loads((joint / arm / 'registry.json').read_text())
        assert registry['mode_zero_model_sha256'] == source_hash
        assert registry['mode_one_model_sha256'] == digest(joint / f'{arm}_selected.pt')
        changed = {prefix: sum(not torch.equal(value, item['model_state_dict'][key])
                              for key, value in source['model_state_dict'].items() if key.startswith(prefix))
                   for prefix in ('encoder.', 'decoder.', 'quantizer.')}
        if item['step'] > 0:
            assert all(changed.values()), (arm, changed)
        selected[arm] = {'step': item['step'], 'changed_tensors': changed}
        last = torch.load(joint / f'{arm}_last.pt', map_location='cpu', weights_only=False)
        assert last['step'] == 1024
        assert last['source_sha256'] == source_hash
        for record in json.loads((joint / arm / 'records.json').read_text()):
            if record['use_candidate']:
                assert record['candidate_frame_bytes'] <= record['source_frame_bytes']
                assert guarded_admit(record['source'], record['candidate'])
            guard_checks += 1
        del item, last
    physical = directories['physical_search']
    stream = FrozenIndexStream(np.load(physical / 'counts.npy', allow_pickle=False), source_hash)
    e1_frames = 0
    for index, record in enumerate(json.loads((physical / 'records.json').read_text())):
        for arm, admit in [('standard', guarded_admit), ('physical', physical_admit)]:
            details = record[arm]['details']
            assert details['searched_frame_bytes'] <= details['baseline_frame_bytes']
            assert details['changed_fraction'] == 0 or admit(record['source'], record[arm]['metrics'])
            frame = (physical / f'{index:05d}.{arm}.frame').read_bytes()
            assert len(frame) == details['searched_frame_bytes']
            indices, shape, latent, side = stream.decode(frame)
            assert len(indices) == int(np.prod(latent)) and len(shape) == 2 and side
            assert (physical / f'{index:05d}.standard.baseline').read_bytes() == (physical / f'{index:05d}.physical.baseline').read_bytes()
            e1_frames += 1
            guard_checks += 1
    assert digest(args.source) == source_hash
    smoke_updates = sum(json.loads((root / name / 'results.json').read_text())['parameter_updates'] for name in
                        ('spatial_output_local_smoke_20261102', 'joint_vq_rate_local_smoke_20261102'))
    assert smoke_updates == 20
    result = {'snapshot_hashes_verified': snapshot_count, 'fsc_frames_crc_verified': framed_count,
        'physical_e1_frames_decoded_verified': e1_frames,
        'guard_decisions_verified': guard_checks, 'source_checkpoint_unchanged': True,
        'all_neural_training_gpus': 'NVIDIA GeForce RTX 4060', 'production_neural_updates': 4096,
        'smoke_neural_updates': smoke_updates, 'total_new_neural_updates_including_smokes': 4096 + smoke_updates,
        'joint_scope_verification': selected, 'auditor_sha256': digest(Path(__file__))}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
