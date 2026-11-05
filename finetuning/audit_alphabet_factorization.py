#!/usr/bin/env python3
"""Read-only lineage, budget, wire, guard and per-group audit for the alphabet pilot."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from finetuning.latent_geometry import latent_positions
from finetuning.representation_study import digest, write_json
from finetuning.run_physical_guard_search import physical_admit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    root = args.experiment
    assert (root / 'COMPLETE').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    results = json.loads((root / 'results.json').read_text())
    assert digest(args.source) == protocol['source_sha256']
    assert '4090' in protocol['gpu']
    verified_hashes = 0
    for f, expected in protocol['code_sha256'].items():
        assert digest(root / 'code_snapshot' / f) == expected
        verified_hashes += 1
    for f, expected in protocol['vendor_sha256'].items():
        assert digest(root / 'code_snapshot/vendor/vqvae' / f) == expected
        verified_hashes += 1
    for f, expected in protocol['input_sha256'].items():
        assert digest(ROOT / f) == expected
        verified_hashes += 1
    assert len(protocol['sequence']) == protocol['budget_per_arm']
    assert not set(protocol['sequence']) & set(protocol['internal_train_check'])
    assert not set(protocol['calibration_train']) & set(protocol['internal_train_check'])
    assert not set(protocol['validation']) & (set(protocol['sequence']) | set(protocol['calibration_train']) | set(protocol['internal_train_check']))
    # File identities, not only tile paths, must be disjoint for the internal check.
    import csv
    paths = {r['sample_path']: r for r in csv.DictReader((ROOT / 'data/finetune/spatial_band_v1_fold1_20261026/samples_manifest.csv').open())}
    check_ids = {paths[f]['drive_id'] for f in protocol['internal_train_check']}
    assert not check_ids & {paths[f]['drive_id'] for f in [*protocol['sequence'], *protocol['calibration_train']]}
    assert all(paths[f]['split'] == 'train' for f in [*protocol['sequence'], *protocol['calibration_train'], *protocol['internal_train_check']])
    assert all(paths[f]['split'] == 'val' for f in protocol['validation'])
    records = {arm: json.loads((root / arm / 'records.json').read_text()) for arm in protocol['arms']}
    frames_verified = 0
    output = {'smoke_only': protocol['smoke_only'], 'gpu': protocol['gpu'], 'arms': {}}
    for arm, (geometry, k) in protocol['arms'].items():
        folder = root / arm
        descriptor = json.loads((folder / 'descriptor.json').read_text())
        assert descriptor['checkpoint_sha256'] == digest(folder / 'selected.pt')
        assert descriptor['counts_sha256'] == digest(folder / 'counts.npy')
        stream = AlphabetStream(np.load(folder / 'counts.npy', allow_pickle=False), descriptor['checkpoint_sha256'])
        assert k == stream.k and descriptor['identifier'] == stream.identifier.hex()
        history = json.loads((folder / 'history.json').read_text())
        selected = min(history, key=lambda r: r['train_check_distortion'])
        assert results['arms'][arm]['selected_step'] == selected['step']
        assert history[-1]['step'] == results['arms'][arm]['updates'] == protocol['budget_per_arm']
        nbytes = routed_bytes = index_count = nominal_bits = accepted = 0
        for i, record in enumerate(records[arm]):
            assert record['sample_path'] == protocol['validation'][i]
            source = records['compact2048'][i]
            assert record['source'] == source['metrics']
            frame = (folder / f'{i:05d}.candidate.fak').read_bytes()
            indices, shape, latent, side = stream.decode(frame)
            frames_verified += 1
            expected_positions = latent_positions(SimpleNamespace(downsample_levels=4, latent_geometry=geometry), *shape)
            assert len(indices) == record['positions'] == expected_positions == int(np.prod(latent))
            assert k == descriptor['config']['num_embeddings']
            assert record['nominal_fixed_bits'] == expected_positions * stream.bits
            assert len(frame) == record['frame_bytes']
            old_frame = (root / 'compact2048' / f'{i:05d}.candidate.fak').read_bytes()
            assert len(old_frame) == record['source_frame_bytes']
            eligible = len(frame) <= len(old_frame) and (arm == 'compact2048' or physical_admit(record['source'], record['metrics']))
            assert record['admitted'] == eligible
            route = (folder / f'{i:05d}.guarded.route').read_bytes()
            assert route == bytes([int(eligible)]) + (frame if eligible else old_frame)
            assert record['guarded'] == (record['metrics'] if eligible else record['source'])
            nbytes += len(frame)
            routed_bytes += len(route)
            index_count += expected_positions
            nominal_bits += expected_positions * stream.bits
            accepted += eligible
        assert nbytes == results['arms'][arm]['candidate_frame_bytes']
        assert routed_bytes == results['arms'][arm]['guarded_routed_frame_bytes']
        regressions = {}
        controls = {(g['family'], g['configuration']): g for g in results['arms']['compact2048']['groups']}
        for values in results['arms'][arm]['groups']:
            key = (values['family'], values['configuration'])
            before = controls[key]
            guarded = next(g for g in results['arms'][arm]['guarded_groups']
                           if (g['family'], g['configuration']) == key)
            regressions['/'.join(key)] = {
                'raw_relative_rmse_change': values['relative_rmse'] / before['relative_rmse'] - 1,
                'guarded_relative_rmse_change': guarded['relative_rmse'] / before['relative_rmse'] - 1,
                'raw_channel_p90_change': values['ordinary_channel_p90_relative_rmse'] - before['ordinary_channel_p90_relative_rmse'],
                'guarded_channel_p90_change': guarded['ordinary_channel_p90_relative_rmse'] - before['ordinary_channel_p90_relative_rmse']}
        output['arms'][arm] = {'frames': len(records[arm]), 'accepted': accepted,
            'index_positions': index_count, 'nominal_bits': nominal_bits,
            'frame_bytes': nbytes, 'route_bytes': routed_bytes, 'group_raw_rmse': regressions}
    assert results['parameter_updates'] == len(protocol['arms']) * protocol['budget_per_arm']
    output.update({'snapshot_and_input_hashes_verified': verified_hashes, 'candidate_frames_verified': frames_verified,
        'route_decisions_verified': frames_verified, 'neural_updates_verified': results['parameter_updates'],
        'auditor_sha256': digest(Path(__file__)), 'source_weights_unchanged': True})
    write_json(root / 'audit.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
