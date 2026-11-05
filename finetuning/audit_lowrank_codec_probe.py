#!/usr/bin/env python3
"""Read-only wire, rank-budget, normalization, guard and selection verification."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from compression.lowrank_stream import factorize, compress_factors, decode_normalized, decompress
from finetuning.affine_codec_study import score_restored, temporal_distortion
from finetuning.representation_study import digest, read_manifest, load_packet, write_json
from finetuning.run_physical_guard_search import physical_admit
from finetuning.run_spatial_output_study import macro_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--verify-rank-search', action='store_true')
    args = parser.parse_args()
    root = args.experiment
    assert (root / 'COMPLETE').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    results = json.loads((root / 'results.json').read_text())
    assert results['training_updates'] == results['validation_signal_windows_read'] == 0 and not results['gpu_used']
    assert digest(args.samples_dir / 'samples_manifest.csv') == protocol['manifest_sha256']
    hashes = 0
    for path, expected in protocol['code_sha256'].items():
        assert digest(root / 'code_snapshot' / path) == expected
        hashes += 1
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    for path, expected in protocol['input_sha256'].items():
        assert path in train and digest(ROOT / path) == expected
        hashes += 1
    descriptor = json.loads((root / 'source/descriptor.json').read_text())
    assert digest(root / 'source/counts.npy') == descriptor['counts_sha256']
    codec = AlphabetStream(np.load(root / 'source/counts.npy', allow_pickle=False), descriptor['checkpoint_sha256'])
    rows = json.loads((root / 'records.json').read_text())
    assert [r['sample_path'] for r in rows] == protocol['selection']
    frames = proposals = trials = source_bytes = selected_bytes = accepted = 0
    source_losses, guarded_losses = [], []
    for i, record in enumerate(rows):
        row = train[record['sample_path']]
        raw, packet = load_packet(row)
        source_route = (root / 'source' / f'{i:05d}.route').read_bytes()
        assert source_route[0] == 0 and len(source_route) == record['source_frame_bytes'] + 1
        _, shape, _, side = codec.decode(source_route[1:])
        assert shape == raw.shape and side == packet.side_stream
        frames += 1
        feasible = [t['rank'] for t in record['rank_trace'] if t['fits']]
        assert record['rank'] == (max(feasible) if feasible else None)
        assert all(t['fits'] == (t['bytes'] <= record['source_frame_bytes']) for t in record['rank_trace'])
        if args.verify_rank_search:
            factors = factorize(packet)
            for trial in record['rank_trace']:
                frame = compress_factors(packet, *factors, trial['rank'])
                assert len(frame) == trial['bytes']
                trials += 1
        chosen, value, chosen_route = 'source', record['source_normalized_D'], source_route
        for mode, proposal in record['proposals'].items():
            frame = (root / 'candidate' / f'{i:05d}.{mode}.flr').read_bytes()
            values, side, metadata = decode_normalized(frame)
            assert side == packet.side_stream
            assert metadata['rank'] in feasible
            if not protocol.get('rank_search'):
                assert metadata['rank'] == record['rank']
            assert len(frame) == proposal['frame_bytes'] <= record['source_frame_bytes']
            restored = decompress(frame)
            np.testing.assert_array_equal(raw.view(np.uint32)[~packet.valid], restored.view(np.uint32)[~packet.valid])
            metric = score_restored(row, raw, packet, restored)
            assert metric == proposal['metrics']
            loss = temporal_distortion(restored, packet)
            assert loss == proposal['normalized_D']
            allowed = physical_admit(record['source'], metric)
            assert proposal['physical_eligible'] == allowed
            if mode.endswith('unit_rms'):
                raw_name = mode.replace('unit_rms', 'raw')
                unscaled, _, _ = decode_normalized((root / 'candidate' / f'{i:05d}.{raw_name}.flr').read_bytes())
                target_power = float(np.sum(packet.values[packet.valid].astype(np.float64) ** 2))
                predicted_power = float(np.sum(unscaled[packet.valid].astype(np.float64) ** 2))
                assert metadata['gain'] == np.sqrt(target_power / predicted_power)
            else:
                assert metadata['gain'] == 1.
            if allowed and loss < value:
                chosen, value, chosen_route = mode, loss, b'\x01' + frame
            frames += 1
            proposals += 1
        assert record['selected'] == chosen and record['guarded_normalized_D'] == value
        route = (root / 'guarded' / f'{i:05d}.route').read_bytes()
        assert route == chosen_route
        source_bytes += len(source_route)
        selected_bytes += len(route)
        accepted += chosen != 'source'
        source_losses.append((row['family'], row['configuration'], record['source_normalized_D']))
        guarded_losses.append((row['family'], row['configuration'], value))
    assert results['source_macro_normalized_D'] == macro_loss(source_losses)
    assert results['guarded_macro_normalized_D'] == macro_loss(guarded_losses)
    assert results['admitted_windows'] == accepted
    common = (root / 'transport.json').stat().st_size
    shared = sum((root / 'source' / f).stat().st_size for f in ('counts.npy', 'descriptor.json'))
    assert results['source_stream_bytes'] == source_bytes + common + shared
    assert results['guarded_stream_bytes'] == selected_bytes + common + shared + (root / 'descriptor.json').stat().st_size
    assert results['advance_gate_met'] == (results['guarded_gain'] >= .01 and results['guarded_stream_bytes'] <= results['source_stream_bytes'])
    output = {'frames_verified': frames, 'proposals_replayed': proposals, 'rank_trials_reencoded': trials,
        'routing_decisions_verified': 2 * len(rows), 'snapshot_and_input_hashes_verified': hashes,
        'admitted_windows': int(accepted), 'training_updates': 0, 'validation_signal_windows_accessed': 0,
        'advance_gate_met': results['advance_gate_met'], 'auditor_sha256': digest(Path(__file__))}
    write_json(root / 'audit.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
