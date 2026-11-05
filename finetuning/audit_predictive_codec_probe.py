#!/usr/bin/env python3
"""Audit predictive frames, precision bounds, byte-only choice and unchanged guards."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.predictive_stream import decode_normalized, decompress, compress_packet
from finetuning.affine_codec_study import score_restored, temporal_distortion
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, read_manifest, load_packet, write_json
from finetuning.run_physical_guard_search import physical_admit
from finetuning.run_spatial_output_study import macro_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--verify-rate-search', action='store_true')
    args = parser.parse_args()
    root = args.experiment
    assert (root / 'COMPLETE').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    result = json.loads((root / 'results.json').read_text())
    assert result['training_updates'] == result['validation_signal_windows_read'] == 0 and not result['gpu_used']
    hashes = 0
    for f, expected in protocol['code_sha256'].items():
        assert digest(root / 'code_snapshot' / f) == expected
        hashes += 1
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    packets = {}
    for f, expected in protocol['input_sha256'].items():
        assert f in train and digest(ROOT / f) == expected
        packets[f] = load_packet(train[f])
        hashes += 1
    rows = json.loads((root / 'records.json').read_text())
    assert [r['sample_path'] for r in rows] == protocol['selection']
    frames = routes = admitted = trials = 0
    source_bytes = guarded_bytes = 0
    guarded_loss, source_loss, matched_loss = [], [], []
    variable_source, variable_matched = [], []
    variable_paths = set()
    for i, record in enumerate(rows):
        path = record['sample_path']
        row = train[path]
        raw, packet = packets[path]
        source = (root / 'source' / f'{i:05d}.route').read_bytes()
        assert source[0] == 0 and len(source) == record['source_frame_bytes'] + 1
        routes += 1
        candidate = record['candidate']
        if args.verify_rate_search:
            for trial in record['rate_trace']:
                actual, metadata = compress_packet(packet, trial['epsilon'])
                assert len(actual) == trial['bytes'] and metadata['predictor'] == trial['predictor']
                trials += 1
        if candidate is None:
            assert not any(t['feasible'] for t in record['rate_trace'])
            allowed = False
            loss = record['source_normalized_distortion']
        else:
            best = min(t['epsilon'] for t in record['rate_trace'] if t['feasible'])
            assert candidate['epsilon'] == best
            assert all(t['feasible'] == (t['bytes'] <= record['source_frame_bytes']) for t in record['rate_trace'])
            frame = (root / 'matched' / f'{i:05d}.candidate.fpi').read_bytes()
            assert len(frame) == candidate['frame_bytes'] <= record['source_frame_bytes']
            normalized, side, metadata = decode_normalized(frame)
            assert side == packet.side_stream and metadata['normalized_epsilon'] == best
            max_error = float(np.max(np.abs(normalized[packet.valid].astype(np.float64) - packet.values[packet.valid])))
            margin = np.finfo(np.float32).eps * max(1., float(np.abs(packet.values).max()))
            assert max_error <= best + margin
            restored = decompress(frame)
            assert score_restored(row, raw, packet, restored) == candidate['metrics']
            loss = temporal_distortion(restored, packet)
            assert loss == candidate['normalized_distortion']
            allowed = physical_admit(record['source'], candidate['metrics']) and loss < record['source_normalized_distortion']
            frames += 1
            matched_loss.append((row['family'], row['configuration'], loss))
        route = (root / 'matched' / f'{i:05d}.guarded.route').read_bytes()
        assert allowed == record['admitted']
        assert route == (b'\x01' + frame if allowed else source)
        routes += 1
        selected_loss = loss if allowed else record['source_normalized_distortion']
        assert selected_loss == record['guarded_normalized_distortion']
        guarded_loss.append((row['family'], row['configuration'], selected_loss))
        source_loss.append((row['family'], row['configuration'], record['source_normalized_distortion']))
        admitted += allowed
        source_bytes += len(source)
        guarded_bytes += len(route)
        if float(np.std(packet.values[packet.valid].astype(np.float64))) > 1e-6:
            variable_paths.add(path)
            variable_source.append(record['source'])
            if candidate is not None:
                variable_matched.append(candidate['metrics'])
    precision_records = json.loads((root / 'precision_records.json').read_text())
    precision = {}
    for record in precision_records:
        path, epsilon = record['sample_path'], record['epsilon']
        raw, packet = packets[path]
        i = protocol['selection'].index(path)
        frame = (root / f'epsilon_{epsilon}' / f'{i:05d}.fpi').read_bytes()
        assert len(frame) == record['frame_bytes']
        values, side, metadata = decode_normalized(frame)
        assert side == packet.side_stream and metadata['normalized_epsilon'] == epsilon
        max_error = float(np.max(np.abs(values[packet.valid].astype(np.float64) - packet.values[packet.valid])))
        assert max_error == record['ordinary_max_normalized_error'] <= epsilon + record['float32_margin']
        restored = decompress(frame)
        np.testing.assert_array_equal(raw.view(np.uint32)[~packet.valid], restored.view(np.uint32)[~packet.valid])
        assert score_restored(train[path], raw, packet, restored) == record['metrics']
        entry = precision.setdefault(str(epsilon), {'losses': [], 'variable': [], 'max_error': 0.})
        row = train[path]
        entry['losses'].append((row['family'], row['configuration'], temporal_distortion(restored, packet)))
        entry['max_error'] = max(entry['max_error'], max_error)
        if path in variable_paths:
            entry['variable'].append(record['metrics'])
        frames += 1
    assert len(precision_records) == len(rows) * len(protocol['bound_points'])
    assert macro_loss(source_loss) == result['source_macro_normalized_D']
    assert macro_loss(guarded_loss) == result['guarded_macro_normalized_D']
    assert admitted == result['admitted_windows']
    common = (root / 'transport.json').stat().st_size
    source_aux = sum((root / 'source' / f).stat().st_size for f in ('counts.npy', 'descriptor.json'))
    assert result['source_stream_bytes'] == source_bytes + source_aux + common
    assert result['guarded_stream_bytes'] == guarded_bytes + source_aux + common + (root / 'descriptor.json').stat().st_size
    diagnostics = {}
    for epsilon, entry in precision.items():
        macro, _ = aggregate_metrics(entry['variable'])
        diagnostics[epsilon] = {'macro_normalized_D': macro_loss(entry['losses']),
            'variable_target_macro_relative_rmse': macro['macro_relative_rmse'],
            'variable_target_macro_correlation': macro['macro_correlation'],
            'variable_target_macro_fft_phase_alignment': macro['macro_fft_phase_alignment'],
            'ordinary_max_normalized_error': entry['max_error']}
    source_macro, _ = aggregate_metrics(variable_source)
    matched_macro, _ = aggregate_metrics(variable_matched)
    assert result['advance_gate_met'] == (result['guarded_gain'] >= .01 and result['guarded_stream_bytes'] <= result['source_stream_bytes'])
    output = {'frames_verified': frames, 'rate_search_trials_reencoded': trials,
        'routing_decisions_verified': routes, 'snapshot_and_input_hashes_verified': hashes,
        'source_macro_normalized_D': result['source_macro_normalized_D'],
        'matched_macro_normalized_D': macro_loss(matched_loss), 'guarded_macro_normalized_D': result['guarded_macro_normalized_D'],
        'variable_target_windows': len(variable_paths), 'source_variable_macro_rmse': source_macro['macro_relative_rmse'],
        'matched_variable_macro_rmse': matched_macro['macro_relative_rmse'],
        'precision_diagnostics': diagnostics, 'training_updates': 0, 'validation_signal_windows_accessed': 0,
        'diagnostic_limitations': 'variable-target subset only avoids constant-tile relative-denominator pathology; NO coverage or primary guard/rate change',
        'auditor_sha256': digest(Path(__file__))}
    write_json(root / 'audit.json', finite_json(output))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
