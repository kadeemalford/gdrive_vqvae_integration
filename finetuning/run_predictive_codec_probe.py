#!/usr/bin/env python3
"""CPU train-only predictive rate control, plus explicitly exploratory precision points."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from compression.predictive_stream import compress_packet, decode_normalized, decompress
from finetuning.affine_codec_study import score_restored, temporal_distortion
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, read_manifest, load_packet, write_json
from finetuning.run_physical_guard_search import physical_admit
from finetuning.run_spatial_output_study import macro_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-probe', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--refine-rate', action='store_true')
    args = parser.parse_args()
    base, out = args.reference_probe, args.output_dir
    assert (base / 'COMPLETE').exists()
    prior = json.loads((base / 'protocol.json').read_text())
    previous_results = json.loads((base / 'results.json').read_text())
    assert digest(args.samples_dir / 'samples_manifest.csv') == prior['manifest_sha256']
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    rows = [train[f] for f in prior['check_train']]
    original_records = json.loads((base / 'pq2x32/records.json').read_text())
    assert [r['sample_path'] for r in original_records] == prior['check_train']
    source_descriptor = json.loads((base / 'source/descriptor.json').read_text())
    assert digest(base / 'source/counts.npy') == source_descriptor['counts_sha256']
    assert source_descriptor['checkpoint_sha256'] == prior['source_checkpoint_sha256']
    source_codec = AlphabetStream(np.load(base / 'source/counts.npy', allow_pickle=False), prior['source_checkpoint_sha256'])
    precisions = [2. ** (-k) for k in range(-4, 17)]
    bound_points = (.1, .01, .001)
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_predictive_codec_probe.py', 'compression/predictive_stream.py',
            'finetuning/affine_codec_study.py', 'finetuning/run_physical_guard_search.py',
            'finetuning/representation_study.py', 'compression/numeric_restoration.py']
    write_json(out / 'protocol.json', {'kind': 'predictive_train_byte_matched_v2' if args.refine_rate else 'predictive_train_byte_matched_v1', 'training_updates': 0,
        'gpu_used': False, 'validation_signal_windows_read': 0, 'reference_results_sha256': digest(base / 'results.json'),
        'code_sha256': {f: digest(ROOT / f) for f in code}, 'manifest_sha256': prior['manifest_sha256'],
        'selection': [r['sample_path'] for r in rows], 'input_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in rows},
        'wire': 'FPI1, input side/escapes, scalar step, integer identity/temporal/Lorenzo, byte planes+zlib6, shape and full CRC',
        'rate_control': 'enumerate fixed epsilon powers 2^4 through 2^-16; choose smallest epsilon whose measured frame fits VQ frame ceiling, NOT best waveform error; do not assume monotonic byte sizes',
        'refinement': '24 geometric-midpoint byte-only bisection proposals within the finest observed feasible / next finer infeasible grid bracket; retain finest observed feasible, not a global monotonicity or optimality claim' if args.refine_rate else None,
        'precision_grid': precisions, 'predictor_control': 'shortest frame for SAME quantized grid; no quality-based predictor choice',
        'bound_points': list(bound_points), 'bound_semantics': 'absolute bound in protected normalized float32 coordinates, plus measured float32 rounding; NOT supplied application tolerances',
        'guard': 'unchanged per-window bytes/RMSE/correlation/channel-p90/FFT magnitude/phase/RMS-distance plus lower normalized D; same one-byte routing cost for source/fallback',
        'advance_gate': '>=1% lower guarded train macro normalized D at nonincreasing routed stream, prior to any development confirmation',
        'shared_cost': 'predictive codec has no learned tables/weights; hybrid retains the source model/dictionary, outer descriptor and transport charged',
        'limitations': ['same consumed train checks, not independent holdout',
            'fixed-precision points are DIFFERENT rates and not operational acceptance or a lower bound',
            'existing corpus coverage and escape mask unchanged; no new external/test signal access',
            'normalized source D is archived float32 evaluation, candidate D is measured after physical restoration']})
    for f in code:
        dest = out / 'code_snapshot' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    for folder in ('source', 'matched', 'epsilon_0.1', 'epsilon_0.01', 'epsilon_0.001'):
        (out / folder).mkdir()
    for f in ('counts.npy', 'descriptor.json'):
        shutil.copy2(base / 'source' / f, out / 'source' / f)
    write_json(out / 'descriptor.json', {'format': 'FPI1', 'version': 1,
        'source_checkpoint_sha256': prior['source_checkpoint_sha256'],
        'route': {'0': 'source FAK1 unchanged', '1': 'predictive FPI1'},
        'trained_predictive_weight_bytes': 0, 'learned_predictive_table_bytes': 0})
    write_json(out / 'transport.json', [{k: r[k] for k in ('sample_path', 'drive_id', 'family', 'configuration', 'spatial_region_index')} for r in rows])
    records, precision_records, guarded_losses, source_losses = [], [], [], []
    metrics = {p: [] for p in bound_points}
    sizes = {p: 0 for p in bound_points}
    source_size = guarded_size = matched_size = accepted = cells = 0
    for i, (row, baseline) in enumerate(zip(rows, original_records)):
        raw, packet = load_packet(row)
        source_frame = (base / 'source' / f'{i:05d}.fak').read_bytes()
        _, shape, _, side = source_codec.decode(source_frame)
        assert shape == raw.shape and side == packet.side_stream
        assert len(source_frame) == baseline['source_frame_bytes']
        before, source_loss = baseline['source'], baseline['source_normalized_distortion']
        (out / 'source' / f'{i:05d}.route').write_bytes(b'\x00' + source_frame)
        trace, selected = [], None
        for epsilon in precisions:
            frame, stats = compress_packet(packet, epsilon)
            feasible = len(frame) <= len(source_frame)
            trace.append({'epsilon': epsilon, 'bytes': len(frame), 'feasible': feasible, 'predictor': stats['predictor']})
            if feasible and (selected is None or epsilon < selected[0]):
                selected = epsilon, frame, stats
        if args.refine_rate and selected is not None:
            finer_failed = [t['epsilon'] for t in trace if t['epsilon'] < selected[0] and not t['feasible']]
            if finer_failed:
                lo, hi = max(finer_failed), selected[0]
                for _ in range(24):
                    epsilon = float(np.sqrt(lo * hi))
                    frame, stats = compress_packet(packet, epsilon)
                    feasible = len(frame) <= len(source_frame)
                    trace.append({'epsilon': epsilon, 'bytes': len(frame), 'feasible': feasible,
                                  'predictor': stats['predictor'], 'refinement': True})
                    if feasible:
                        hi = epsilon
                        if epsilon < selected[0]:
                            selected = epsilon, frame, stats
                    else:
                        lo = epsilon
        candidate = None
        if selected is not None:
            epsilon, frame, stats = selected
            restored = decompress(frame)
            after = score_restored(row, raw, packet, restored)
            value = temporal_distortion(restored, packet)
            admitted = physical_admit(before, after) and value < source_loss
            chosen = frame if admitted else source_frame
            (out / 'matched' / f'{i:05d}.candidate.fpi').write_bytes(frame)
            candidate = {'metrics': after, 'normalized_distortion': value, 'epsilon': epsilon,
                         'frame_bytes': len(frame), 'predictor': stats['predictor']}
            matched_size += len(frame)
        else:
            admitted, chosen, value = False, source_frame, source_loss
        (out / 'matched' / f'{i:05d}.guarded.route').write_bytes(bytes([int(admitted)]) + chosen)
        guarded_loss = value if admitted else source_loss
        guarded_losses.append((row['family'], row['configuration'], guarded_loss))
        source_losses.append((row['family'], row['configuration'], source_loss))
        source_size += len(source_frame) + 1
        guarded_size += len(chosen) + 1
        accepted += admitted
        records.append({'sample_path': row['sample_path'], 'source': before, 'source_normalized_distortion': source_loss,
                        'source_frame_bytes': len(source_frame), 'candidate': candidate, 'admitted': bool(admitted),
                        'guarded_normalized_distortion': guarded_loss, 'rate_trace': trace})
        for epsilon in bound_points:
            frame, stats = compress_packet(packet, epsilon)
            normalized, side, _ = decode_normalized(frame)
            error = float(np.max(np.abs(normalized[packet.valid].astype(np.float64) - packet.values[packet.valid])))
            margin = float(np.finfo(np.float32).eps * max(1., float(np.abs(packet.values).max())))
            assert error <= epsilon + margin
            restored = decompress(frame)
            np.testing.assert_array_equal(raw.view(np.uint32)[~packet.valid], restored.view(np.uint32)[~packet.valid])
            measured = score_restored(row, raw, packet, restored)
            metrics[epsilon].append(measured)
            sizes[epsilon] += len(frame)
            label = f'epsilon_{epsilon}'
            (out / label / f'{i:05d}.fpi').write_bytes(frame)
            precision_records.append({'sample_path': row['sample_path'], 'epsilon': epsilon, 'metrics': measured,
                'ordinary_max_normalized_error': error, 'float32_margin': margin, 'frame_bytes': len(frame),
                'predictor': stats['predictor'], 'predictor_bytes': stats['predictor_bytes'], 'input_cells': raw.size})
        cells += raw.size
    source_aux = sum((out / 'source' / f).stat().st_size for f in ('counts.npy', 'descriptor.json'))
    common = (out / 'transport.json').stat().st_size
    descriptor_bytes = (out / 'descriptor.json').stat().st_size
    before_d, after_d = macro_loss(source_losses), macro_loss(guarded_losses)
    source_stream, guarded_stream = source_size + source_aux + common, guarded_size + source_aux + common + descriptor_bytes
    assert source_stream == previous_results['source']['stream_bytes']
    result = {'training_updates': 0, 'gpu_used': False, 'validation_signal_windows_read': 0,
        'check_windows': len(rows), 'input_cells': cells, 'source_macro_normalized_D': before_d,
        'guarded_macro_normalized_D': after_d, 'guarded_gain': 1 - after_d / before_d,
        'admitted_windows': int(accepted), 'source_stream_bytes': source_stream, 'guarded_stream_bytes': guarded_stream,
        'source_frame_bytes': source_size - len(rows), 'matched_candidate_frame_bytes': matched_size,
        'matched_candidate_windows': sum(r['candidate'] is not None for r in records),
        'source_shared_waveform_weight_bytes': previous_results['source_waveform_tensor_bytes'], 'precision_points': {}}
    for epsilon in bound_points:
        macro, groups = aggregate_metrics(metrics[epsilon])
        stream_size = sizes[epsilon] + common + descriptor_bytes
        result['precision_points'][str(epsilon)] = {'macro_diagnostic_only': macro, 'groups': groups,
            'frame_bytes': sizes[epsilon], 'stream_bytes': stream_size, 'bits_per_cell': 8 * stream_size / cells,
            'ratio_vs_float32': 4 * cells / stream_size, 'trained_weight_bytes': 0}
    result['advance_gate_met'] = result['guarded_gain'] >= .01 and guarded_stream <= source_stream
    write_json(out / 'records.json', records)
    write_json(out / 'precision_records.json', precision_records)
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({k: v for k, v in result.items() if k != 'precision_points'}), flush=True)
    print(json.dumps({k: {'bytes': v['stream_bytes'], 'bpc': v['bits_per_cell'], 'ratio': v['ratio_vs_float32']}
                      for k, v in result['precision_points'].items()}), flush=True)


if __name__ == '__main__':
    main()
