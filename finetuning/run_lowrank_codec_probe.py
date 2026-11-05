#!/usr/bin/env python3
"""Train-only transmitted low-rank factors under the archived VQ frame ceilings."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from compression.lowrank_stream import factorize, compress_factors, decode_normalized, decompress, with_gain, MAX_RANK
from finetuning.affine_codec_study import score_restored, temporal_distortion
from finetuning.paired_search_diagnostics import finite_json, aggregate_metrics
from finetuning.representation_study import digest, read_manifest, load_packet, write_json
from finetuning.run_physical_guard_search import physical_admit
from finetuning.run_spatial_output_study import macro_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-probe', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--search-ranks', action='store_true')
    args = parser.parse_args()
    base, out = args.reference_probe, args.output_dir
    assert (base / 'COMPLETE').exists()
    prior = json.loads((base / 'protocol.json').read_text())
    previous = json.loads((base / 'results.json').read_text())
    assert digest(args.samples_dir / 'samples_manifest.csv') == prior['manifest_sha256']
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    rows = [train[f] for f in prior['check_train']]
    baseline = json.loads((base / 'pq2x32/records.json').read_text())
    assert [r['sample_path'] for r in baseline] == prior['check_train']
    descriptor = json.loads((base / 'source/descriptor.json').read_text())
    assert descriptor['checkpoint_sha256'] == prior['source_checkpoint_sha256']
    assert digest(base / 'source/counts.npy') == descriptor['counts_sha256']
    source_codec = AlphabetStream(np.load(base / 'source/counts.npy', allow_pickle=False), prior['source_checkpoint_sha256'])
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_lowrank_codec_probe.py', 'compression/lowrank_stream.py',
            'finetuning/affine_codec_study.py', 'finetuning/representation_study.py',
            'finetuning/run_physical_guard_search.py', 'compression/numeric_restoration.py']
    write_json(out / 'protocol.json', {'kind': 'transmitted_lowrank_train_probe_v2' if args.search_ranks else 'transmitted_lowrank_train_probe_v1', 'training_updates': 0,
        'gpu_used': False, 'validation_signal_windows_read': 0, 'reference_results_sha256': digest(base / 'results.json'),
        'manifest_sha256': prior['manifest_sha256'], 'code_sha256': {f: digest(ROOT / f) for f in code},
        'selection': [r['sample_path'] for r in rows], 'input_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in rows},
        'factorization': 'SVD of same protected matrix, no new context or coverage; singular values folded into temporal factors',
        'quantization': 'fixed signed8 per-column spatial/per-row temporal factors, float32 scales transmitted per component; no bit-depth sweep',
        'wire': 'FLR1 shape/rank/gain, source normalization/escapes, factor scales, raw or modular-delta int8 factors+zlib6, full CRC; all paid',
        'rate_control': 'enumerate ranks 0..min(64,H,W), choose largest rank whose actual frame fits source ceiling, never best waveform metric',
        'rank_search': 'nested selector may additionally choose any lower feasible rank by normalized D under unchanged per-window physical/byte guards; rank/gain explicitly transmitted, not shared model tuning' if args.search_ranks else None,
        'restorations': 'one raw reconstruction and one unit-RMS reconstruction, gain explicitly transmitted in same charged float64 field',
        'selector': 'min normalized D among proposals satisfying source per-window bytes plus unchanged RMSE/correlation/p90/FFT magnitude/phase/RMS-distance; source fallback always eligible',
        'advance_gate': '>=1% lower guarded train macro normalized D at nonincreasing total routed stream; no automatic development access',
        'limitations': ['same consumed train checks, not independent holdout', 'ideal SVD residual is diagnostic, not transmitted continuous factors or an attainable lower bound after coding',
            'SVD sees protected zeros at exact escape positions, like source encoder; no promise of optimal ordinary-mask approximation',
            'unit RMS is a competing transmitted proposal, not relaxed physical guards', 'no learned shared weights or EDAS/SMF/test signal access']})
    for f in code:
        dest = out / 'code_snapshot' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    for folder in ('source', 'candidate', 'guarded'):
        (out / folder).mkdir()
    for f in ('counts.npy', 'descriptor.json'):
        shutil.copy2(base / 'source' / f, out / 'source' / f)
    write_json(out / 'transport.json', [{k: r[k] for k in ('sample_path', 'drive_id', 'family', 'configuration', 'spatial_region_index')} for r in rows])
    write_json(out / 'descriptor.json', {'format': 'FLR1', 'version': 1, 'factor_bits': 8,
        'route': {'0': 'original FAK1', '1': 'transmitted low-rank FLR1'},
        'source_checkpoint_sha256': prior['source_checkpoint_sha256'], 'learned_lowrank_weight_bytes': 0})
    records, source_losses, raw_losses, guarded_losses, source_variable, candidate_variable = [], [], [], [], [], []
    source_size = guarded_size = candidate_size = accepted = cells = 0
    for i, (row, old) in enumerate(zip(rows, baseline)):
        raw, packet = load_packet(row)
        source_frame = (base / 'source' / f'{i:05d}.fak').read_bytes()
        _, shape, _, side = source_codec.decode(source_frame)
        assert shape == raw.shape and side == packet.side_stream and len(source_frame) == old['source_frame_bytes']
        source_loss, before = old['source_normalized_distortion'], old['source']
        source_route = b'\x00' + source_frame
        (out / 'source' / f'{i:05d}.route').write_bytes(source_route)
        u, singular, vt = factorize(packet)
        chosen_rank, feasible_frames = None, []
        trace = []
        total_energy = float(np.sum(packet.values.astype(np.float64) ** 2))
        for rank in range(min(MAX_RANK, *raw.shape) + 1):
            frame = compress_factors(packet, u, singular, vt, rank)
            fits = len(frame) <= len(source_frame)
            ideal = float(np.sqrt(max(0., 1 - np.sum(singular[:rank] ** 2) / total_energy))) if total_energy > 1e-20 else 0.
            trace.append({'rank': rank, 'bytes': len(frame), 'fits': fits, 'ideal_unquantized_matrix_relative_rmse': ideal})
            if fits:
                chosen_rank = rank, frame
                feasible_frames.append((rank, frame))
        loss, route, chosen_mode = source_loss, source_route, 'source'
        proposals = {}
        if chosen_rank is not None:
            rank, frame = chosen_rank
            (out / 'candidate' / f'{i:05d}.raw.flr').write_bytes(frame)
            candidate_size += len(frame)
            target_power = float(np.sum(packet.values[packet.valid].astype(np.float64) ** 2))
            options = []
            candidates = feasible_frames if args.search_ranks else [(rank, frame)]
            for candidate_rank, candidate_frame in candidates:
                candidate_values, _, _ = decode_normalized(candidate_frame)
                candidate_power = float(np.sum(candidate_values[packet.valid].astype(np.float64) ** 2))
                raw_name = f'r{candidate_rank:02d}_raw' if args.search_ranks else 'raw'
                rms_name = f'r{candidate_rank:02d}_unit_rms' if args.search_ranks else 'unit_rms'
                options.append((raw_name, candidate_frame))
                if target_power > 1e-20 and candidate_power > 1e-20:
                    options.append((rms_name, with_gain(candidate_frame, np.sqrt(target_power / candidate_power))))
            for mode, encoded in options:
                (out / 'candidate' / f'{i:05d}.{mode}.flr').write_bytes(encoded)
                restored = decompress(encoded)
                np.testing.assert_array_equal(raw.view(np.uint32)[~packet.valid], restored.view(np.uint32)[~packet.valid])
                metric = score_restored(row, raw, packet, restored)
                value = temporal_distortion(restored, packet)
                eligible = len(encoded) <= len(source_frame) and physical_admit(before, metric)
                proposals[mode] = {'metrics': metric, 'normalized_D': value, 'frame_bytes': len(encoded), 'physical_eligible': bool(eligible)}
                if eligible and value < loss:
                    loss, route, chosen_mode = value, b'\x01' + encoded, mode
            raw_key = f'r{rank:02d}_raw' if args.search_ranks else 'raw'
            raw_loss = proposals[raw_key]['normalized_D']
        else:
            rank, raw_loss = None, source_loss
        used = chosen_mode != 'source'
        (out / 'guarded' / f'{i:05d}.route').write_bytes(route)
        source_losses.append((row['family'], row['configuration'], source_loss))
        raw_losses.append((row['family'], row['configuration'], raw_loss))
        guarded_losses.append((row['family'], row['configuration'], loss))
        if float(np.std(packet.values[packet.valid].astype(np.float64))) > 1e-6:
            source_variable.append(before)
            if proposals:
                candidate_variable.append(proposals[raw_key]['metrics'])
        source_size += len(source_route)
        guarded_size += len(route)
        accepted += used
        cells += raw.size
        records.append({'sample_path': row['sample_path'], 'source': before, 'source_normalized_D': source_loss,
            'source_frame_bytes': len(source_frame), 'rank': rank, 'rank_trace': trace, 'proposals': proposals,
            'selected': chosen_mode, 'guarded_normalized_D': loss})
    common = (out / 'transport.json').stat().st_size
    aux = sum((out / 'source' / f).stat().st_size for f in ('counts.npy', 'descriptor.json'))
    source_total, guarded_total = source_size + common + aux, guarded_size + common + aux + (out / 'descriptor.json').stat().st_size
    assert source_total == previous['source']['stream_bytes']
    source_D, guarded_D = macro_loss(source_losses), macro_loss(guarded_losses)
    source_macro, _ = aggregate_metrics(source_variable)
    raw_macro, _ = aggregate_metrics(candidate_variable)
    result = {'training_updates': 0, 'gpu_used': False, 'validation_signal_windows_read': 0,
        'check_windows': len(rows), 'input_cells': cells, 'admitted_windows': int(accepted),
        'source_macro_normalized_D': source_D, 'raw_macro_normalized_D': macro_loss(raw_losses),
        'guarded_macro_normalized_D': guarded_D, 'guarded_gain': 1 - guarded_D / source_D,
        'source_frame_bytes': source_size - len(rows), 'candidate_raw_frame_bytes': candidate_size,
        'source_stream_bytes': source_total, 'guarded_stream_bytes': guarded_total,
        'source_shared_waveform_weight_bytes': previous['source_waveform_tensor_bytes'],
        'learned_lowrank_weight_bytes': 0, 'variable_target_windows': len(source_variable),
        'source_variable_macro_rmse': source_macro['macro_relative_rmse'], 'raw_variable_macro_rmse': raw_macro['macro_relative_rmse'],
        'diagnostic_scope': 'variable-target relative RMSE only avoids constant denominator pathology; all windows retained for bytes, guard and normalized D'}
    result['advance_gate_met'] = result['guarded_gain'] >= .01 and guarded_total <= source_total
    write_json(out / 'records.json', records)
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
