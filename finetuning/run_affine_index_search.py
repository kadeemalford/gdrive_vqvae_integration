#!/usr/bin/env python3
"""One preregistered fresh-development comparison after the train-only affine probe."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.affine_index_search import paired_search
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, load_packet, pilot_rows, read_manifest, write_json
from finetuning.run_index_search import FrozenIndexStream


def excluded_windows(protocol):
    selection = protocol.get('selection')
    paths = protocol.get('validation', selection.get('val', []) if isinstance(selection, dict) else [])
    return set(paths) | set(protocol.get('excluded_validation_windows', []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--search-calibration', type=Path, required=True)
    parser.add_argument('--exclude-protocol', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bounded-ols', action='store_true')
    args = parser.parse_args()
    assert (args.probe_dir / 'COMPLETE').exists() and json.loads((args.probe_dir / 'results.json').read_text())['advance_rule_met']
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    seed = 2089 if args.bounded_ols else 2087
    if args.bounded_ols:
        assert json.loads((args.probe_dir / 'protocol.json').read_text())['bounded_ols']
        assert json.loads((args.probe_dir / 'results.json').read_text())['gain_vs_two_affine_proposals'] > 0
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    excluded = set()
    for p in args.exclude_protocol:
        value = json.loads(p.read_text())
        excluded.update(excluded_windows(value))
    rows = pilot_rows([r for r in read_manifest(args.samples_dir)['val'] if r['sample_path'] not in excluded], seed=seed)
    source_hash = digest(args.source)
    calibration = json.loads((args.calibration_dir / 'calibration.json').read_text())
    assert calibration['checkpoint_sha256'] == source_hash
    assert calibration['manifest_sha256'] == digest(args.samples_dir / 'samples_manifest.csv')
    lr = json.loads(args.search_calibration.read_text())['selected_lr']
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_affine_index_search.py', 'finetuning/affine_index_search.py', 'finetuning/affine_codec_study.py',
            'compression/affine_restoration.py', 'finetuning/run_physical_guard_search.py', 'finetuning/run_index_search.py']
    write_json(out / 'protocol.json', {'kind': 'fresh_paired_affine_e1_v2' if args.bounded_ols else 'fresh_paired_affine_e1_v1',
        'seed': seed, 'bounded_ols': args.bounded_ols, 'parameter_updates': 0,
        'source_sha256': source_hash, 'gpu': torch.cuda.get_device_name(), 'steps': 48, 'lr': lr,
        'probe_results_sha256': digest(args.probe_dir / 'results.json'),
        'code_sha256': {p: digest(ROOT / p) for p in code},
        'validation': [r['sample_path'] for r in rows], 'excluded_validation_windows': sorted(excluded),
        'sample_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in rows},
        'guard': 'same strict bytes/RMSE/correlation/p90/FFT magnitude/phase/RMS-distance guard for every selector',
        'contrast': 'one identical 48-step proposal trajectory; raw strict, two affine proposals' +
                    (' and a nested third bounded-OLS proposal' if args.bounded_ols else ''),
        'bounded_numeric_margin': '32 float32 eps * max(1, gain_lo, gain_hi), capped at 1/4 of gain interval; no physical guard relaxation' if args.bounded_ols else None,
        'timing': 'combined cost of shared trajectory and both selectors; no per-arm isolated latency claim',
        'selection': 'minimum mixed normalized distortion; no tuning using validation',
        'rate_contract': 'same train-fitted symbol tables and source cap per window, FNE1/FAS1 explicit descriptor; shared bytes charged equally',
        'extension': 'no hyperparameter sweep or continuation; no promotion without confirmation/physical acceptance',
        'limitations': 'new recortes of reused development configurations; not new acquisitions; FAS1 is not input-normalization statistics'})
    for p in code:
        dest = out / 'code_snapshot' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p, dest)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    del saved
    forward = model.decoder.forward
    model.decoder.forward = lambda x: checkpoint(forward, x, use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else forward(x)
    stream = FrozenIndexStream(np.load(args.calibration_dir / 'counts.npy', allow_pickle=False), source_hash)
    names = ('source', 'strict', 'affine', 'bounded') if args.bounded_ols else ('source', 'strict', 'affine')
    records, metrics = [], {k: [] for k in names}
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for i, row in enumerate(rows):
        raw, packet = load_packet(row)
        tensor, valid = torch.from_numpy(packet.values)[None, None].cuda(), torch.from_numpy(packet.valid)[None, None].cuda()
        source_frame, before, chosen, evaluated = paired_search(model, tensor, valid, row, raw, packet, stream, lr, bounded_ols=args.bounded_ols)
        metrics['source'].append(before)
        (out / f'{i:05d}.source.frame').write_bytes(source_frame)
        record = {'sample_path': row['sample_path'], 'drive_id': row['drive_id'], 'source': before,
                  'source_frame_bytes': len(source_frame), 'evaluated': evaluated}
        for arm, item in chosen.items():
            frame = item.pop('frame')
            (out / f'{i:05d}.{arm}.frame').write_bytes(frame)
            metrics[arm].append(item['metrics'])
            record[arm] = {**item, 'frame_bytes': len(frame), 'uses_affine_side': frame != source_frame and item['restoration'] != 'original'}
        records.append(record)
        write_json(out / 'records.json', records)
        if (i + 1) % 16 == 0:
            print(json.dumps({'completed': i + 1, 'total': len(rows)}), flush=True)
    result = {'parameter_updates': 0, 'windows': len(rows), 'combined_seconds': time.perf_counter() - start,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'arms': {}}
    for arm, values in metrics.items():
        macro, groups = aggregate_metrics(values)
        size = sum((out / f'{i:05d}.{arm}.frame').stat().st_size for i in range(len(rows)))
        result['arms'][arm] = {'macro': macro, 'groups': groups, 'frame_bytes': size}
    source_rmse = result['arms']['source']['macro']['macro_relative_rmse']
    for value in result['arms'].values():
        value['gain_vs_source'] = 1 - value['macro']['macro_relative_rmse'] / source_rmse
    result['affine_gain_vs_strict'] = 1 - result['arms']['affine']['macro']['macro_relative_rmse'] / result['arms']['strict']['macro']['macro_relative_rmse']
    result['affine_side_windows'] = sum(r['affine']['uses_affine_side'] for r in records)
    if args.bounded_ols:
        result['bounded_gain_vs_affine'] = 1 - result['arms']['bounded']['macro']['macro_relative_rmse'] / result['arms']['affine']['macro']['macro_relative_rmse']
        result['bounded_side_windows'] = sum(r['bounded']['uses_affine_side'] for r in records)
    np.save(out / 'counts.npy', np.load(args.calibration_dir / 'counts.npy', allow_pickle=False))
    write_json(out / 'descriptor.json', {'source_checkpoint_sha256': source_hash,
        'side_formats': {'FNE1': 'input normalization mean/std restoration', 'FAS1': 'adaptive output offset/scale; original input mean/std not retained'},
        'new_waveform_weights_bytes': 0, 'counts_sha256': digest(out / 'counts.npy')})
    maps = [json.loads(line) for line in (args.samples_dir / 'channel_maps.jsonl').read_text().splitlines()]
    ids = {r['channel_map_id'] for r in rows}
    write_json(out / 'maps.json', [m for m in maps if m['channel_map_id'] in ids])
    write_json(out / 'transport.json', [{'index': i, 'sample_path': r['sample_path'], 'drive_id': r['drive_id'],
        'block_index': int(r['block_index']), 'time_start': int(r['time_start']), 'channel_map_id': r['channel_map_id']}
        for i, r in enumerate(rows)])
    common = sum((out / name).stat().st_size for name in ('counts.npy', 'descriptor.json', 'maps.json', 'transport.json'))
    for value in result['arms'].values():
        value['stream_bytes'] = value['frame_bytes'] + common
    result['common_auxiliary_bytes'] = common
    assert digest(args.source) == source_hash
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}), flush=True)
    print(json.dumps({k: {'rmse': v['macro']['macro_relative_rmse'], 'gain': v['gain_vs_source'], 'bytes': v['frame_bytes']} for k, v in result['arms'].items()}), flush=True)


if __name__ == '__main__':
    main()
