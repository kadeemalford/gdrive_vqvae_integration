#!/usr/bin/env python3
"""Predefined nested physical nonregression guards for frozen discrete E1 search."""
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
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import finite_json, aggregate_metrics, guarded_admit
from finetuning.representation_study import digest, load_packet, pilot_rows, read_manifest, score, write_json
from finetuning.run_index_search import FrozenIndexStream, search


def physical_admit(before, after):
    if not guarded_admit(before, after):
        return False
    keys = ('fft_magnitude_relative_error', 'fft_phase_alignment', 'reconstruction_rms_ratio')
    if any(row.get(key) is None or not np.isfinite(row[key]) for row in (before, after) for key in keys):
        return False
    return (after['fft_magnitude_relative_error'] <= before['fft_magnitude_relative_error'] and
            after['fft_phase_alignment'] >= before['fft_phase_alignment'] and
            abs(after['reconstruction_rms_ratio'] - 1) <= abs(before['reconstruction_rms_ratio'] - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--search-calibration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('This campaign requires the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(2086)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paired = json.loads((args.paired_dir / 'protocol.json').read_text())
    source_hash = digest(args.source)
    assert source_hash == paired['checkpoint_sha256']['source']
    assert digest(args.samples_dir / 'samples_manifest.csv') == paired['manifest_sha256']
    assert digest(args.calibration_dir / 'counts.npy') == paired['counts_sha256']
    excluded = set(paired['selection']['val'])
    exclusion_hashes = {str(args.paired_dir / 'protocol.json'): digest(args.paired_dir / 'protocol.json')}
    # Also exclude windows used by earlier E1 pilots when their local protocols exist.
    for name in ('guard_aware_search_fresh_fold1_20261031', 'guard_aware_search_known_fold1_20261031',
                 'representation_e1_guarded_fresh_fold1_20261031', 'representation_e1_wire_transfer_fold1_20261031'):
        path = ROOT / 'results/experiments' / name / 'protocol.json'
        if path.exists():
            excluded.update(json.loads(path.read_text())['validation'])
            exclusion_hashes[str(path.relative_to(ROOT))] = digest(path)
    rows = pilot_rows([r for r in read_manifest(args.samples_dir)['val'] if r['sample_path'] not in excluded], seed=2086)
    assert not ({r['sample_path'] for r in rows} & excluded)
    lr = json.loads(args.search_calibration.read_text())['selected_lr']
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_physical_guard_search.py', 'finetuning/run_index_search.py',
            'finetuning/paired_search_diagnostics.py', 'finetuning/representation_study.py', 'finetuning/model_factory.py']
    write_json(out / 'protocol.json', {'kind': 'nested_physical_guard_e1_v1', 'parameter_updates': 0,
        'gpu': torch.cuda.get_device_name(), 'device': 'cuda', 'seed': 2086,
        'source_sha256': source_hash, 'counts_sha256': digest(args.calibration_dir / 'counts.npy'),
        'code_sha256': {p: digest(ROOT / p) for p in code}, 'exclusion_protocol_sha256': exclusion_hashes,
        'validation': [r['sample_path'] for r in rows],
        'sample_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in rows},
        'learning_rate': lr, 'steps': 48, 'projection_period': 4,
        'policies': {'standard': 'real frame ceiling and RMSE/correlation/p90 nonregression per candidate',
                     'physical': 'standard plus FFT magnitude error nonincrease, phase nondecrease, abs(RMS ratio-1) nonincrease'},
        'math_policy': 'FP32 deterministic cuDNN, TF32 disabled, decoder activation checkpointing',
        'limitations': ['strict diagnostic nonregression, not application-derived physical thresholds',
                        'additional instants of known files/configurations, not external acquisitions',
                        'no retroactive change to guards/admission in completed experiments',
                        'nested feasible candidate sets do not imply monotonic ranking on every metric; selector minimizes temporal distortion'],
        'extension': 'none; no tuning of physical tolerances against validation'})
    for path in code:
        dest = out / 'code_snapshot' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    del saved
    original_forward = model.decoder.forward
    model.decoder.forward = lambda x: checkpoint(original_forward, x, use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else original_forward(x)
    stream = FrozenIndexStream(np.load(args.calibration_dir / 'counts.npy', allow_pickle=False), source_hash)
    measured, records = {k: [] for k in ('source', 'standard', 'physical')}, []
    for i, row in enumerate(rows):
        raw, packet = load_packet(row)
        tensor = torch.from_numpy(packet.values)[None, None].cuda()
        valid = torch.from_numpy(packet.valid)[None, None].cuda()
        before_cache = []
        record = {'sample_path': row['sample_path'], 'drive_id': row['drive_id']}
        for policy, admission in [('standard', guarded_admit), ('physical', physical_admit)]:
            def guard(original, candidate):
                if not before_cache:
                    before_cache.append(finite_json(score(row, raw, packet, original)))
                return admission(before_cache[0], finite_json(score(row, raw, packet, candidate)))
            torch.cuda.synchronize()
            start = time.perf_counter()
            a, b, details = search(model, tensor, valid, 48, lr, stream, packet.side_stream,
                                   out / f'{i:05d}.{policy}.frame', candidate_guard=guard)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            before, after = finite_json(score(row, raw, packet, a)), finite_json(score(row, raw, packet, b))
            if policy == 'standard':
                measured['source'].append(before)
                record['source'] = before
            else:
                assert before == record['source']
            assert details['searched_frame_bytes'] <= details['baseline_frame_bytes']
            assert details['changed_fraction'] == 0 or admission(before, after)
            measured[policy].append(after)
            record[policy] = {'metrics': after, 'details': details, 'seconds': seconds}
        records.append(record)
    write_json(out / 'records.json', records)
    result = {'parameter_updates': 0, 'windows': len(rows), 'arms': {}}
    for policy, metrics in measured.items():
        macro, groups = aggregate_metrics(metrics)
        size = sum(r['standard']['details']['baseline_frame_bytes'] if policy == 'source' else r[policy]['details']['searched_frame_bytes'] for r in records)
        result['arms'][policy] = {'macro': macro, 'groups': groups, 'frame_bytes': size}
        if policy != 'source':
            result['arms'][policy].update(seconds_total=sum(r[policy]['seconds'] for r in records),
                changed_windows=sum(r[policy]['details']['changed_fraction'] > 0 for r in records))
    for arm in result['arms'].values():
        arm['rmse_gain_vs_source'] = 1 - arm['macro']['macro_relative_rmse'] / result['arms']['source']['macro']['macro_relative_rmse']
    # Same tables, geometry and transport for the two policies; never claim frames are a full codec ratio.
    np.save(out / 'counts.npy', np.load(args.calibration_dir / 'counts.npy', allow_pickle=False))
    result['limitations'] = 'frames only; shared tables/maps/transport/weights excluded and unchanged; known development files; strict diagnostic guards do not establish universal physical fidelity'
    assert digest(args.source) == source_hash
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({k: {'gain': v['rmse_gain_vs_source'], 'bytes': v['frame_bytes']} for k, v in result['arms'].items()}), flush=True)


if __name__ == '__main__':
    main()
