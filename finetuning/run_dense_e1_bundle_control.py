#!/usr/bin/env python3
"""Fixed-rate dense E1 control on the same consumed SMF middle-window subset."""
import argparse
import itertools
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
from compression.numeric_restoration import encode_numeric
from compression.partition_bundle import pack_bundle, decode_bundle
from finetuning.affine_codec_study import score_restored, temporal_distortion, decode_affine_frame
from finetuning.affine_index_search import paired_search
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, write_json
from finetuning.run_index_search import FrozenIndexStream
from finetuning.run_physical_guard_search import physical_admit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-experiment', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = json.loads((args.baseline_experiment / 'protocol.json').read_text())
    source_hash = digest(args.source)
    assert source_hash == baseline['source_checkpoint_sha256']['dense']
    samples = json.loads((args.samples_dir / 'samples.json').read_text())
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_dense_e1_bundle_control.py', 'finetuning/affine_index_search.py',
            'compression/partition_bundle.py', 'compression/affine_restoration.py',
            'compression/numeric_restoration.py', 'finetuning/affine_codec_study.py']
    write_json(out / 'protocol.json', {'kind': 'dense_fixed_rate_e1_bundle_control_v1', 'parameter_updates': 0,
        'source_sha256': source_hash, 'gpu': torch.cuda.get_device_name(),
        'baseline_results_sha256': digest(args.baseline_experiment / 'results.json'),
        'code_sha256': {s: digest(ROOT / s) for s in code},
        'selection': 'same six external DEVELOPMENT SMF files, middle temporal window 10, two original 512+39 fragments',
        'steps_per_fragment': 48, 'lr': baseline['search_lr'],
        'proposal_method': 'fixed transfer of train-calibrated historical LR, not a dense-specific tuning or optimality claim',
        'selection_method': 'nested strict/affine/bounded fragment proposals; combine original/selected fragments, choose minimum whole-bundle distortion under unchanged whole-window physical and byte guards',
        'guard_scope': 'each fragment protected first, then full 551x1000 bundle; stronger than partition-only full-bundle guard',
        'wire': 'same FPB1 two-fragment framing and same latent counts as source; no new weights or partition choices',
        'extension': 'none; no LR, iteration or temperature sweep',
        'limitations': ['external DEVELOPMENT reuse, not independent confirmation',
                        'one middle window per file, not pooled 1s evidence',
                        'heuristic negative is not a decoder optimum or a family impossibility result']})
    for name in code:
        dest = out / 'code_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    del saved
    forward = model.decoder.forward
    model.decoder.forward = lambda x: checkpoint(forward, x, use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else forward(x)
    counts_path = args.baseline_experiment / 'dense/counts.npy'
    stream = FrozenIndexStream(np.load(counts_path, allow_pickle=False), source_hash)
    arms = ('strict', 'affine', 'bounded')
    measured = {k: [] for k in ('source', *arms)}
    sizes = {k: 0 for k in measured}
    records = []
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    for file_index, sample in enumerate(samples):
        assert digest(args.samples_dir / sample['path']) == sample['sha256']
        with np.load(args.samples_dir / sample['path'], allow_pickle=False) as data:
            raw = data['data'][:, 10000:11000].copy()
        packet = encode_numeric(raw)
        row = {'family': 'single_mode', 'configuration': 'External SMF/' + sample['orientation']}
        options, source_fragments = [], []
        for region, begin in enumerate((0, 512)):
            part = raw[begin:min(begin + 512, 551)].copy()
            local = encode_numeric(part)
            source, _, best, _ = paired_search(model, torch.from_numpy(local.values)[None, None].cuda(),
                torch.from_numpy(local.valid)[None, None].cuda(), row, part, local, stream,
                baseline['search_lr'], bounded_ols=True)
            assert source == (args.baseline_experiment / 'dense' / f'f{file_index:02d}_w10_r{region}.direct.frame').read_bytes()
            options.append({'source': source, **{arm: best[arm]['frame'] for arm in arms}})
            source_fragments.append((0, begin, part.shape[0], source))
        source_bundle = pack_bundle(raw.shape, source_fragments)
        original = decode_bundle(source_bundle, stream, model, 'cuda')
        before = score_restored(row, raw, packet, original)
        chosen = {'frame': source_bundle, 'metrics': before, 'loss': temporal_distortion(original, packet), 'choice': [False, False]}
        measured['source'].append(before)
        sizes['source'] += len(source_bundle)
        (out / f'f{file_index:02d}.source.fpb').write_bytes(source_bundle)
        record = {'drive_id': sample['drive_id'], 'source': before, 'source_bytes': len(source_bundle)}
        for arm in arms:
            # Carry the prior selector's best feasible bundle to preserve nesting.
            best_bundle = chosen.copy()
            for use in itertools.product((False, True), repeat=2):
                fragments = [(0, begin, min(512, 551 - begin), options[region][arm if use[region] else 'source'])
                             for region, begin in enumerate((0, 512))]
                bundle = pack_bundle(raw.shape, fragments)
                if len(bundle) > len(source_bundle):
                    continue
                restored = decode_bundle(bundle, stream, model, 'cuda')
                metric = score_restored(row, raw, packet, restored)
                value = temporal_distortion(restored, packet)
                if value < best_bundle['loss'] and physical_admit(before, metric):
                    best_bundle = {'frame': bundle, 'metrics': metric, 'loss': value, 'choice': list(use)}
            chosen = best_bundle
            measured[arm].append(chosen['metrics'])
            sizes[arm] += len(chosen['frame'])
            (out / f'f{file_index:02d}.{arm}.fpb').write_bytes(chosen['frame'])
            record[arm] = {k: v for k, v in chosen.items() if k != 'frame'}
            record[arm]['frame_bytes'] = len(chosen['frame'])
        records.append(record)
        print(json.dumps({'files_completed': file_index + 1, 'total': len(samples)}), flush=True)
    write_json(out / 'records.json', records)
    np.save(out / 'counts.npy', np.load(counts_path, allow_pickle=False))
    write_json(out / 'descriptor.json', {'format': 'FPB1', 'source_checkpoint_sha256': source_hash,
        'counts_sha256': digest(out / 'counts.npy'), 'new_waveform_weights_bytes': 0})
    result = {'parameter_updates': 0, 'windows': 6, 'combined_seconds': time.perf_counter() - start_time,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'arms': {}}
    for arm, values in measured.items():
        macro, groups = aggregate_metrics(values)
        result['arms'][arm] = {'macro_full_window': macro, 'groups': groups, 'bundle_bytes': sizes[arm]}
    for item in result['arms'].values():
        item['gain_vs_source'] = 1 - item['macro_full_window']['macro_relative_rmse'] / result['arms']['source']['macro_full_window']['macro_relative_rmse']
    assert digest(args.source) == source_hash
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({a: {'gain': r['gain_vs_source'], 'rmse': r['macro_full_window']['macro_relative_rmse'], 'bytes': r['bundle_bytes']} for a, r in result['arms'].items()}), flush=True)


if __name__ == '__main__':
    main()
