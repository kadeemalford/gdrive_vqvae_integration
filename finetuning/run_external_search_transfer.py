#!/usr/bin/env python3
"""Fixed bounded-affine check on the already-consumed external SMF subset."""
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
from compression.numeric_restoration import encode_numeric
from finetuning.affine_index_search import paired_search
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, write_json
from finetuning.run_index_search import FrozenIndexStream


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-experiment', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--development-confirmation', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    assert (args.development_confirmation / 'COMPLETE').exists()
    assert json.loads((args.development_confirmation / 'results.json').read_text())['bounded_gain_vs_affine'] > 0
    baseline = json.loads((args.baseline_experiment / 'protocol.json').read_text())
    source_hash = digest(args.source)
    assert source_hash == baseline['source_checkpoint_sha256']['compact']
    samples = json.loads((args.samples_dir / 'samples.json').read_text())
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_external_search_transfer.py', 'finetuning/affine_index_search.py',
            'compression/affine_restoration.py', 'compression/numeric_restoration.py',
            'finetuning/affine_codec_study.py', 'finetuning/run_physical_guard_search.py']
    write_json(out / 'protocol.json', {'kind': 'external_development_bounded_affine_transfer_v1', 'parameter_updates': 0,
        'source_sha256': source_hash, 'gpu': torch.cuda.get_device_name(),
        'baseline_results_sha256': digest(args.baseline_experiment / 'results.json'),
        'development_confirmation_sha256': digest(args.development_confirmation / 'results.json'),
        'code_sha256': {s: digest(ROOT / s) for s in code},
        'selection': 'same six consumed SMF files, temporal frame 10 and regions 512+39; 12 recortes',
        'steps': 48, 'lr': baseline['search_lr'], 'bounded_margin': 'same train-confirmed 32-float32-epsilon formula',
        'guard': 'unchanged source bytes, RMSE/correlation/p90/FFT magnitude/phase/RMS-distance',
        'role': 'external DEVELOPMENT reuse, not fresh confirmation or final holdout',
        'limitations': ['one linked SMF group; no parameter sweep or EDAS signal access',
                        'same masks/escapes/tables as closed baseline; no neural training']})
    for name in code:
        dest = out / 'code_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    del saved
    forward = model.decoder.forward
    model.decoder.forward = lambda x: checkpoint(forward, x, use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else forward(x)
    counts_path = args.baseline_experiment / 'compact/counts.npy'
    stream = FrozenIndexStream(np.load(counts_path, allow_pickle=False), source_hash)
    records, measured = [], {k: [] for k in ('source', 'strict', 'affine', 'bounded')}
    start_time = time.perf_counter()
    for index, sample in enumerate(samples):
        assert digest(args.samples_dir / sample['path']) == sample['sha256']
        with np.load(args.samples_dir / sample['path'], allow_pickle=False) as data:
            raw_full = data['data']
        for region, begin in enumerate((0, 512)):
            raw = raw_full[begin:min(begin + 512, 551), 10000:11000].copy()
            packet = encode_numeric(raw)
            row = {'family': 'single_mode', 'configuration': 'External SMF/' + sample['orientation']}
            source, before, chosen, _ = paired_search(model, torch.from_numpy(packet.values)[None, None].cuda(),
                torch.from_numpy(packet.valid)[None, None].cuda(), row, raw, packet, stream,
                baseline['search_lr'], bounded_ols=True)
            key = f'f{index:02d}_w10_r{region}'
            assert source == (args.baseline_experiment / 'compact' / f'{key}.direct.frame').read_bytes()
            (out / f'{key}.source.frame').write_bytes(source)
            measured['source'].append(before)
            record = {'drive_id': sample['drive_id'], 'region': region, 'source': before, 'source_frame_bytes': len(source)}
            for arm, item in chosen.items():
                frame = item.pop('frame')
                (out / f'{key}.{arm}.frame').write_bytes(frame)
                measured[arm].append(item['metrics'])
                record[arm] = {**item, 'frame_bytes': len(frame)}
            records.append(record)
    write_json(out / 'records.json', records)
    result = {'parameter_updates': 0, 'windows': len(records), 'combined_seconds': time.perf_counter() - start_time, 'arms': {}}
    for arm, values in measured.items():
        macro, groups = aggregate_metrics(values)
        size = sum(s.stat().st_size for s in out.glob(f'*.{arm}.frame'))
        result['arms'][arm] = {'macro': macro, 'groups': groups, 'frame_bytes': size}
    for item in result['arms'].values():
        item['gain_vs_source'] = 1 - item['macro']['macro_relative_rmse'] / result['arms']['source']['macro']['macro_relative_rmse']
    np.save(out / 'counts.npy', np.load(counts_path, allow_pickle=False))
    shutil.copy2(args.baseline_experiment / 'compact/descriptor.json', out / 'descriptor.json')
    result['rate_note'] = 'frame totals; unchanged shared models/tables/maps/transport, no complete subset ratio claim'
    assert digest(args.source) == source_hash
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({k: {'gain': v['gain_vs_source'], 'rmse': v['macro']['macro_relative_rmse'], 'frames': v['frame_bytes']} for k, v in result['arms'].items()}), flush=True)


if __name__ == '__main__':
    main()
