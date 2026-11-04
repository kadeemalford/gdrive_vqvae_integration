#!/usr/bin/env python3
"""Separate boundary error concentration from target energy in frozen train frames."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import finite_json
from finetuning.representation_study import digest, load_packet, read_manifest, write_json
from finetuning.run_index_search import FrozenIndexStream, vectors


def boundary_energy(target, prediction, valid):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if target.ndim != 2 or target.shape != prediction.shape or target.shape != valid.shape:
        raise ValueError('Matching two-dimensional arrays required')
    error = target - prediction
    # Center each channel using all its ordinary temporal samples, not each region.
    counts = valid.sum(axis=1, keepdims=True)
    mean = np.where(valid, target, 0).sum(axis=1, keepdims=True) / np.maximum(counts, 1)
    power = (target - mean) ** 2
    result = {}
    for axis, width, name in [(0, 8, 'spatial'), (1, 32, 'temporal')]:
        edge = np.zeros_like(valid)
        if axis == 0:
            edge[:width] = True
            edge[-width:] = True
        else:
            edge[:, :width] = True
            edge[:, -width:] = True
        regions = {}
        for label, mask in [('edge', edge & valid), ('interior', ~edge & valid)]:
            n = int(mask.sum())
            ep = float(np.sum(error[mask] ** 2))
            tp = float(np.sum(power[mask]))
            regions[label] = {'cells': n, 'error_squared_sum': ep, 'target_centered_squared_sum': tp,
                'mse': ep / n if n else None, 'target_variance': tp / n if n else None,
                'error_target_power_ratio': ep / tp if tp > 1e-20 else None}
        a, b = regions['edge'], regions['interior']
        regions['edge_interior_relative_error_power_ratio'] = (
            a['error_target_power_ratio'] / b['error_target_power_ratio']
            if a['error_target_power_ratio'] is not None and b['error_target_power_ratio'] is not None
            and b['error_target_power_ratio'] > 1e-20 else None)
        result[name] = regions
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert (args.experiment_dir / 'COMPLETE').exists()
    protocol = json.loads((args.experiment_dir / 'protocol.json').read_text())
    assert digest(args.source) == protocol['checkpoint_sha256']['source']
    assert digest(args.samples_dir / 'samples_manifest.csv') == protocol['manifest_sha256']
    assert digest(args.calibration_dir / 'counts.npy') == protocol['counts_sha256']
    manifest = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'protocol.json', {'kind': 'train_boundary_energy_v1',
        'paired_protocol_sha256': digest(args.experiment_dir / 'protocol.json'),
        'source_sha256': digest(Path(__file__)), 'checkpoint_sha256': digest(args.source),
        'selection': protocol['selection']['train'], 'parameter_updates': 0, 'device': 'cuda',
        'gpu': torch.cuda.get_device_name(), 'no_validation_or_test_waveforms': True,
        'regions': {'spatial_edge_rows': 8, 'temporal_edge_samples': 32},
        'normalization': 'per-channel target mean over all ordinary temporal samples; region error power divided by region centered target power',
        'limitations': 'descriptive energy-controlled concentration, not a causal padding test; no continuous-time reconstruction'})
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model({**saved['config'], 'exact_eval_quantization': True}, saved, 'cuda').eval().requires_grad_(False)
    del saved
    stream = FrozenIndexStream(np.load(args.calibration_dir / 'counts.npy', allow_pickle=False), digest(args.source))
    records = []
    with torch.no_grad():
        for index, name in enumerate(protocol['selection']['train']):
            row = manifest[name]
            assert digest(ROOT / name) == protocol['sample_sha256'][name]
            raw, packet = load_packet(row)
            record = {'sample_path': name, 'family': row['family'], 'configuration': row['configuration'],
                      'dx_m': float(row['dx_m']), 'contiguous_spatial_rows': row['spatial_rows_contiguous'], 'arms': {}}
            for arm in ('source', 'search'):
                frame = (args.experiment_dir / 'train' / f'{index:05d}.{arm}.frame').read_bytes()
                indices, shape, latent, side = stream.decode(frame)
                assert shape == raw.shape and side == packet.side_stream
                prediction = model.decoder(vectors(torch.from_numpy(indices.astype(np.int64)).cuda(),
                    model.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]][0, 0].cpu().numpy()
                record['arms'][arm] = boundary_energy(packet.values, prediction, packet.valid)
            records.append(record)
    grouped = defaultdict(list)
    for row in records:
        grouped[row['family'], row['configuration']].append(row)
    summary = []
    for key, rows in sorted(grouped.items()):
        entry = {'family': key[0], 'configuration': key[1], 'windows': len(rows), 'arms': {}}
        for arm in ('source', 'search'):
            entry['arms'][arm] = {}
            for axis in ('spatial', 'temporal'):
                ratios = [r['arms'][arm][axis]['edge_interior_relative_error_power_ratio'] for r in rows]
                finite = [v for v in ratios if v is not None]
                entry['arms'][arm][axis] = {'eligible_windows': len(finite),
                    'median_edge_interior_relative_error_power_ratio': float(np.median(finite)) if finite else None}
        summary.append(entry)
    write_json(args.output_dir / 'results.json', finite_json({'parameter_updates': 0, 'records': records,
        'groups': summary, 'aggregation': 'unweighted descriptive median within each configuration'}))
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
