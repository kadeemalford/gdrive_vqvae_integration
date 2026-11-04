#!/usr/bin/env python3
"""Paired frozen-codec evaluation and train-only residual diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.model_factory import build_model
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import digest, load_packet, pilot_rows, read_manifest, score, tensor_packet, write_json
from finetuning.run_index_search import FrozenIndexStream, nearest, search, vectors
from finetuning.summarize_index_search import admit_search


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def guarded_admit(before, after):
    keys = ('relative_rmse', 'correlation', 'ordinary_channel_p90_relative_rmse')
    if any(v.get(k) is None or not np.isfinite(v[k]) for v in (before, after) for k in keys):
        return False
    return admit_search(before, after)


def aggregate_metrics(rows):
    # Undefined metrics remain missing, never silently replaced with a good score.
    return aggregate_validation([{k: np.nan if v is None else v for k, v in row.items()} for row in rows])


def residual_diagnostics(target, prediction, valid, sample_rate=20000):
    """Describe error, excluding escapes; spatial probe uses a held-out time suffix.

    The predictor sees neighboring ORIGINAL residuals, not decoder-available
    information. This is an optimistic dependency probe, never a codec result.
    FFT metrics use only entirely ordinary channels to avoid filling escape holes.
    """
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if target.ndim != 2 or prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Matching two-dimensional arrays required')
    if not valid.any() or not np.isfinite(target[valid]).all() or not np.isfinite(prediction[valid]).all():
        raise ValueError('Finite ordinary samples required')
    error = target - prediction
    edge = np.zeros_like(valid)
    edge[:min(8, target.shape[0]), :] = True
    edge[-8:, :] = True
    edge[:, :min(32, target.shape[1])] = True
    edge[:, -32:] = True
    result = {'ordinary_cells': int(valid.sum()), 'mse': float(np.mean(error[valid] ** 2)),
              'mae': float(np.mean(np.abs(error[valid])))}
    for name, mask in [('edge', edge & valid), ('interior', ~edge & valid)]:
        result[name + '_cells'] = int(mask.sum())
        result[name + '_mse'] = float(np.mean(error[mask] ** 2)) if mask.any() else None
    ordinary_channels = valid.all(axis=1)
    result['fft_channels'] = int(ordinary_channels.sum())
    result['fft_coverage_fraction'] = float(ordinary_channels.mean())
    if ordinary_channels.any():
        ep = np.abs(np.fft.rfft(error[ordinary_channels], axis=1)) ** 2
        tp = np.abs(np.fft.rfft(target[ordinary_channels], axis=1)) ** 2
        freq = np.fft.rfftfreq(target.shape[1], 1 / sample_rate)
        result['bands'] = []
        for lo, hi in [(0, 1000), (1000, 5000), (5000, sample_rate / 2 + 1)]:
            mask = (freq >= lo) & (freq < hi)
            numerator, denominator = float(ep[:, mask].sum()), float(tp[:, mask].sum())
            result['bands'].append({'lo_hz': lo, 'hi_hz': hi, 'error_power': numerator,
                                    'target_power': denominator,
                                    'error_target_power_ratio': numerator / denominator if denominator > 1e-20 else None})
    cut = target.shape[1] // 2
    pair = valid[:-1] & valid[1:]
    x, y = error[:-1], error[1:]
    fit = pair[:, :cut]
    test = pair[:, cut:]
    result['spatial_probe'] = {'fit_cells': int(fit.sum()), 'test_cells': int(test.sum()),
                               'status': 'insufficient_pairs', 'original_neighbor_access': True}
    if fit.sum() >= 32 and test.sum() >= 32:
        xf, yf = x[:, :cut][fit], y[:, :cut][fit]
        xm, ym = float(xf.mean()), float(yf.mean())
        variance = float(np.sum((xf - xm) ** 2))
        coefficient = float(np.sum((xf - xm) * (yf - ym)) / (variance + 1e-3 * len(xf)))
        yt, xt = y[:, cut:][test], x[:, cut:][test]
        baseline = float(np.mean((yt - ym) ** 2))
        mse = float(np.mean((yt - (ym + coefficient * (xt - xm))) ** 2))
        result['spatial_probe'].update(status='ok' if baseline > 1e-20 else 'constant_target',
            coefficient=coefficient, baseline_mse=baseline, predicted_mse=mse,
            relative_rmse_gain=1 - float(np.sqrt(mse / baseline)) if baseline > 1e-20 else None)
    return result


def timed(device, operation):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = operation()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--sharp', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--search-calibration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-windows', type=int, help='Smoke only; production uses all preselected windows')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.max_windows is not None and args.max_windows < 1:
        raise ValueError('Positive smoke window count required')
    device = torch.device(args.device)
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(2083)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splits = read_manifest(args.samples_dir)
    selected = {s: pilot_rows(rows, seed=2083) for s, rows in splits.items()}
    if args.max_windows:
        selected = {s: rows[:args.max_windows] for s, rows in selected.items()}
    calibration = json.loads((args.calibration_dir / 'calibration.json').read_text())
    source_hash = digest(args.source)
    assert source_hash == calibration['checkpoint_sha256']
    assert digest(args.samples_dir / 'samples_manifest.csv') == calibration['manifest_sha256']
    lr = json.loads(args.search_calibration.read_text())['selected_lr']
    hashes = {name: digest(getattr(args, name)) for name in ('source', 'control', 'sharp')}
    for name, filename in [('control', args.control), ('sharp', args.sharp)]:
        historical = json.loads((filename.parent / 'results.json').read_text())
        assert hashes[name] == historical['checkpoint_sha256']
    protocol = {'kind': 'paired_e1_residual_v2', 'seed': 2083, 'steps': 48, 'learning_rate': lr,
        'checkpoint_sha256': hashes, 'calibration_sha256': digest(args.calibration_dir / 'calibration.json'),
        'counts_sha256': digest(args.calibration_dir / 'counts.npy'),
        'search_calibration_sha256': digest(args.search_calibration),
        'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'code_sha256': {p: digest(ROOT / p) for p in ('finetuning/paired_search_diagnostics.py',
            'finetuning/run_index_search.py', 'finetuning/representation_study.py',
            'finetuning/summarize_index_search.py', 'finetuning/model_factory.py')},
        'selection': {s: [r['sample_path'] for r in rows] for s, rows in selected.items()},
        'sample_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for rows in selected.values() for r in rows},
        'external_code_sha256': {str(args.repo_root / p): digest(args.repo_root / p)
                                 for p in ('vqvae/train_vqvae.py', 'vqvae/evaluate_vqvae.py')},
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch_version': str(torch.__version__), 'parameter_updates': 0, 'no_test_inference': True,
        'smoke_only': args.max_windows is not None,
        'undefined_metric_policy': 'reject altered candidates if any guarded metric is undefined; retain source frame; serialize undefined metrics as null and aggregate finite values with coverage counts',
        'contract': 'same train-fitted tables and source decoder; per-window real frame ceiling and RMSE/correlation/p90 guards',
        'residual_policy': 'train only; normalized ordinary target; spatial adjacency within tile, not physical distance; no continuous-time or pooled-file claim',
        'timing_policy': 'synchronized wall time; candidate score/encoding included; search includes guarded candidate scoring; startup and model loading excluded; not an isolated hardware benchmark',
        'search_memory': 'non-reentrant decoder activation checkpointing; FP32; same 48-step proposal and every-four-step discrete guard'}
    out = args.output_dir
    if args.resume:
        assert json.loads((out / 'protocol.json').read_text()) == protocol, 'Resume protocol mismatch'
        assert not (out / 'COMPLETE').exists()
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'protocol.json', protocol)
        for path in protocol['code_sha256']:
            dest = out / 'code_snapshot' / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, dest)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model({**saved['config'], 'exact_eval_quantization': True}, saved, device).eval().requires_grad_(False)
    source_encoder = model.encoder
    encoders = {'source': source_encoder}
    for name in ('control', 'sharp'):
        item = torch.load(getattr(args, name), map_location='cpu', weights_only=False)
        for key, value in saved['model_state_dict'].items():
            if not key.startswith('encoder.'):
                torch.testing.assert_close(value, item['model_state_dict'][key], rtol=0, atol=0)
        other = build_model({**item['config'], 'exact_eval_quantization': True}, item, 'cpu')
        encoders[name] = other.encoder.to(device).eval().requires_grad_(False)
        del item, other
    del saved
    # Checkpointing changes storage/recomputation, not the frozen synthesis function.
    decoder_forward = model.decoder.forward
    model.decoder.forward = lambda x: checkpoint(decoder_forward, x, use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else decoder_forward(x)
    stream = FrozenIndexStream(np.load(args.calibration_dir / 'counts.npy', allow_pickle=False), source_hash)
    for name, data in [('huffman_table.json', stream.huffman.serialized), ('range_descriptor.json', stream.codec.serialized),
                       ('range_probabilities.f32', stream.probabilities.tobytes())]:
        (out / name).write_bytes(data)
    for split, rows in selected.items():
        directory = out / split
        directory.mkdir(exist_ok=True)
        for index, row in enumerate(rows):
            record_path = directory / f'{index:05d}.json'
            if record_path.exists():
                assert json.loads(record_path.read_text())['sample_path'] == row['sample_path']
                continue
            raw, packet = load_packet(row)
            tensor, valid = tensor_packet(packet, device)
            measured, timing, sizes, predictions = {}, {}, {}, {}
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for label, encoder in encoders.items():
                model.encoder = encoder
                def evaluate():
                    with torch.no_grad():
                        h, w = raw.shape
                        z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
                        indices = nearest(z, model.quantizer.embedding.weight)
                        pred = model.decoder(vectors(indices, model.quantizer.embedding.weight, z.shape[-2:]))[..., :h, :w][0, 0].cpu().numpy()
                        frame = stream.encode(indices, raw.shape, z.shape[-2:], packet.side_stream)
                        recovered, shape, latent, side = stream.decode(frame)
                        np.testing.assert_array_equal(recovered, indices.cpu().numpy())
                        assert shape == raw.shape and tuple(latent) == tuple(z.shape[-2:]) and side == packet.side_stream
                        replay = model.decoder(vectors(torch.from_numpy(recovered.astype(np.int64)).to(device),
                            model.quantizer.embedding.weight, latent))[..., :h, :w][0, 0].cpu().numpy()
                        np.testing.assert_array_equal(pred, replay)
                        metrics = finite_json(score(row, raw, packet, pred))
                        return pred, frame, metrics
                (prediction, frame, metric), timing[label] = timed(device, evaluate)
                (directory / f'{index:05d}.{label}.frame').write_bytes(frame)
                measured[label], sizes[label], predictions[label] = metric, len(frame), prediction
                if label != 'source':
                    eligible = len(frame) <= sizes['source'] and guarded_admit(measured['source'], metric)
                    measured[label + '_guarded'] = metric if eligible else measured['source']
                    sizes[label + '_guarded'] = len(frame) if eligible else sizes['source']
            model.encoder = source_encoder
            before = measured['source']
            def guard(original, proposed):
                return guarded_admit(before, score(row, raw, packet, proposed))
            (baseline, searched, details), timing['search'] = timed(device, lambda: search(
                model, tensor, valid, 48, lr, stream, packet.side_stream,
                directory / f'{index:05d}.search.frame', candidate_guard=guard))
            np.testing.assert_array_equal(baseline, predictions['source'])
            assert details['baseline_frame_bytes'] == sizes['source']
            measured['search'] = finite_json(score(row, raw, packet, searched))
            sizes['search'] = details['searched_frame_bytes']
            assert sizes['search'] <= sizes['source']
            assert details['changed_fraction'] == 0 or guarded_admit(before, measured['search'])
            record = {'sample_path': row['sample_path'], 'drive_id': row['drive_id'],
                'family': row['family'], 'configuration': row['configuration'], 'channels': raw.shape[0],
                'cells': raw.size, 'metrics': measured, 'frame_bytes': sizes, 'seconds': timing,
                'search_details': details, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
            if split == 'train':
                record['residual'] = {label: residual_diagnostics(packet.values, pred, packet.valid)
                                      for label, pred in [('source', predictions['source']), ('search', searched)]}
            write_json(record_path, record)
            print(json.dumps({'split': split, 'completed': index + 1, 'total': len(rows), 'seconds': timing}), flush=True)
    result = {'parameter_updates': 0, 'no_test_inference': True, 'splits': {},
        'limitations': ['development configurations only', 'frame bytes exclude shared tables/maps/transport/weights; no complete compression ratio claimed',
                        'train residual probe accesses original neighbors', 'spaced 50ms windows; no pooled continuous file quantiles',
                        'timings include warmup and desktop GPU contention; guarded encoder also requires source evaluation']}
    for split, rows in selected.items():
        records = [json.loads((out / split / f'{i:05d}.json').read_text()) for i in range(len(rows))]
        summary = {}
        for label in records[0]['metrics']:
            macro, groups = aggregate_metrics([r['metrics'][label] for r in records])
            summary[label] = {'macro': macro, 'groups': groups,
                              'undefined_metric_windows': {key: sum(r['metrics'][label][key] is None for r in records)
                                                           for key in records[0]['metrics'][label]},
                              'frame_bytes': sum(r['frame_bytes'][label] for r in records)}
        for label in summary:
            summary[label]['rmse_gain_vs_source'] = 1 - summary[label]['macro']['macro_relative_rmse'] / summary['source']['macro']['macro_relative_rmse']
        result['splits'][split] = {'windows': len(rows), 'arms': summary,
            'seconds_total': {label: sum(r['seconds'][label] for r in records) for label in records[0]['seconds']},
            'peak_cuda_allocated_bytes': max((r['peak_cuda_allocated_bytes'] or 0) for r in records)}
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({s: {label: {'gain': arm['rmse_gain_vs_source'], 'bytes': arm['frame_bytes']}
        for label, arm in data['arms'].items()} for s, data in result['splits'].items()}), flush=True)


if __name__ == '__main__':
    main()
