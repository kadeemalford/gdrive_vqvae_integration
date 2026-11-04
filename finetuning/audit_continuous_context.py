#!/usr/bin/env python3
"""Train-only contiguous-HDF5 temporal prediction diagnostic, not a codec."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.representation_study import digest, write_json
from finetuning.robust_escape import robust_threshold

SHORT_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)
LONG_LAGS = (*SHORT_LAGS, 2000, 4000)


def predict_ar(values, valid, split, lags, first_fit, ridge=.001):
    """One-step teacher-forced original-past prediction, prefix-fitted per channel."""
    prediction = np.empty_like(values[:, split:], dtype=np.float64)
    fallback = 0
    for channel in range(values.shape[0]):
        x = values[channel].astype(np.float64)
        mask = valid[channel]
        prefix = x[:split][mask[:split]]
        mean = float(prefix.mean()) if len(prefix) else 0.
        scale = max(float(prefix.std()), 1e-12) if len(prefix) else 1.
        z = (x - mean) / scale
        z[~mask] = 0
        fit_indices = np.arange(first_fit, split)
        fit_valid = mask[fit_indices].copy()
        for lag in lags:
            fit_valid &= mask[fit_indices - lag]
        fit_indices = fit_indices[fit_valid]
        prediction[channel] = mean
        if len(fit_indices) < 4 * len(lags):
            fallback += prediction.shape[1]
            continue
        design = np.stack([z[fit_indices - lag] for lag in lags], axis=1)
        gram = design.T @ design / len(fit_indices)
        rhs = design.T @ z[fit_indices] / len(fit_indices)
        coefficients = np.linalg.solve(gram + ridge * np.eye(len(lags)), rhs)
        future = np.arange(split, x.size)
        usable = np.ones(len(future), dtype=bool)
        for lag in lags:
            usable &= mask[future - lag]
        design = np.stack([z[future[usable] - lag] for lag in lags], axis=1)
        prediction[channel, usable] = mean + scale * (design @ coefficients)
        fallback += int((~usable).sum())
    return prediction, fallback


def describe(values, valid, dt):
    count = values.shape[1]
    split = count // 2
    first_fit = max(LONG_LAGS)
    if split < first_fit + 1000:
        raise ValueError('Insufficient contiguous prefix for predefined temporal lags')
    prefix_mean = np.array([v[:split][m[:split]].mean() if m[:split].any() else 0.
                           for v, m in zip(values, valid)])
    baseline = np.broadcast_to(prefix_mean[:, None], values[:, split:].shape).copy()
    target, target_mask = values[:, split:], valid[:, split:]
    if not target_mask.any():
        raise ValueError('No ordinary future targets to score')
    denominator = float(np.mean((target[target_mask] - baseline[target_mask]) ** 2))
    predictions = {'prefix_mean': (baseline, 0)}
    predictions['short_ar'] = predict_ar(values, valid, split, SHORT_LAGS, first_fit)
    predictions['long_ar'] = predict_ar(values, valid, split, LONG_LAGS, first_fit)
    result = {'cells': int(target.size), 'ordinary_cells': int(target_mask.sum()),
              'escaped_or_nonfinite_target_cells': int((~target_mask).sum()), 'duration_s': count * dt,
              'fit_seconds': split * dt, 'test_seconds': (count - split) * dt}
    frequencies = np.fft.rfftfreq(count - split, dt)
    for name, (prediction, fallback) in predictions.items():
        error = prediction - target
        mse = float(np.mean(error[target_mask] ** 2))
        residual = np.where(target_mask, error, 0.)
        power = np.abs(np.fft.rfft(residual, axis=1)) ** 2
        result[name] = {'ordinary_raw_rmse': float(np.sqrt(mse)),
            'mse_ratio_to_prefix_mean': mse / max(denominator, 1e-30),
            'invalid_context_fallback_cells': fallback,
            'high_band_residual_power_fraction': float(power[:, frequencies >= 5000].sum() / max(power.sum(), 1e-30))}
    short_error = result['short_ar']['ordinary_raw_rmse']
    result['long_vs_short_rmse_gain'] = (1 - result['long_ar']['ordinary_raw_rmse'] / short_error
                                        if short_error > 1e-30 else 0.)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with (args.samples_dir / 'samples_manifest.csv').open() as stream:
        train = [r for r in csv.DictReader(stream) if r['split'] == 'train']
    if not train:
        raise ValueError('Train manifest required')
    selected_files = {}
    for row in sorted(train, key=lambda r: (r['drive_id'], int(r['block_index']), r['sample_path'])):
        selected_files.setdefault((row['family'], row['configuration']), row['drive_id'])
    maps = {m['channel_map_id']: m for m in map(json.loads, (args.samples_dir / 'channel_maps.jsonl').read_text().splitlines())}
    selected = {}
    for row in sorted(train, key=lambda r: (int(r['block_index']), r['sample_path'])):
        if row['drive_id'] == selected_files[row['family'], row['configuration']]:
            selected.setdefault((row['drive_id'], row['spatial_region_index']), row)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'protocol.json', {'kind': 'continuous_train_temporal_ar_v1',
        'source_sha256': digest(Path(__file__)), 'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'selection': 'lexical first train file per configuration; first available block, all retained spatial regions',
        'short_lags': SHORT_LAGS, 'long_lags': LONG_LAGS, 'ridge': .001,
        'temporal_contract': 'up to first 1 s within one HDF5 block; first half fits, second half scores; no NPZ concatenation',
        'limitation': 'original-past one-step predictor; optimistic dependency diagnostic, not reconstructed context or a bitrate result',
        'selected_rows': list(selected.values()), 'no_test_or_validation_waveform_access': True})
    records = []
    for row in selected.values():
        path = args.raw_dir / f'{row["drive_id"]}.h5'
        mapping = maps[row['channel_map_id']]
        indices = np.asarray(mapping['source_channel_indices'], dtype=int)
        assert indices.size and np.all(np.diff(indices) == 1)
        with h5py.File(path, 'r') as handle:
            dataset = handle[row['dataset']]
            spacing = np.asarray(dataset.parent.attrs['Spacing'], dtype=float).reshape(-1)
            dt = float(spacing[1]) * .001
            assert abs(dt - float(row['dt_s'])) < 1e-12 and abs(dt - .00005) < 1e-12
            count = int(round(1 / dt))
            start, stop = int(indices[0]), int(indices[-1]) + 1
            if dataset.ndim == 3:
                raw = np.asarray(dataset[int(row['block_index']), :count, start:stop], dtype=np.float32).T
            elif dataset.shape[0] >= dataset.shape[1]:
                raw = np.asarray(dataset[:count, start:stop], dtype=np.float32).T
            else:
                raw = np.asarray(dataset[start:stop, :count], dtype=np.float32)
        prefix = raw[:, :raw.shape[1] // 2]
        if not np.isfinite(prefix).all():
            raise ValueError('Nonfinite selected train prefix; inspect retained geometry')
        threshold = robust_threshold(prefix)
        valid = np.isfinite(raw) & (np.abs(raw.astype(np.float64)) <= threshold)
        record = {'drive_id': row['drive_id'], 'family': row['family'], 'configuration': row['configuration'],
            'channel_map_id': row['channel_map_id'], 'channels': raw.shape[0], 'dt_s': dt,
            'block_index': int(row['block_index']), 'dataset': row['dataset'],
            'selected_array_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
            'ordinary_threshold_from_fit_prefix': threshold,
            **describe(raw.astype(np.float64), valid, dt)}
        records.append(record)
        write_json(args.output_dir / 'records.json', records)
        print(json.dumps({'regions': len(records), 'configuration': row['configuration'],
                          'long_vs_short_gain': record['long_vs_short_rmse_gain']}), flush=True)
    write_json(args.output_dir / 'results.json', {'regions': len(records), 'files': len(selected_files),
        'median_long_vs_short_rmse_gain': float(np.median([r['long_vs_short_rmse_gain'] for r in records])),
        'no_training_updates': True, 'no_codec_or_independence_claim': True})
    (args.output_dir / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
