#!/usr/bin/env python3
"""CPU-only physical metadata and bounded waveform diagnostics for QuantX pretraining."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def scalar(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return [scalar(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def diagnostics(values, sample_rate):
    """Describe time x locus data; no filtering, fitting, or training-target changes."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or sample_rate <= 0 or not np.isfinite(x).all():
        raise ValueError('Expected finite time x locus data and positive sample rate')
    x = x - x.mean(axis=0, keepdims=True)
    energy = np.square(x).sum(axis=0)
    total = float(energy.sum())
    if total == 0:
        return {'constant': True}
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    spectrum[1:-1 if len(x) % 2 == 0 else None] *= 2
    spectrum[0] = 0
    pooled = spectrum.sum(axis=1)
    freq = np.fft.rfftfreq(len(x), 1 / sample_rate)
    eigenvalues = np.maximum(np.linalg.eigvalsh(x.T @ x), 0)
    probabilities = eigenvalues[eigenvalues > 0] / eigenvalues.sum()

    def cosine(a, b):
        return float(np.sum(a * b) / max(np.sqrt(np.square(a).sum() * np.square(b).sum()), 1e-30))

    return {
        'constant': False, 'centered_rms': float(np.sqrt(total / x.size)),
        'channel_rms': np.sqrt(energy / len(x)).tolist(),
        'top10_channel_energy_fraction': float(np.sort(energy)[-10:].sum() / total),
        'spatial_effective_rank': float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        'temporal_lag1_cosine': cosine(x[1:], x[:-1]),
        'adjacent_channel_cosine': cosine(x[:, 1:], x[:, :-1]),
        'top5_spectral_bin_energy_fraction': float(np.sort(pooled)[-5:].sum() / pooled.sum()),
        'above_fs_quarter_energy_fraction': float(pooled[freq >= sample_rate / 4].sum() / pooled.sum()),
        'dominant_frequency_hz': float(freq[pooled.argmax()]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit_dir
    if not (audit / 'DOWNLOAD_COMPLETE').is_file():
        raise ValueError('Verified downloads must finish before waveform analysis')
    if (audit / 'signal_diagnostics.json').exists():
        raise FileExistsError('Do not overwrite an existing diagnostic result')
    downloads = json.loads((audit / 'downloads.json').read_text())
    metadata, rows = [], []
    for entry in downloads:
        if not entry['name'].endswith('.h5'):
            continue
        path = Path(entry['local_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Raw file SHA-256 mismatch')
        with h5py.File(path, 'r') as handle:
            acq, raw = handle['Acquisition'], handle['Acquisition/Raw[0]']
            ds = raw['RawData']
            if scalar(ds.attrs['Dimensions']) != ['time', 'locus']:
                raise ValueError('Unsupported dimension ordering')
            fs = float(raw.attrs['OutputDataRate'])
            setup = json.loads((path.parent / 'IUSetupConfig.JSON').read_text())
            meta = {'id': entry['id'], 'family': entry['family'], 'scenario': entry['scenario'],
                    'shape': list(ds.shape), 'dtype': str(ds.dtype), 'duration_s': ds.shape[0] / fs,
                    'acquisition': {k: scalar(v) for k, v in acq.attrs.items()},
                    'raw': {k: scalar(v) for k, v in raw.attrs.items()},
                    'data_attributes': {k: scalar(v) for k, v in ds.attrs.items()},
                    'iu_type': setup['installationDetails']['iuType'],
                    'configured_fibers': setup.get('fibres'), 'sample_rate_hz': fs}
            metadata.append(meta)
            starts = np.linspace(0, ds.shape[0] - 2048, 5, dtype=int)
            if ds.shape[0] < 2048 or len(set(starts)) != 5:
                raise ValueError('Insufficient duration for predefined five windows')
            for start in starts:
                values = ds[int(start):int(start) + 2048, :].astype(np.float64)
                rows.append({'id': entry['id'], 'family': entry['family'], 'scenario': entry['scenario'],
                             'start_sample': int(start), 'nonfinite_count': int((~np.isfinite(values)).sum()),
                             'raw_phase': diagnostics(values, fs),
                             'first_difference_diagnostic_only': diagnostics(np.diff(values, axis=0), fs)})
    result = {'kind': 'quantx_waveform_descriptive_v1', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'window_samples': 2048, 'windows_per_file': 5, 'rows': rows,
              'training_updates': 0, 'model_evaluations': 0,
              'limitations': ['Neither first difference nor raw phase is declared calibrated FEBUS strain rate',
                             'Configured fiber length is not a validated channel selection map',
                             'Statistical diagnostics cannot establish transfer learning benefit',
                             'No new final holdout is established; all sampled data are exploratory']}
    (audit / 'hdf5_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    (audit / 'signal_diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    summary = []
    for meta in metadata:
        selected = [r for r in rows if r['id'] == meta['id']]
        summary.append({'family': meta['family'], 'scenario': meta['scenario'],
                        'sample_rate_hz': meta['sample_rate_hz'], 'channels': meta['shape'][1],
                        'duration_s': meta['duration_s'],
                        **{f'{scope}_{key}': float(np.median([r[scope][key] for r in selected]))
                           for scope in ('raw_phase', 'first_difference_diagnostic_only')
                           for key in ('spatial_effective_rank', 'top10_channel_energy_fraction',
                                       'top5_spectral_bin_energy_fraction', 'dominant_frequency_hz',
                                       'above_fs_quarter_energy_fraction')}})
    with (audit / 'signal_summary.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (audit / 'COMPLETE').touch()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
