#!/usr/bin/env python3
"""Prepare raw-phase source windows for a bounded, explicitly cross-product transfer pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np


def normalize_phase(values):
    """Center/scale in float64 before float32 conversion; never differentiate or denoise."""
    raw = np.asarray(values)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('Expected original integer time x locus phase array')
    x = raw.astype(np.float64).T
    mean, scale = float(x.mean()), float(x.std())
    if not np.isfinite(mean) or not np.isfinite(scale):
        raise ValueError('Nonfinite phase statistics')
    normalized = ((x - mean) / (scale + 1e-6)).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError('Nonfinite normalized source phase')
    return normalized, mean, scale


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not (args.audit_dir / 'COMPLETE').exists():
        raise ValueError('Audited QuantX inputs required')
    if shutil.disk_usage(args.samples_dir.parent).free < 3 * 1024**3:
        raise RuntimeError('At least 3 GiB disk headroom required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.samples_dir.mkdir(parents=True, exist_ok=False)
    downloads = json.loads((args.audit_dir / 'downloads.json').read_text())
    source_files = [r for r in downloads if r['name'].endswith('.h5')]
    protocol = {'kind': 'quantx_raw_phase_bridge_preparation_v1', 'source_files': [r['id'] for r in source_files],
                'download_manifest_sha256': hashlib.sha256((args.audit_dir / 'downloads.json').read_bytes()).hexdigest(),
                'window_samples': 1000, 'sampling_rate_hz': 10000, 'window_duration_s': .1,
                'input_product': 'Original QuantX Diversity Processed Phase, rad * 2PI/2^16',
                'source_channel_policy': 'All recorded contiguous loci; no heuristic physical-validity mask',
                'normalization': 'Float64 global window mean/std, divide by std+1e-6, then float32; no filtering/differentiation/unwrap',
                'source_use': 'Pretraining only; not an independent external evaluation or strain-rate-equivalent target',
                'coverage_note': 'Complete nonoverlapping 1000-sample windows; trailing incomplete source windows omitted from training only, not counted as codec savings',
                'no_febus_signal_access': True, 'no_parameter_updates': True,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows, summaries = [], []
    for record in source_files:
        path = Path(record['local_path'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != record['sha256']:
                raise ValueError('Audited source SHA-256 changed')
        with h5py.File(path, 'r') as handle:
            raw = handle['Acquisition/Raw[0]']
            if float(raw.attrs['OutputDataRate']) != 10000:
                raise ValueError('Source sample rate changed')
            ds = raw['RawData']
            total = ds.shape[0] // 1000
            for index in range(total):
                values, mean, scale = normalize_phase(ds[index * 1000:(index + 1) * 1000, :])
                sample = args.samples_dir / f"{record['id']}_w{index:04d}.npz"
                np.savez_compressed(sample, values=values)
                rows.append({'sample_name': sample.name, 'source_id': record['id'], 'family': record['family'],
                             'scenario': record['scenario'], 'time_start': index * 1000,
                             'channels': values.shape[0], 'time_samples': values.shape[1],
                             'mean_phase_counts': mean, 'std_phase_counts': scale,
                             'sha256': hashlib.sha256(sample.read_bytes()).hexdigest()})
            summaries.append({'source_id': record['id'], 'family': record['family'], 'scenario': record['scenario'],
                              'source_sha256': record['sha256'], 'windows': total, 'channels': ds.shape[1],
                              'omitted_incomplete_suffix_samples': ds.shape[0] % 1000})
        print('Prepared',record['family'],record['scenario'],total,'windows',flush=True)
    manifest = args.samples_dir / 'source_manifest.csv'
    with manifest.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'windows': len(rows), 'source_files': len(source_files), 'sources': summaries,
               'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
               'prepared_bytes': sum(p.stat().st_size for p in args.samples_dir.glob('*.npz')),
               'constant_source_windows': sum(r['std_phase_counts'] == 0 for r in rows),
               'training_updates': 0, 'independence_proven': False}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copy2(Path(__file__), args.output_dir / Path(__file__).name)
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))


if __name__ == '__main__':
    main()
