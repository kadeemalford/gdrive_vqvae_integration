#!/usr/bin/env python3
"""Bounded CPU preflight of one public PoroTomo iDAS differential-phase recording."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acquisition.analyze_quantx_pretraining import diagnostics
from acquisition.audit_external_hdf5_headers import describe_metadata

URL = 'https://nrel-pds-porotomo.s3.amazonaws.com/DAS/H5/DASV/20160317/PoroTomo_iDAS025_160318160317.h5'
SIZE = 46590272
README = 'https://raw.githubusercontent.com/openEDI/documentation/main/PoroTomo/README.md'


def resolve_metadata_references(value, handle):
    if isinstance(value, dict):
        return {k: resolve_metadata_references(v, handle) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolve_metadata_references(v, handle) for v in value]
    if isinstance(value, h5py.Reference):
        return {'hdf5_reference': handle[value].name if value else None}
    return value


def sampling_from_file(timestamps, units, interval_seconds):
    """Prefer file units/interval over generic README; tolerate epoch-float rounding."""
    if isinstance(units, bytes):
        units = units.decode('utf-8')
    text = units.lower()
    if text.startswith('seconds'):
        factor = 1.
    elif text.startswith('microseconds'):
        factor = 1e-6
    else:
        raise ValueError('Unsupported or missing file timestamp units')
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    differences = np.diff(times) * factor
    interval = float(interval_seconds)
    if interval <= 0 or not np.isfinite(interval) or not len(differences) or np.any(differences <= 0):
        raise ValueError('Invalid file sampling interval/clock')
    span_interval = (times[-1] - times[0]) * factor / (len(times) - 1)
    if not np.isclose(span_interval, interval, rtol=1e-6, atol=1e-9):
        raise ValueError('File clock and sampling interval disagree')
    return 1 / interval, differences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--reuse-downloaded', action='store_true')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    protocol = {'kind': 'porotomo_pretraining_preflight_v1', 'url': URL, 'expected_size': SIZE,
                'selection': 'First lexicographic HDF5 in first DASV daily prefix; selected before signal inspection',
                'readme_url': README, 'landing_page': 'https://gdr.openei.org/submissions/980',
                'license': 'CC-BY-4.0 according to GDR landing page',
                'training_updates': 0, 'model_evaluations': 0,
                'reuse_existing_download': args.reuse_downloaded,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    session = requests.Session()
    response = session.get(README, timeout=60); response.raise_for_status()
    (args.output_dir / 'source_README.md').write_text(response.text)
    path = args.raw_dir / URL.rsplit('/', 1)[1]
    if not args.reuse_downloaded:
        received = 0
        with session.get(URL, stream=True, timeout=90) as response, path.open('xb') as stream:
            response.raise_for_status()
            for block in response.iter_content(1024 * 1024):
                received += len(block)
                if received > SIZE:
                    raise ValueError('Download exceeds bounded source size')
                stream.write(block)
    if path.stat().st_size != SIZE:
        raise ValueError('Download size mismatch')
    received = path.stat().st_size
    with path.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    with h5py.File(path, 'r') as handle:
        metadata = resolve_metadata_references(describe_metadata(handle), handle)
        data, timestamps = handle['das'], handle['t'][...].reshape(-1)
        fs, difference = sampling_from_file(timestamps, handle['t'].attrs.get('units', ''),
                                             data.attrs['sampling_interval_seconds'])
        median_dt = float(np.median(difference))
        if data.shape[0] == len(timestamps):
            time_axis = 0
        elif data.shape[1] == len(timestamps):
            time_axis = 1
        else:
            raise ValueError('Source time axis cannot be identified')
        shape, windows = list(data.shape), []
        for start in np.linspace(0, len(timestamps) - 2048, 5, dtype=int):
            values = data[int(start):int(start) + 2048, :] if time_axis == 0 else data[:, int(start):int(start) + 2048].T
            values = values.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite source waveform; preserve for inspection, do not silently impute')
            windows.append({'start_sample': int(start), 'raw_product': diagnostics(values, fs)})
    result = {'bytes': received, 'sha256': source_hash, 'source_shape': shape, 'time_axis': time_axis,
              'sample_rate_hz_from_file_metadata_and_clock': fs,
              'timestamp_step_min': float(difference.min()), 'timestamp_step_median': median_dt,
              'timestamp_step_max': float(difference.max()), 'windows': windows, 'metadata': metadata,
              'limitations': ['Single well/time segment, not full-dataset characterization',
                             'Differential phase per sample per source documentation; not declared absolutely calibrated FEBUS strain-rate',
                             'Directory date and file name differ; acquisition independence not inferred from paths',
                             'No compression model evaluation or transfer benefit established']}
    (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps({k:v for k,v in result.items() if k not in ('windows','metadata')}, indent=2))


if __name__ == '__main__':
    main()
