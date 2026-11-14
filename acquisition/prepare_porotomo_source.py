#!/usr/bin/env python3
"""Prepare a bounded, preregistered multi-day PoroTomo source without model selection."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acquisition.audit_porotomo_pretraining import sampling_from_file

BASE = 'https://nrel-pds-porotomo.s3.amazonaws.com/'
DAYS = ('20160318', '20160320', '20160323', '20160326')
PER_DAY = 15
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def spread_selection(files, count):
    ordered = sorted(files, key=lambda r: r['key'])
    if len(ordered) < count or count < 2:
        raise ValueError('Insufficient files for predefined day-stratum sampling')
    return [ordered[int(i)] for i in np.linspace(0, len(ordered) - 1, count, dtype=int)]


def normalize_rate(values):
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Finite original time x channel samples required; no silent imputation')
    mean, scale = float(raw.mean()), float(raw.std())
    normalized = ((raw - mean) / (scale if scale > 0 else 1.)).T.astype(np.float32)
    return normalized, mean, scale


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if shutil.disk_usage(args.raw_dir.parent).free < 8 * 1024**3:
        raise RuntimeError('8 GiB headroom required; no automatic removal of other data')
    session = requests.Session()
    selected, counts, exclusions = [], {}, []
    for day in DAYS:
        entries, token = [], None
        while True:
            params = {'list-type': '2', 'prefix': f'DAS/H5/DASV/{day}/', 'max-keys': 1000}
            if token:
                params['continuation-token'] = token
            response = session.get(BASE, params=params, timeout=60); response.raise_for_status()
            root = ET.fromstring(response.content)
            for node in root.findall('s:Contents', NS):
                key = node.find('s:Key', NS).text
                if key.endswith('.h5'):
                    entries.append({'key': key, 'size': int(node.find('s:Size', NS).text),
                                    'etag': node.find('s:ETag', NS).text, 'day_stratum': day})
            token_node = root.find('s:NextContinuationToken', NS)
            token = token_node.text if token_node is not None else None
            if not token:
                break
        eligible = []
        for entry in entries:
            if entry['size'] <= 65536:
                # Inspect structure, not waveform values or reconstruction quality.
                response = session.get(BASE + entry['key'], timeout=60); response.raise_for_status()
                if len(response.content) != entry['size']:
                    raise ValueError('Small-container structural inspection size mismatch')
                with h5py.File(io.BytesIO(response.content), 'r') as h:
                    has_signal = ('das' in h and 't' in h and isinstance(h['das'], h5py.Dataset)
                                  and h['das'].ndim == 2 and h['das'].size > 0)
                if not has_signal:
                    exclusions.append({**entry, 'reason': 'HDF5 lacks nonempty das/time waveform structure',
                                       'sha256': hashlib.sha256(response.content).hexdigest()})
                    continue
            eligible.append(entry)
        counts[day] = {'listed_hdf5': len(entries), 'structurally_eligible_candidates': len(eligible)}
        selected += spread_selection(eligible, PER_DAY)
    if sum(r['size'] for r in selected) > 3 * 1024**3:
        raise ValueError('Predefined 3 GiB raw download budget exceeded')
    protocol = {'kind': 'porotomo_multiday_source_preparation_v2', 'days': list(DAYS), 'files_per_day': PER_DAY,
                'selection': 'Even lexicographic quantiles of complete day-prefix inventories, before waveform inspection',
                'source_groups': 'Four directory-date strata, NOT four fiber families or proven independent acquisitions',
                'prefix_file_counts': counts, 'selected_files': selected, 'window_samples': 1000,
                'structural_exclusions': exclusions,
                'eligibility': 'Inspect every HDF5 object <=64KiB for nonempty das/time structure before fixed-quantile selection; larger selected objects must pass full structure/provenance/cadence checks. No waveform-quality exclusion.',
                'source_sample_rate_hz': 1000, 'source_window_duration_s': 1.,
                'normalization': 'Float64 global window mean/std; unit-independent positive-std scaling; float32 model inputs; constant windows retained as zeros',
                'source_quantity': 'Native iDAS trace amplitude proportional to fiber strain-rate, not absolute FEBUS calibration',
                'native_identity_evidence': 'porotomo_native_identity_20261110: full HDF5/native SEG-Y bit identity and matching source SHA3-256',
                'preservation': 'All recorded channels and complete nonoverlapping source windows; no filtering, clipping, denoising or cross-file concatenation',
                'clock_caveat': 'Use file units/interval for within-file cadence; absolute HDF5 conversion clocks can disagree with native header',
                'license': 'CC-BY-4.0, https://gdr.openei.org/submissions/980',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'parameter_updates': 0, 'model_evaluations': 0, 'final_holdout': False}
    if args.resume:
        if json.loads((args.output_dir / 'protocol.json').read_text()) != protocol:
            raise ValueError('Source preparation resume protocol changed')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        args.samples_dir.mkdir(parents=True, exist_ok=False)
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / 'protocol.json', protocol)
        shutil.copy2(Path(__file__), args.output_dir / Path(__file__).name)
    records, source_summaries = [], []
    native_sources = set()
    for index, item in enumerate(selected):
        path = args.raw_dir / item['key'].rsplit('/', 1)[1]
        if not path.exists():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=args.raw_dir, prefix=path.name + '.', suffix='.partial', delete=False) as target:
                    temporary = Path(target.name)
                    received = 0
                    with session.get(BASE + item['key'], stream=True, timeout=120) as response:
                        response.raise_for_status()
                        for block in response.iter_content(1024 * 1024):
                            received += len(block)
                            if received > item['size']:
                                raise ValueError('Source exceeds declared bounded object size')
                            target.write(block)
                if temporary.stat().st_size != item['size']:
                    raise ValueError('Source download size mismatch')
                temporary.rename(path); temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink()
        if path.stat().st_size != item['size']:
            raise ValueError('Existing source file size differs; refusing overwrite')
        with path.open('rb') as stream:
            source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        source_id = f"DASV_{item['day_stratum']}_{path.stem}"
        with h5py.File(path, 'r') as h:
            native_checksum = tuple(str(v) for v in h.attrs.get('source_checksum', []))
            if not native_checksum or native_checksum in native_sources:
                raise ValueError('Missing or duplicate native source provenance; do not silently expand exposure')
            native_sources.add(native_checksum)
            data = h['das']; clock = h['t'][...].reshape(-1)
            rate, clock_steps = sampling_from_file(clock, h['t'].attrs.get('units', ''), data.attrs['sampling_interval_seconds'])
            if rate != 1000 or data.ndim != 2 or data.shape[0] != len(clock):
                raise ValueError('Unexpected source cadence or axes')
            channels = h['channel'][...].reshape(-1)
            if len(channels) != data.shape[1] or not np.all(np.diff(channels) == 1):
                raise ValueError('Channel gaps require an explicit new preparation policy')
            windows = data.shape[0] // 1000
            for window in range(windows):
                values, mean, scale = normalize_rate(data[window * 1000:(window + 1) * 1000, :])
                sample = args.samples_dir / f'{source_id}_w{window:04d}.npz'
                if sample.exists():
                    with np.load(sample, allow_pickle=False) as old:
                        if not np.array_equal(old['values'], values):
                            raise ValueError('Existing prepared values differ; refusing overwrite')
                else:
                    np.savez_compressed(sample, values=values)
                records.append({'sample_name': sample.name, 'source_id': source_id,
                    'family': 'DASV_day_' + item['day_stratum'], 'scenario': path.stem,
                    'time_start': window * 1000, 'channels': values.shape[0], 'time_samples': 1000,
                    'mean_original': mean, 'std_original': scale, 'sha256': hashlib.sha256(sample.read_bytes()).hexdigest()})
            source_summaries.append({**item, 'source_id': source_id, 'sha256': source_hash,
                'shape': list(data.shape), 'windows': windows,
                'omitted_suffix_samples': data.shape[0] % 1000,
                'clock_step_min_seconds': float(clock_steps.min()), 'clock_step_max_seconds': float(clock_steps.max()),
                'hdf_time_coverage_start': str(h.attrs.get('time_coverage_start', '')),
                'hdf_source_checksum': list(native_checksum)})
        write_json(args.output_dir / 'source_files.json', source_summaries)
        print('Prepared',index+1,'/',len(selected),item['key'],windows,'windows',flush=True)
    manifest = args.samples_dir / 'source_manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = {'source_files': len(source_summaries), 'windows': len(records),
               'raw_bytes': sum(r['size'] for r in selected),
               'prepared_bytes': sum(path.stat().st_size for path in args.samples_dir.glob('*.npz')),
               'constant_source_windows': sum(row['std_original'] == 0 for row in records),
               'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
               'parameter_updates': 0, 'final_holdout': False}
    write_json(args.output_dir / 'summary.json', summary)
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps(summary, indent=2),flush=True)


if __name__ == '__main__':
    main()
