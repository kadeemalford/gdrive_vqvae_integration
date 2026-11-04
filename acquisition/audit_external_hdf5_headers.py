#!/usr/bin/env python3
"""Bounded HTTP-range HDF5 metadata inspection, without reading signal datasets."""
import argparse
import io
import json
import re
from collections import defaultdict
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.representation_study import digest, write_json


class BoundedRangeReader(io.RawIOBase):
    def __init__(self, session, url, size, byte_budget=2 * 1024 * 1024, block_size=16384):
        super().__init__()
        self.session, self.url, self.size = session, url, int(size)
        self.byte_budget, self.block_size = byte_budget, block_size
        self.position, self.fetched_bytes, self.requests = 0, 0, 0
        self.blocks = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError('Invalid seek mode')
        value = offset + (self.position if whence == io.SEEK_CUR else self.size if whence == io.SEEK_END else 0)
        if value < 0:
            raise ValueError('Negative seek')
        self.position = value
        return value

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Unbounded reads are forbidden')
        size = max(0, min(size, self.size - self.position))
        result = bytearray()
        while size:
            start = self.position // self.block_size * self.block_size
            if start not in self.blocks:
                end = min(self.size, start + self.block_size) - 1
                expected = end - start + 1
                if self.fetched_bytes + expected > self.byte_budget:
                    raise ValueError('HDF5 metadata byte budget exceeded')
                response = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}',
                    'Accept-Encoding': 'identity'}, stream=True, timeout=60)
                try:
                    if response.status_code != 206:
                        raise ValueError('Server did not honor bounded range request')
                    if response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                        raise ValueError('Unexpected range identity/size')
                    block = response.raw.read(expected + 1)
                    if len(block) != expected:
                        raise ValueError('Unexpected range response length')
                finally:
                    response.close()
                self.blocks[start] = block
                self.fetched_bytes += len(block)
                self.requests += 1
            block = self.blocks[start]
            offset = self.position - start
            take = min(size, len(block) - offset)
            result.extend(block[offset:offset + take])
            self.position += take
            size -= take
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def json_attribute(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        if value.size > 256:
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'values_omitted': True}
        return [json_attribute(v) for v in value.tolist()] if value.ndim else json_attribute(value.item())
    if isinstance(value, (list, tuple)):
        return [json_attribute(v) for v in value]
    if isinstance(value, np.generic):
        return json_attribute(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def describe_metadata(handle, clock_metadata=False):
    objects = []
    def visit(name, obj):
        entry = {'path': name, 'kind': 'dataset' if isinstance(obj, h5py.Dataset) else 'group',
                 'attributes': {key: json_attribute(value) for key, value in obj.attrs.items()}}
        if isinstance(obj, h5py.Dataset):
            entry.update(shape=list(obj.shape), dtype=str(obj.dtype), chunks=list(obj.chunks) if obj.chunks else None)
            if (clock_metadata and re.fullmatch(r'[^/]+/Source\d+/time', name) and obj.ndim == 1
                    and obj.size <= 256 and np.issubdtype(obj.dtype, np.floating)):
                entry['clock_metadata_values'] = json_attribute(obj[...])
        objects.append(entry)
    handle.visititems(visit)
    return {'root_attributes': {key: json_attribute(value) for key, value in handle.attrs.items()}, 'objects': objects}


def acquisition_screen(records):
    development, candidates = [], []
    for record in records:
        if record['status'] != 'metadata_read':
            continue
        attributes = next((o['attributes'] for o in record['objects'] if 'AcqStartTime' in o['attributes'] and 'Hostname' in o['attributes']), {})
        start = attributes.get('AcqStartTime')
        start = start[0] if isinstance(start, list) and start else start
        clocks = [value for obj in record['objects'] for value in obj.get('clock_metadata_values', []) if value is not None]
        row = {'id': record['id'], 'host': attributes.get('Hostname'), 'acq_start_time': start,
               'clock_values': clocks, 'folder': record.get('folder')}
        (development if record.get('role') == 'development' else candidates).append(row)
    grouped, matches = defaultdict(list), []
    for item in candidates:
        key = (item['host'], item['acq_start_time'])
        grouped[key].append(item['id'])
        shared_session = [d['id'] for d in development if key[0] and key[1] is not None and (d['host'], d['acq_start_time']) == key]
        shared_clock = [d['id'] for d in development if item['host'] and d['host'] == item['host'] and
                        set(item['clock_values']) & set(d['clock_values'])]
        matches.append({'id': item['id'], 'same_development_host_acq_start_ids': shared_session,
                        'same_development_recorded_clock_ids': shared_clock})
    return {'candidate_files_read': len(candidates), 'development_files_read': len(development),
            'candidate_acquisition_attribute_groups': [{'hostname': key[0], 'acq_start_time': key[1], 'ids': ids} for key, ids in grouped.items()],
            'candidate_shared_development_acq_attribute_count': sum(bool(r['same_development_host_acq_start_ids']) for r in matches),
            'candidate_shared_development_recorded_clock_count': sum(bool(r['same_development_recorded_clock_ids']) for r in matches),
            'matches': matches, 'independence_proven': False,
            'limitation': 'AcqStartTime may be session-level or stale; exact clock equality is a linkage warning, not complete interval-overlap coverage. Keep related records grouped conservatively.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash-screen-dir', type=Path, required=True)
    parser.add_argument('--token-path', type=Path, default=Path.home() / '.config/gdrive-vqvae/token.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--all-candidates', action='store_true')
    parser.add_argument('--clock-metadata', action='store_true')
    parser.add_argument('--development-raw-dir', type=Path)
    args = parser.parse_args()
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2.credentials import Credentials

    metadata = json.loads((args.hash_screen_dir / 'drive_metadata.json').read_text())
    grouped = {}
    for item in metadata['candidate_files']:
        grouped.setdefault(item['candidate_folder']['id'], []).append(item)
    selected = (sorted(metadata['candidate_files'], key=lambda item: item['id']) if args.all_candidates else
                [min(items, key=lambda item: item['id']) for _, items in sorted(grouped.items())])
    credentials = Credentials.from_authorized_user_file(args.token_path, ['https://www.googleapis.com/auth/drive.readonly'])
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError('Existing OAuth credentials unavailable; no interactive login attempted')
    session = AuthorizedSession(credentials)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'protocol.json', {'kind': 'bounded_external_hdf5_metadata_v2',
        'source_sha256': digest(Path(__file__)), 'hash_screen_sha256': digest(args.hash_screen_dir / 'results.json'),
        'selected_ids': [r['id'] for r in selected], 'selection': 'all candidate files' if args.all_candidates else 'lexicographically smallest Drive ID per candidate folder',
        'byte_budget_per_file': 2 * 1024 * 1024, 'block_size': 16384,
        'signal_dataset_array_reads': False, 'clock_metadata_array_reads': args.clock_metadata, 'model_evaluations': 0, 'parameter_updates': 0,
        'local_development_metadata_ids': [r['id'] for r in metadata['development_files']] if args.development_raw_dir else [],
        'limitations': ['metadata only, not waveform fidelity or independence validation',
            'HDF5 metadata IO requests may fetch adjacent bytes in bounded pages; signal datasets are never read as arrays; optional whitelisted clock arrays only',
            'no final holdout declaration; acquisition links and broader coverage still require review']})
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    results = []
    for item in selected:
        reader = BoundedRangeReader(session, f'https://www.googleapis.com/drive/v3/files/{item["id"]}?alt=media', int(item['size']))
        record = {'id': item['id'], 'name': item['name'], 'folder': item['candidate_folder'], 'role': 'candidate', 'objects': []}
        try:
            with h5py.File(reader, 'r') as handle:
                record.update(describe_metadata(handle, args.clock_metadata))
            record['status'] = 'metadata_read'
        except Exception as error:
            record['status'] = 'metadata_incomplete'
            # Record only the exception type, never request/authorization headers.
            record['error_type'] = type(error).__name__
        finally:
            record['fetched_bytes'] = reader.fetched_bytes
            record['range_requests'] = reader.requests
            reader.close()
        results.append(record)
        print(json.dumps({'id': item['id'], 'status': record['status'], 'bytes': record['fetched_bytes']}), flush=True)
    session.close()
    if args.development_raw_dir:
        for item in metadata['development_files']:
            record = {'id': item['id'], 'name': item['name'], 'role': 'development', 'fetched_bytes': 0}
            try:
                with h5py.File(args.development_raw_dir / f'{item["id"]}.h5', 'r') as handle:
                    record.update(describe_metadata(handle, args.clock_metadata))
                record['status'] = 'metadata_read'
            except Exception as error:
                record['status'], record['error_type'] = 'metadata_incomplete', type(error).__name__
            results.append(record)
    write_json(out / 'results.json', {'files': results, 'signal_dataset_array_reads': False,
        'clock_metadata_array_reads': args.clock_metadata,
        'acquisition_screen': acquisition_screen(results),
        'total_fetched_bytes': sum(r['fetched_bytes'] for r in results), 'external_holdout_validated': False})
    (out / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
