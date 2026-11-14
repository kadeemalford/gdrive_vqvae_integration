#!/usr/bin/env python3
"""Inventory QuantX candidates and download a bounded, predefined CPU audit sample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials

ROOT_ID = '16HRc9KiVRAMSDXERcRybeX1ek2k30Cvr'
FOLDER_TYPE = 'application/vnd.google-apps.folder'
SCENARIOS = {'segmento1_3m_200hz', 'segmento3_4m_500hz', 'segmento4_5m_1000hz'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--token-path', type=Path, default=Path.home() / '.config/gdrive-vqvae/token.json')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'protocol.json', {
        'kind': 'quantx_pretraining_cpu_audit_v1', 'root_id': ROOT_ID,
        'selection': 'Largest HDF5 in each predefined scenario, for all four fiber folders',
        'scenarios': sorted(SCENARIOS), 'max_download_bytes': 1600000000,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_updates': 0, 'model_evaluations': 0,
        'holdout_claim': False, 'drive_writes': False})
    creds = Credentials.from_authorized_user_file(args.token_path, ['https://www.googleapis.com/auth/drive.readonly'])
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds.valid:
        raise RuntimeError('Existing read-only OAuth credentials unavailable')
    session = AuthorizedSession(creds)

    def children(folder):
        files, token = [], None
        while True:
            params = {'q': f"'{folder}' in parents and trashed = false", 'pageSize': 1000,
                      'fields': 'nextPageToken,files(id,name,mimeType,size,md5Checksum,description)'}
            if token:
                params['pageToken'] = token
            for attempt in range(4):
                response = session.get('https://www.googleapis.com/drive/v3/files', params=params, timeout=90)
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(2 ** attempt)
            response.raise_for_status()
            result = response.json()
            files.extend(result.get('files', []))
            token = result.get('nextPageToken')
            if not token:
                return files

    inventory, selected = [], []
    for family in sorted(children(ROOT_ID), key=lambda r: r['name']):
        if family['mimeType'] != FOLDER_TYPE:
            continue
        for scenario in sorted(children(family['id']), key=lambda r: r['name']):
            if scenario['mimeType'] != FOLDER_TYPE:
                continue
            entries = children(scenario['id'])
            for entry in entries:
                inventory.append({**entry, 'family': family['name'], 'scenario': scenario['name'],
                                  'scenario_id': scenario['id']})
            if scenario['name'] in SCENARIOS:
                candidates = [r for r in inventory if r['scenario_id'] == scenario['id'] and r['name'].endswith('.h5')]
                if candidates:
                    selected.append(max(candidates, key=lambda r: (int(r['size']), r['id'])))
                selected.extend(r for r in inventory if r['scenario_id'] == scenario['id'] and
                                r['name'] in ('IUSetupConfig.JSON', 'contents.xml') or
                                r['scenario_id'] == scenario['id'] and r['name'].startswith('TimeLog_'))
            write_json(args.output_dir / 'inventory.json', inventory)
        print('Inventoried:', family['name'], flush=True)
    total = sum(int(r['size']) for r in selected)
    if total > 1600000000:
        raise ValueError('Predefined download budget exceeded')
    write_json(args.output_dir / 'selected.json', selected)
    downloads = []
    for entry in selected:
        directory = args.raw_dir / entry['family'] / entry['scenario']
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / entry['name']
        if path.exists():
            raise FileExistsError(path)
        md5, sha256 = hashlib.md5(), hashlib.sha256()
        with session.get(f"https://www.googleapis.com/drive/v3/files/{entry['id']}",
                         params={'alt': 'media'}, stream=True, timeout=120) as response:
            response.raise_for_status()
            with path.open('xb') as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
                    md5.update(block)
                    sha256.update(block)
                    if stream.tell() > int(entry['size']):
                        raise ValueError('Download exceeded declared file size')
        if path.stat().st_size != int(entry['size']) or md5.hexdigest() != entry['md5Checksum']:
            raise ValueError('Download integrity mismatch')
        downloads.append({**entry, 'local_path': str(path), 'sha256': sha256.hexdigest(), 'md5_verified': True})
        write_json(args.output_dir / 'downloads.json', downloads)
    write_json(args.output_dir / 'summary.json', {
        'files': len(inventory), 'hdf5_files': sum(r['name'].endswith('.h5') for r in inventory),
        'families': dict(Counter(r['family'] for r in inventory)),
        'scenario_groups': len({r['scenario_id'] for r in inventory}),
        'inventory_bytes': sum(int(r.get('size', 0)) for r in inventory),
        'download_bytes': total, 'downloaded_hdf5': sum(r['name'].endswith('.h5') for r in downloads)})
    (args.output_dir / 'DOWNLOAD_COMPLETE').touch()


if __name__ == '__main__':
    main()
