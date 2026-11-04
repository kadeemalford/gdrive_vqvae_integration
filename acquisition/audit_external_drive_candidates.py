#!/usr/bin/env python3
"""Read-only Drive identity/hash screening; never download or evaluate waveforms."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.representation_study import digest, write_json

FIELDS = 'id,name,mimeType,size,md5Checksum,createdTime,modifiedTime,description,parents,originalFilename'


def duplicate_screen(candidates, development):
    by_id = {item['id']: item for item in development}
    hashes = defaultdict(list)
    for item in development:
        if item.get('md5Checksum') and item.get('size'):
            hashes[item['md5Checksum'], str(item['size'])].append(item['id'])
    internal = defaultdict(list)
    screened = []
    for item in candidates:
        key = item.get('md5Checksum'), str(item.get('size'))
        matches = hashes.get(key, []) if key[0] and item.get('size') else []
        same_id = item['id'] in by_id
        screened.append({'id': item['id'], 'name': item.get('name'), 'same_development_id': same_id,
                         'same_development_md5_size_ids': matches, 'hash_available': bool(key[0]),
                         'eligible_for_provenance_review': not same_id and not matches})
        if key[0] and item.get('size'):
            internal[key].append(item['id'])
    return {'candidates': screened,
            'same_development_id_count': sum(r['same_development_id'] for r in screened),
            'same_development_md5_size_count': sum(bool(r['same_development_md5_size_ids']) for r in screened),
            'candidate_hash_missing_count': sum(not r['hash_available'] for r in screened),
            'candidate_duplicate_clusters': [ids for ids in internal.values() if len(ids) > 1],
            'candidate_unique_md5_size_pairs': len(internal),
            'provenance_review_candidates': sum(r['eligible_for_provenance_review'] for r in screened)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--development-inventory', type=Path, default=ROOT / 'data/manifests/febus_drive_inventory.csv')
    parser.add_argument('--token-path', type=Path, default=Path.home() / '.config/gdrive-vqvae/token.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    candidates = json.loads(args.candidates.read_text())
    with args.development_inventory.open() as stream:
        development_ids = sorted({r['drive_id'] for r in csv.DictReader(stream)})
    credentials = Credentials.from_authorized_user_file(args.token_path, ['https://www.googleapis.com/auth/drive.readonly'])
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError('Existing read-only OAuth credentials are unavailable; no interactive login attempted')
    # Refresh is in memory only: do not overwrite the user's token or print it.
    service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'protocol.json', {'kind': 'external_drive_hash_screen_v1', 'waveforms_downloaded': False,
        'waveforms_evaluated': False, 'drive_writes': False, 'source_sha256': digest(Path(__file__)),
        'candidate_manifest_sha256': digest(args.candidates), 'development_inventory_sha256': digest(args.development_inventory),
        'candidate_folder_ids': [f['id'] for f in candidates['candidate_folders']], 'development_ids': development_ids,
        'fields': FIELDS, 'limitations': ['Drive timestamps are not acquisition timestamps',
            'MD5+size screening is not SHA-256 verification or acquisition-independence proof',
            'hash-distinct reprocessed/simultaneous records may still belong to the same acquisition',
            'EDAS/SMF candidates do not independently cover all four development families']})
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    listed, development, errors = [], [], []
    for folder in candidates['candidate_folders']:
        token = None
        while True:
            response = service.files().list(q=f"'{folder['id']}' in parents and trashed = false",
                pageToken=token, pageSize=1000, fields=f'nextPageToken,files({FIELDS})').execute()
            for item in response.get('files', []):
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    raise ValueError('Unexpected nested candidate folder; explicit manifest review required')
                listed.append({**item, 'candidate_folder': folder})
            token = response.get('nextPageToken')
            if not token:
                break
    def receive(request_id, response, exception):
        if exception:
            errors.append({'id': request_id, 'status': getattr(getattr(exception, 'resp', None), 'status', None)})
        else:
            development.append(response)
    for start in range(0, len(development_ids), 100):
        batch = service.new_batch_http_request(callback=receive)
        for file_id in development_ids[start:start + 100]:
            batch.add(service.files().get(fileId=file_id, fields=FIELDS), request_id=file_id)
        batch.execute()
    write_json(out / 'drive_metadata.json', {'candidate_files': listed, 'development_files': development, 'errors': errors})
    result = duplicate_screen(listed, development)
    result.update(candidate_files=len(listed), development_files_requested=len(development_ids),
        development_files_read=len(development), metadata_errors=errors,
        development_hash_missing_count=sum(not r.get('md5Checksum') for r in development),
        external_holdout_validated=False, final_holdout_selection_performed=False,
        next_requirement='Acquisition provenance and HDF5 physical compatibility, before independent holdout declaration')
    write_json(out / 'results.json', result)
    (out / 'COMPLETE').touch()
    print(json.dumps({k: v for k, v in result.items() if k not in ('candidates', 'candidate_duplicate_clusters')}, indent=2))


if __name__ == '__main__':
    main()
