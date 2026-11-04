#!/usr/bin/env python3
"""Predefined SMF external-development excerpts; EDAS signal datasets remain untouched."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acquisition.audit_external_hdf5_headers import BoundedRangeReader
from finetuning.representation_study import digest, write_json


def select_smf(metadata):
    selected = []
    for orientation in ('Horizontal', 'Vertical'):
        for frequency in (200, 1000, 10000):
            choices = []
            for item in metadata['candidate_files']:
                folder = item['candidate_folder']
                match = re.search(r'(\d+)Hz', item['name'])
                if folder['fiber'] == 'SMF' and folder['orientation'] == orientation and match and int(match[1]) == frequency:
                    choices.append(item)
            if not choices:
                raise ValueError('Missing preregistered SMF frequency/orientation')
            item = min(choices, key=lambda x: x['id'])
            selected.append({**item, 'stimulus_label_hz': frequency})
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash-screen-dir', type=Path, required=True)
    parser.add_argument('--provenance-dir', type=Path, required=True)
    parser.add_argument('--token-path', type=Path, default=Path.home() / '.config/gdrive-vqvae/token.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2.credentials import Credentials

    metadata = json.loads((args.hash_screen_dir / 'drive_metadata.json').read_text())
    provenance = json.loads((args.provenance_dir / 'provenance_summary.json').read_text())
    assert provenance['candidates_with_development_clock_interval_overlap'] == 0
    selected = select_smf(metadata)
    smf = next(g for g in provenance['reserved_acquisition_attribute_groups'] if g['fibers'] == ['SMF'])
    assert {r['id'] for r in selected} <= set(smf['candidate_ids'])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'kind': 'external_smf_full_zone_stress_v2_numeric_audit', 'source_sha256': digest(Path(__file__)),
        'provenance_sha256': digest(args.provenance_dir / 'provenance_summary.json'),
        'selected_files': selected, 'external_development_group_ids': smf['candidate_ids'],
        'signal_unseen_reserved_groups': [g for g in provenance['reserved_acquisition_attribute_groups'] if g['fibers'] != ['SMF']],
        'selection': 'six metadata-fixed files, orientations H/V x stimulus labels 200/1000/10000Hz; first block, central 1s samples [500:20500]',
        'coverage': 'all 551 supplied Zone1 spatial rows; no amplitude-based channel removal; 512-row tiling planned without gaps',
        'limits': {'file_byte_budget': 64 * 1024 * 1024, 'range_block_size': 4 * 1024 * 1024},
        'group_role': 'entire SMF acquisition-attribute group becomes external DEVELOPMENT, never call remaining SMF files untouched final holdout',
        'parameter_updates': 0, 'limitations': ['one acquisition-attribute group; frequencies/orientations not independent replicates',
            'full supplied zone differs from historically amplitude-filtered development coverage',
            'metadata does not prove physical independence; no four-family external validation claim',
            'frequency in filename is stimulus label, not sample rate']}
    write_json(out / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    credentials = Credentials.from_authorized_user_file(args.token_path, ['https://www.googleapis.com/auth/drive.readonly'])
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError('Existing OAuth unavailable; no interactive login attempted')
    session = AuthorizedSession(credentials)
    records = []
    for index, item in enumerate(selected):
        reader = BoundedRangeReader(session, f'https://www.googleapis.com/drive/v3/files/{item["id"]}?alt=media',
            int(item['size']), byte_budget=64 * 1024 * 1024, block_size=4 * 1024 * 1024)
        with h5py.File(reader, 'r') as handle:
            names = []
            handle.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) and name.endswith('/Strain Rate [nStrain|s]') else None)
            assert len(names) == 1
            dataset = handle[names[0]]
            spacing = np.asarray(dataset.parent.attrs['Spacing'], dtype=float)
            origin = np.asarray(dataset.parent.attrs['Origin'], dtype=float)
            extent = np.asarray(dataset.parent.attrs['Extent'], dtype=float)
            assert dataset.ndim == 3 and dataset.shape[1] >= 20500 and dataset.shape[2] == 551
            assert abs(spacing[1] * .001 - .00005) < 1e-12
            raw = np.asarray(dataset[0, 500:20500, :], dtype=np.float32).T.copy()
            assert raw.shape == (551, 20000)
            clock = float(handle[names[0].rsplit('/', 2)[0] + '/time'][0])
        filename = f'{index:02d}_{item["id"]}.npz'
        np.savez_compressed(out / filename, data=raw)
        finite = np.isfinite(raw)
        patterns, pattern_counts = np.unique(raw[~finite].view(np.uint32), return_counts=True)
        records.append({'drive_id': item['id'], 'title': item['name'], 'orientation': item['candidate_folder']['orientation'],
            'stimulus_label_hz': item['stimulus_label_hz'], 'path': filename, 'sha256': digest(out / filename),
            'shape': list(raw.shape), 'dataset': names[0], 'dt_s': spacing[1] * .001, 'dx_m': spacing[0],
            'start_position_m': float(origin[0] + extent[0] * spacing[0]), 'block': 0, 'time_start': 500,
            'clock_interval_s': [clock + origin[1] * .001 + .025, clock + origin[1] * .001 + 1.025],
            'finite_cells': int(finite.sum()), 'nonfinite_cells': int((~finite).sum()),
            'fully_finite_rows': int(finite.all(axis=1).sum()),
            'nonfinite_float32_bit_patterns': {f'{int(p):08x}': int(n) for p, n in zip(patterns, pattern_counts)},
            'fetched_bytes': reader.fetched_bytes, 'range_requests': reader.requests})
        reader.close()
        write_json(out / 'samples.json', records)
        print(json.dumps({'file': index + 1, 'shape': list(raw.shape), 'network_bytes': records[-1]['fetched_bytes']}), flush=True)
    session.close()
    write_json(out / 'results.json', {'files': len(records), 'input_cells': sum(np.prod(r['shape']).item() for r in records),
        'network_bytes': sum(r['fetched_bytes'] for r in records), 'signal_dataset_reads': 'one fixed central 1s block excerpt per selected SMF file',
        'EDAS_signal_dataset_reads': 0, 'neural_inference': False, 'parameter_updates': 0,
        'numeric_audit_only_until_codec_contract_review': any(r['nonfinite_cells'] for r in records),
        'nonfinite_cells': sum(r['nonfinite_cells'] for r in records)})
    (out / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
