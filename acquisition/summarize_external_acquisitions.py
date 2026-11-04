#!/usr/bin/env python3
"""Conservative acquisition grouping and metadata-derived clock interval screening."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.representation_study import digest, write_json


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end)])
    return merged


def overlap_seconds(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    overlap = 0.
    while i < len(left) and j < len(right):
        overlap += max(0., min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def metadata_intervals(record):
    objects = {obj['path']: obj for obj in record.get('objects', [])}
    intervals, schemas = [], []
    for path, obj in objects.items():
        shape = obj.get('shape', [])
        if not path.split('/')[-1].startswith('Strain Rate') or len(shape) != 3:
            continue
        parent, source = path.rsplit('/', 1)[0], path.rsplit('/', 2)[0]
        attrs = objects[parent]['attributes']
        spacing, origin = attrs.get('Spacing', []), attrs.get('Origin', [])
        clocks = objects.get(source + '/time', {}).get('clock_metadata_values', [])
        if len(spacing) < 2 or len(origin) < 2 or len(clocks) != shape[0]:
            continue
        dt, offset = float(spacing[1]) * .001, float(origin[1]) * .001
        if dt <= 0 or not np.isfinite([dt, offset, *clocks]).all():
            continue
        intervals.extend((float(t) + offset, float(t) + offset + dt * shape[1]) for t in clocks)
        schemas.append({'dataset': path, 'dt_s': dt, 'dx_m': spacing[0], 'gauge_length': attrs.get('GaugeLength'),
                        'shape': shape, 'clock_unit_assumption': 'Source/time seconds, Zone Origin/Spacing temporal coordinates in milliseconds'})
    return merge_intervals(intervals), schemas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    assert (root / 'COMPLETE').exists()
    data = json.loads((root / 'results.json').read_text())
    rows = []
    for record in data['files']:
        if record['status'] != 'metadata_read':
            continue
        attrs = next((o['attributes'] for o in record['objects'] if 'AcqStartTime' in o['attributes'] and 'Hostname' in o['attributes']), {})
        start = attrs.get('AcqStartTime')
        start = start[0] if isinstance(start, list) and start else start
        intervals, schemas = metadata_intervals(record)
        rows.append({'id': record['id'], 'role': record['role'], 'host': attrs.get('Hostname'),
                     'acq_start_time': start, 'intervals': intervals, 'schemas': schemas, 'folder': record.get('folder')})
    development = [r for r in rows if r['role'] == 'development']
    candidates = [r for r in rows if r['role'] == 'candidate']
    matches = []
    for candidate in candidates:
        overlaps = [{'development_id': d['id'], 'overlap_seconds': overlap_seconds(candidate['intervals'], d['intervals'])}
                    for d in development if candidate['host'] and d['host'] == candidate['host']]
        matches.append({'candidate_id': candidate['id'], 'development_overlaps': [r for r in overlaps if r['overlap_seconds'] > 1e-6]})
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate['host'], candidate['acq_start_time']].append(candidate)
    groups = []
    for (host, start), items in sorted(grouped.items()):
        groups.append({'hostname': host, 'acq_start_time': start, 'candidate_ids': [r['id'] for r in items],
            'orientations': sorted({r['folder']['orientation'] for r in items}),
            'fibers': sorted({r['folder']['fiber'] for r in items}),
            'unique_clock_union_seconds': sum(end - begin for begin, end in merge_intervals([v for r in items for v in r['intervals']])),
            'reservation': 'keep entire acquisition-attribute group together; no final split or waveform evaluation'})
    result = {'metadata_source_sha256': digest(root / 'results.json'), 'summary_code_sha256': digest(Path(__file__)),
        'candidate_count': len(candidates), 'development_count': len(development),
        'missing_candidate_clock_intervals': sum(not r['intervals'] for r in candidates),
        'missing_development_clock_intervals': sum(not r['intervals'] for r in development),
        'candidates_with_development_clock_interval_overlap': sum(bool(r['development_overlaps']) for r in matches),
        'candidate_matches': matches, 'reserved_acquisition_attribute_groups': groups, 'file_metadata': rows,
        'final_holdout_declared': False, 'signal_arrays_read': False, 'model_evaluations': 0,
        'limitations': ['metadata-time convention must be verified against acquisition documentation',
            'no clock overlap or repeated AcqStartTime is not a proof of physical/event independence',
            'AcqStartTime may be session-level; do not split orientations of these reserved groups',
            'new EDAS/SMF domains do not independently validate all four development families']}
    write_json(root / 'provenance_summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('candidate_matches', 'reserved_acquisition_attribute_groups', 'file_metadata')}, indent=2))
    print(json.dumps([{k: v for k, v in group.items() if k != 'candidate_ids'} for group in groups], indent=2))


if __name__ == '__main__':
    main()
