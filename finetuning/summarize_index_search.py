#!/usr/bin/env python3
"""Descriptive E1 analysis with explicit post-hoc, encoder-deployable tail fallback."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import digest, write_json


def admit_search(before, after):
    return (after['relative_rmse'] <= before['relative_rmse'] and
            after['correlation'] >= before['correlation'] and
            after['ordinary_channel_p90_relative_rmse'] <= before['ordinary_channel_p90_relative_rmse'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.experiment_dir
    if not (directory / 'COMPLETE').exists():
        raise ValueError('Completed E1 experiment required')
    rows = json.loads((directory / 'records.json').read_text())
    chosen, decisions = [], []
    for index, row in enumerate(rows):
        a, b = row['baseline'], row['searched']
        eligible = admit_search(a, b)
        chosen.append(b if eligible else a)
        decisions.append({'sample_path': row['sample_path'], 'searched': eligible,
            'frame': f'{index:05d}.searched' if eligible else f'{index:05d}.baseline'})
    baseline, _ = aggregate_validation([r['baseline'] for r in rows])
    macro, groups = aggregate_validation(chosen)
    result = {'analysis': 'post-hoc strict per-window RMSE/correlation/channel-p90 fallback; not an independent confirmation',
        'analysis_source_sha256': digest(Path(__file__)), 'windows': len(rows),
        'searched_windows': sum(r['searched'] for r in decisions),
        'baseline_macro': baseline, 'guarded_macro': macro, 'guarded_groups': groups,
        'relative_macro_rmse_gain': 1 - macro['macro_relative_rmse'] / baseline['macro_relative_rmse'],
        'selected_frame_bytes': sum(r['searched_frame_bytes'] if d['searched'] else r['baseline_frame_bytes']
                                    for r, d in zip(rows, decisions)),
        'baseline_frame_bytes': sum(r['baseline_frame_bytes'] for r in rows),
        'decisions': decisions,
        'decoding': 'reuse existing paired frame files and unchanged shared dictionaries; original is encoder-only for selection'}
    write_json(directory / 'guarded_results.json', result)
    print(json.dumps({k: result[k] for k in ('windows', 'searched_windows', 'relative_macro_rmse_gain',
                                          'selected_frame_bytes', 'baseline_frame_bytes')}))


if __name__ == '__main__':
    main()
