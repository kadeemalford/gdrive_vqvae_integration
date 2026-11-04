#!/usr/bin/env python3
"""Audit paired encoder experiments without changing training or promotion guards."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.representation_study import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', action='append', required=True, help='label=completed experiment directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    entries = {}
    source_hash = None
    reference_paths = None
    for specification in args.experiment:
        label, name = specification.split('=', 1)
        directory = Path(name)
        if not (directory / 'COMPLETE').exists():
            raise ValueError('Completed encoder experiment required')
        result = json.loads((directory / 'results.json').read_text())
        rows = json.loads((directory / 'full_validation/records.json').read_text())
        paths = [r['sample_path'] for r in rows]
        if source_hash is not None and (result['source_checkpoint_sha256'] != source_hash or paths != reference_paths):
            raise ValueError('Unpaired source checkpoint or validation inputs')
        source_hash = result['source_checkpoint_sha256']; reference_paths = paths
        violations = {name: 0 for name in ('bytes', 'rmse', 'correlation', 'p90')}
        for row in rows:
            a, b = row['baseline'], row['guarded']
            violations['bytes'] += row['frame_bytes'] > row['baseline_frame_bytes']
            violations['rmse'] += b['relative_rmse'] > a['relative_rmse']
            violations['correlation'] += b['correlation'] < a['correlation']
            violations['p90'] += b['ordinary_channel_p90_relative_rmse'] > a['ordinary_channel_p90_relative_rmse']
        if any(violations.values()):
            raise ValueError('A declared per-window guard was violated')
        baseline = result['baseline']['macro']['macro_relative_rmse']
        entries[label] = {'selected_step': result['selected_step'], 'completed_steps': result['completed_steps'],
            'rate_coefficient': result['rate_coefficient'], 'sampling': result['sampling'],
            'guarded_rmse_gain_vs_source': 1 - result['guarded']['macro']['macro_relative_rmse'] / baseline,
            'guard_violations': violations, 'exposure': result['exposure'],
            'baseline': result['baseline'], 'candidate': result['candidate'], 'guarded': result['guarded'],
            'checkpoint_sha256': result['checkpoint_sha256'], 'accepted_changed_windows': result['accepted_changed_windows']}
    pairs = {}
    labels = sorted(entries)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            x, y = entries[a], entries[b]
            pairs[f'{a}/{b}'] = {'same_update_budget': x['completed_steps'] == y['completed_steps'],
                'same_configuration_dose': x['exposure']['configuration_updates'] == y['exposure']['configuration_updates'],
                'same_cell_exposure': x['exposure']['cells'] == y['exposure']['cells'],
                'guarded_rmse_ratio_a_over_b': x['guarded']['macro']['macro_relative_rmse'] / y['guarded']['macro']['macro_relative_rmse']}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'results.json', {'experiments': entries, 'pairs': pairs,
        'source_checkpoint_sha256': source_hash, 'source_sha256': digest(Path(__file__)),
        'note': 'strict guards are construction constraints; only gain magnitude/replication is performance evidence; no external validation'})
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps({name: {'rmse_gain': row['guarded_rmse_gain_vs_source'], 'steps': row['completed_steps']}
                      for name, row in entries.items()}))


if __name__ == '__main__':
    main()
