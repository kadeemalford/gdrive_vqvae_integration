#!/usr/bin/env python3
"""Compare scalar points with deployable VQ tile-mode choices at measured cohort budgets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import digest, write_json
from finetuning.summarize_index_search import admit_search


def allocate_modes(compact, dense, budget, auxiliary, guarded=False):
    if len(compact) != len(dense) or [r['sample_path'] for r in compact] != [r['sample_path'] for r in dense]:
        raise ValueError('Paired identical sample order required')
    used = auxiliary + sum(r['frame_bytes'] for r in compact)
    if used > budget:
        return None
    group_cells = Counter()
    family_configs = {}
    for row in compact:
        m = row['metrics']
        group_cells[m['family'], m['configuration']] += m['_aggregation_weight']
        family_configs.setdefault(m['family'], set()).add(m['configuration'])
    priorities = []
    for i, (a, b) in enumerate(zip(compact, dense)):
        x, y = a['metrics'], b['metrics']
        if guarded and not admit_search(x, y):
            continue
        weight = x['_aggregation_weight'] / group_cells[x['family'], x['configuration']]
        weight /= len(family_configs[x['family']]) * len(family_configs)
        gain = weight * (x['relative_rmse'] - y['relative_rmse'])
        if gain <= 0:
            continue
        delta = b['frame_bytes'] - a['frame_bytes']
        priorities.append((-gain / max(1, delta), a['sample_path'], i, delta))
    modes = [0] * len(compact)
    for _, _, i, delta in sorted(priorities):
        if used + delta <= budget:
            modes[i] = 1
            used += delta
    rows = [(dense[i] if mode else compact[i])['metrics'] for i, mode in enumerate(modes)]
    macro, groups = aggregate_validation(rows)
    return {'bytes': used, 'budget': budget, 'dense_windows': sum(modes), 'macro': macro, 'groups': groups,
            'modes': modes, 'strict_per_window_tail_guard': guarded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compact-dir', type=Path, required=True)
    parser.add_argument('--dense-dir', type=Path, required=True)
    parser.add_argument('--candidate', action='append', required=True, help='label=scalar experiment directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads((args.compact_dir / 'records.json').read_text())
    dense = json.loads((args.dense_dir / 'records.json').read_text())
    compact_result = json.loads((args.compact_dir / 'results.json').read_text())
    mixed_auxiliary = compact_result['shared_tables_maps_transport_bytes'] + sum(
        (args.dense_dir / name).stat().st_size for name in
        ('huffman_table.json', 'range_descriptor.json', 'range_probabilities.f32'))
    results = {}
    for specification in args.candidate:
        label, location = specification.split('=', 1)
        candidate = Path(location)
        if not (candidate / 'COMPLETE').exists():
            raise ValueError('Completed scalar candidate required')
        result = json.loads((candidate / 'results.json').read_text())
        candidate_records = json.loads((candidate / 'full_validation/records.json').read_text())
        if [r['sample_path'] for r in candidate_records] != [r['sample_path'] for r in compact]:
            raise ValueError('Scalar and VQ validation inputs differ')
        budget = result['total_stream_bytes']
        results[label] = {'scalar_bytes': budget, 'scalar_bpc': result['total_bpc'], 'scalar_macro': result['macro'],
                         'scalar_groups': result['groups'], 'checkpoint_sha256': result['best_checkpoint_sha256'],
                         'vq_rmse_allocation': allocate_modes(compact, dense, budget, mixed_auxiliary),
                         'vq_guarded_allocation': allocate_modes(compact, dense, budget, mixed_auxiliary, guarded=True)}
        if compact_result['total_stream_bytes'] <= budget:
            results[label]['compact_reference_alone'] = compact_result
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'results.json', {'comparisons': results,
        'source_sha256': digest(Path(__file__)), 'mixed_shared_auxiliary_bytes': mixed_auxiliary,
        'reference_results_sha256': {'compact': digest(args.compact_dir / 'results.json'),
                                    'dense': digest(args.dense_dir / 'results.json')},
        'contract': 'whole-validation-cohort budget; full maps/transport and both model tables; shared waveform weights excluded',
        'selection': 'fixed greedy macro-RMSE gain per extra framed byte; encoder has original; decoder reads selected frame identifier',
        'limitations': 'development comparison, one fold/seed; not a new acquisition, per-file cap, globally optimal allocation or external validation'})
    (args.output_dir / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
