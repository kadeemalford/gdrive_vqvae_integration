#!/usr/bin/env python3
"""Summarize paired search artifacts without fitting or selecting a new model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import digest, write_json
from finetuning.summarize_index_search import admit_search
from finetuning.paired_search_diagnostics import aggregate_metrics, guarded_admit, finite_json


def summarize(directory):
    if not (directory / 'COMPLETE').exists():
        raise ValueError('Completed paired evaluation required')
    protocol = json.loads((directory / 'protocol.json').read_text())
    result = {'protocol_sha256': digest(directory / 'protocol.json'),
              'summary_source_sha256': digest(Path(__file__)), 'splits': {}}
    for split, selected in protocol['selection'].items():
        rows = [json.loads((directory / split / f'{i:05d}.json').read_text()) for i in range(len(selected))]
        assert [r['sample_path'] for r in rows] == selected
        violations = 0
        for row in rows:
            for arm in ('control_guarded', 'sharp_guarded', 'search'):
                unchanged = row['metrics'][arm] == row['metrics']['source'] and row['frame_bytes'][arm] == row['frame_bytes']['source']
                violations += int(row['frame_bytes'][arm] > row['frame_bytes']['source'] or
                                  not (unchanged or guarded_admit(row['metrics']['source'], row['metrics'][arm])))
        assert violations == 0
        files = defaultdict(list)
        for row in rows:
            files[row['family'], row['configuration'], row['drive_id']].append(row)
        file_results = []
        for key, values in sorted(files.items()):
            rmse = {arm: aggregate_metrics([r['metrics'][arm] for r in values])[0]['macro_relative_rmse']
                    for arm in values[0]['metrics']}
            file_results.append({'family': key[0], 'configuration': key[1], 'drive_id': key[2],
                'windows': len(values), 'rmse': rmse,
                'gain_search_vs_sharp_guarded': 1 - rmse['search'] / rmse['sharp_guarded'],
                'limitation': 'sampled windows only, not pooled continuous file metrics'})
        summary = {'windows': len(rows), 'files': len(files), 'guard_violations': violations,
            'file_comparison': file_results,
            'timing_seconds': {arm: {'median': float(np.median([r['seconds'][arm] for r in rows])),
                'p90': float(np.quantile([r['seconds'][arm] for r in rows], .9)),
                'total': sum(r['seconds'][arm] for r in rows)} for arm in rows[0]['seconds']},
            'source_evaluation_required_for_guarded_encoders': True,
            'physical_regression_windows_vs_source': {},
            'changed_search_windows': sum(r['search_details']['changed_fraction'] > 0 for r in rows)}
        for arm in ('control_guarded', 'sharp_guarded', 'search'):
            summary['physical_regression_windows_vs_source'][arm] = {
                'fft_magnitude': sum(r['metrics'][arm]['fft_magnitude_relative_error'] > r['metrics']['source']['fft_magnitude_relative_error'] for r in rows if r['metrics'][arm]['fft_magnitude_relative_error'] is not None and r['metrics']['source']['fft_magnitude_relative_error'] is not None),
                'phase': sum(r['metrics'][arm]['fft_phase_alignment'] < r['metrics']['source']['fft_phase_alignment'] for r in rows if r['metrics'][arm]['fft_phase_alignment'] is not None and r['metrics']['source']['fft_phase_alignment'] is not None),
                'rms_distance_to_one': sum(abs(r['metrics'][arm]['reconstruction_rms_ratio'] - 1) > abs(r['metrics']['source']['reconstruction_rms_ratio'] - 1) for r in rows if r['metrics'][arm]['reconstruction_rms_ratio'] is not None and r['metrics']['source']['reconstruction_rms_ratio'] is not None)}
        if split == 'train':
            grouped = defaultdict(list)
            for row in rows:
                grouped[row['family'], row['configuration']].append(row)
            residual = []
            for key, values in sorted(grouped.items()):
                item = {'family': key[0], 'configuration': key[1], 'windows': len(values)}
                for arm in ('source', 'search'):
                    probes = [r['residual'][arm]['spatial_probe'] for r in values]
                    gains = [p['relative_rmse_gain'] for p in probes if p.get('relative_rmse_gain') is not None]
                    band_ratios = []
                    for band in range(3):
                        records = [r['residual'][arm]['bands'][band] for r in values if 'bands' in r['residual'][arm]]
                        ratios = [b['error_target_power_ratio'] for b in records if b['error_target_power_ratio'] is not None]
                        band_ratios.append(float(np.median(ratios)) if ratios else None)
                    edges = [r['residual'][arm] for r in values if r['residual'][arm]['interior_mse'] is not None and r['residual'][arm]['interior_mse'] > 1e-20]
                    item[arm] = {'spatial_probe_windows': len(gains),
                        'median_spatial_probe_rmse_gain': float(np.median(gains)) if gains else None,
                        'median_band_error_target_power_ratio': band_ratios,
                        'median_edge_interior_mse_ratio': float(np.median([r['edge_mse'] / r['interior_mse'] for r in edges])) if edges else None}
                residual.append(item)
            summary['residual_by_configuration'] = residual
            summary['residual_limitations'] = 'Unweighted descriptive medians within configuration; original neighboring residuals; within-tile adjacency, no physical spacing inference; no codec gain claim.'
        result['splits'][split] = summary
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    write_json(args.directory / 'diagnostics_summary.json', finite_json(result))
    print(json.dumps({s: {k: v for k, v in value.items() if k not in ('file_comparison', 'residual_by_configuration')}
                      for s, value in result['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
