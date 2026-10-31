#!/usr/bin/env python3
"""Predeclared retrospective ROI score ablation plus exact transport accounting.

Uses immutable cached reconstructions; no new model inference or parameter updates.
This is a new development experiment, not independent confirmation.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.study_bundle import write_bundle, read_bundle
from compression.symbol_stream import BitsOut, HEADER, SymbolCodec
from finetuning.reconstruction_objective import aggregate_validation


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def balanced_modes(records, factor):
    base = sum(r['compact']['frame_bytes'] for r in records)
    ceiling = math.floor(base * factor)
    choices = {r['sample_id']: 'compact' for r in records}
    used = base
    ranked = []
    for row in records:
        a, b = row['compact']['metrics'], row['dense']['metrics']
        if (b['relative_rmse'] > a['relative_rmse'] or b['correlation'] < a['correlation']
                or b['ordinary_channel_p90_relative_rmse'] > a['ordinary_channel_p90_relative_rmse']):
            continue
        mse_gain = np.clip(1 - b['relative_rmse']**2 / max(a['relative_rmse']**2, 1e-12), 0, 1)
        tail_gain = np.clip(1 - b['ordinary_channel_p90_relative_rmse'] /
                            max(a['ordinary_channel_p90_relative_rmse'], 1e-6), 0, 1)
        gain = .5 * (mse_gain + tail_gain) * row['input_cells']
        if not np.isfinite(gain) or gain <= 0:
            continue
        extra = row['dense']['frame_bytes'] - row['compact']['frame_bytes']
        ranked.append((-gain / max(extra, 1), row['sample_id'], extra))
    for _, sample_id, extra in sorted(ranked):
        if used + extra <= ceiling:
            choices[sample_id] = 'dense';used += extra
    assert used <= ceiling
    return choices


def fixed_width_frame(codec, original):
    symbols, _, _, side = codec.decode(original)
    writer = BitsOut()
    for symbol in symbols:
        writer.write(int(symbol), 11)
    payload = writer.finish()
    fields = list(HEADER.unpack_from(original))
    fields[2] = 0
    fields[-1] = zlib.crc32(payload, zlib.crc32(side))
    result = HEADER.pack(*fields) + side + payload
    recovered, _, _, recovered_side = codec.decode(result)
    np.testing.assert_array_equal(symbols, recovered)
    assert recovered_side == side
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.resolve()
    if not (source / 'COMPLETE').exists():
        raise ValueError('Completed frozen inference study required')
    if shutil.disk_usage(source).free < 4 * 2**30:
        raise RuntimeError('Insufficient disk headroom')
    output.mkdir(exist_ok=False)
    source_results = json.loads((source / 'results.json').read_text())
    records_by_id = {r['sample_id']: r for r in map(lambda p: json.loads(p.read_text()),
                                                  sorted((source / 'window_cache').glob('*.json')))}
    with (source / 'validation_metadata.csv').open() as stream:
        metadata = list(csv.DictReader(stream))
    records = [records_by_id[Path(row['sample_path']).stem] for row in metadata]
    assert len(records) == 7220
    protocol = {'kind': 'retrospective_fractional_tail_roi_and_lossless_metadata_v1',
                'source_protocol_sha256': digest(source / 'protocol.json'),
                'source_results_sha256': digest(source / 'results.json'),
                'budgets': [1.5, 2.0], 'new_policy': 'balanced_fractional',
                'score': 'cells * 0.5 * (fractional squared-RMSE gain + fractional channel-p90 gain) / extra bytes',
                'score_bounds': 'each fractional gain clipped to [0,1]; epsilon MSE=1e-12, p90=1e-6',
                'admission': 'no per-window RMSE/correlation/channel-p90 regression against compact',
                'metadata': 'ZIP_DEFLATED level9 lossless; length-framed payloads, metadata-declared frame order',
                'no_new_inference': True, 'no_training': True, 'no_independent_validation': True,
                'no_test_access': True, 'final_model_promoted': False,
                'code_sha256': {str(p): digest(p) for p in (Path(__file__), ROOT / 'compression/study_bundle.py')}}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for filename in protocol['code_sha256']:
        path = Path(filename)
        target = output / 'code_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    assets = {name: (source / name).read_bytes() for name in
              ('validation_metadata.csv', 'channel_maps.jsonl', 'preprocessing_policy.json')}
    assets['sequence.json'] = json.dumps([r['sample_id'] for r in records], separators=(',', ':')).encode()
    by_file = defaultdict(list)
    for row in records:
        by_file[row['drive_id']].append(row)
    cases = {name: json.loads((source / f'{name}.modes.json').read_text()) for name in source_results}
    for mode in ('compact', 'dense'):
        codec = SymbolCodec.from_bytes((source / f'{mode}.tables.json').read_bytes())
        folder = output / 'fixed11_frames' / mode
        folder.mkdir(parents=True)
        for record in records:
            name = record['sample_id'] + '.fsc'
            (folder / name).write_bytes(fixed_width_frame(codec, (source / 'frames' / mode / name).read_bytes()))
        cases[f'fixed11_{mode}'] = {r['sample_id']: mode for r in records}
    for factor in protocol['budgets']:
        choices = {}
        for values in by_file.values():
            choices.update(balanced_modes(values, factor))
        cases[f'balanced_{str(factor).replace(".", "p")}'] = choices
    results, group_results = {}, {}
    for label, choices in cases.items():
        modes = sorted(set(choices.values()))
        shared = {**assets, **{f'{mode}.tables.json': (source / f'{mode}.tables.json').read_bytes() for mode in modes}}
        rows, paths = [], []
        for record in records:
            mode = choices[record['sample_id']]
            rows.append({'family': record['family'], 'configuration': record['configuration'],
                         '_aggregation_weight': float(record['input_cells']), **record[mode]['metrics']})
            base = output / 'fixed11_frames' if label.startswith('fixed11_') else source / 'frames'
            paths.append(base / mode / (record['sample_id'] + '.fsc'))
        bundle = output / f'{label}.fsb'
        accounting = write_bundle(bundle, shared, paths)
        reader = read_bundle(bundle)
        assert next(reader) == shared
        count = 0
        for actual, path in zip(reader, paths, strict=True):
            assert actual == path.read_bytes(), path
            count += 1
        assert count == len(records)
        macro, groups = aggregate_validation(rows)
        cells = sum(r['input_cells'] for r in records)
        frame_bytes = sum(path.stat().st_size for path in paths)
        original_frames = sum(r['compact']['frame_bytes'] for r in records)
        result = {**macro, **accounting, 'stream_bytes': bundle.stat().st_size,
                  'ratio_shared_models': 4*cells/bundle.stat().st_size,
                  'frame_bytes': frame_bytes, 'frame_multiplier_vs_compact': frame_bytes/original_frames,
                  'dense_cell_fraction': sum(r['input_cells'] for r in records if choices[r['sample_id']]=='dense')/cells,
                  'all_frames_and_metadata_roundtripped_byte_exact': True,
                  'bundle_sha256': digest(bundle)}
        deployment = sum(source_results[mode]['shared_float32_weight_bytes'] for mode in modes)
        result['ratio_with_weights_once'] = 4*cells/(bundle.stat().st_size+deployment)
        results[label] = result
        group_results[label] = groups
        (output / f'{label}.modes.json').write_text(json.dumps(choices, sort_keys=True)+'\n')
        with (output / f'{label}.groups.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
        print(label, 'RMSE', macro['macro_relative_rmse'], 'ratio', result['ratio_shared_models'], flush=True)
    (output / 'results.json').write_text(json.dumps(results, indent=2)+'\n')
    baseline = {(r['family'], r['configuration']): r for r in group_results['compact']}
    guards = {}
    for label, groups in group_results.items():
        regressions = []
        for row in groups:
            reference = baseline[row['family'], row['configuration']]
            for metric, direction in (('relative_rmse', 1), ('correlation', -1), ('ordinary_channel_p90_relative_rmse', 1)):
                delta = row[metric] - reference[metric]
                if direction * delta > 1e-8:
                    regressions.append({'family': row['family'], 'configuration': row['configuration'],
                                        'metric': metric, 'delta': delta})
        guards[label] = {'regressions_vs_compact': regressions, 'final_model_promoted': False}
    (output / 'guards.json').write_text(json.dumps(guards, indent=2)+'\n')
    report = ['# Lossless transport and fractional-tail ROI ablation', '',
              'Retrospective development, same frozen reconstructions; no training or final promotion.', '',
              '| Mode | RMSE | Correlation | Channel p90 | Actual bundle ratio | Dense cells |',
              '|---|---:|---:|---:|---:|---:|']
    for label, result in results.items():
        report.append(f'| {label} | {result["macro_relative_rmse"]:.6f} | {result["macro_correlation"]:.6f} | {result["macro_ordinary_channel_p90_relative_rmse"]:.6f} | {result["ratio_shared_models"]:.3f} | {result["dense_cell_fraction"]:.3f} |')
    report += ['', 'Actual bundle sizes include compressed physical/acquisition metadata and frame boundaries.',
               'All original metadata and selected frame bytes were recovered exactly.',
               'Float32 shared model weights are reported separately; compare cold-start and amortized use.',
               'Budget multipliers are ceilings on per-file framed bytes; report actual consumed bytes.',
               'Guarded metric monotonicity is an encoder decision rule, not external validation.']
    (output / 'REPORT.md').write_text('\n'.join(report)+'\n')
    (output / 'COMPLETE').write_text('All transport bundles roundtripped; retrospective ROI ablation, no promotion.\n')


if __name__ == '__main__':
    main()
