#!/usr/bin/env python3
"""CPU DCT controls at actual per-frame alphabet-pilot budgets; no tuning or training."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.affine_codec_study import score_restored
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, read_manifest, load_packet, write_json
from finetuning.run_physical_guard_search import physical_admit
from finetuning.sparse_dct_codec import compress_packet, decompress


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.experiment
    assert (root / 'COMPLETE').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    assert digest(args.samples_dir / 'samples_manifest.csv') == protocol['manifest_sha256']
    rows = {r['sample_path']: r for r in read_manifest(args.samples_dir)['val']}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/compare_alphabet_dct.py', 'finetuning/sparse_dct_codec.py',
            'finetuning/affine_codec_study.py', 'finetuning/run_physical_guard_search.py']
    write_json(out / 'protocol.json', {'kind': 'alphabet_actual_frame_matched_dct_v1',
        'training_updates': 0, 'gpu_used': False, 'quant_bits': 8,
        'code_sha256': {f: digest(ROOT / f) for f in code}, 'reference_results_sha256': digest(root / 'results.json'),
        'validation': protocol['validation'],
        'selection': 'same consumed 63 development windows; no budget, coefficient precision or sample sweep',
        'target': 'exact same robust packet/ordinary mask/side as neural, including exact finite escapes',
        'rate': 'each DCT frame <= corresponding neural candidate frame including side/header/coefficients; no free indices or scale; no shared waveform weights/tables for DCT',
        'limitations': ['not independent confirmation', 'classical codec is an existing baseline, not optimized or a lower bound',
                        'different existing frame headers, no transport duplication; neural table/model costs reported separately',
                        'no neural training extension or change of selected checkpoint']})
    for f in code:
        dest = out / 'code_snapshot' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    result = {'training_updates': 0, 'gpu_used': False, 'arms': {}}
    for arm in protocol['arms']:
        records = json.loads((root / arm / 'records.json').read_text())
        folder = out / arm
        folder.mkdir()
        measured, output = [], []
        total = 0
        for i, record in enumerate(records):
            row = rows[record['sample_path']]
            raw, packet = load_packet(row)
            frame, stats = compress_packet(packet, record['frame_bytes'], quant_bits=8)
            restored = decompress(frame)
            np.testing.assert_array_equal(raw.view(np.uint32)[~packet.valid], restored.view(np.uint32)[~packet.valid])
            metric = finite_json(score_restored(row, raw, packet, restored))
            assert len(frame) <= record['frame_bytes']
            (folder / f'{i:05d}.fsd').write_bytes(frame)
            total += len(frame)
            measured.append(metric)
            output.append({'sample_path': row['sample_path'], 'neural': record['metrics'], 'dct': metric,
                'neural_frame_bytes': record['frame_bytes'], 'dct_frame_bytes': len(frame),
                'dct_physically_dominates_neural': bool(physical_admit(record['metrics'], metric)),
                'neural_physically_dominates_dct': bool(physical_admit(metric, record['metrics'])),
                'retained_coefficients': stats['retained_coefficients']})
        macro, groups = aggregate_metrics(measured)
        write_json(folder / 'records.json', output)
        result['arms'][arm] = {'dct_macro': macro, 'dct_groups': groups, 'dct_frame_bytes': total,
            'neural_frame_bytes': sum(r['frame_bytes'] for r in records),
            'dct_physically_dominating_windows': sum(r['dct_physically_dominates_neural'] for r in output),
            'neural_physically_dominating_windows': sum(r['neural_physically_dominates_dct'] for r in output)}
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({a: {'dct_rmse': r['dct_macro']['macro_relative_rmse'],
        'dct_bytes': r['dct_frame_bytes'], 'neural_bytes': r['neural_frame_bytes'],
        'dct_dominates': r['dct_physically_dominating_windows'], 'neural_dominates': r['neural_physically_dominating_windows']}
        for a, r in result['arms'].items()}), flush=True)


if __name__ == '__main__':
    main()
