#!/usr/bin/env python3
"""Development confirmation after both gated 65k product/flat training arms finish."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.codec_wire_evaluation import evaluate_codec
from finetuning.factorized_vq import VARIANTS, build_factorized
from finetuning.evaluate_lr_schedule_study import group_regressions
from finetuning.paired_search_diagnostics import finite_json
from finetuning.representation_study import digest, read_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one remote RTX4090 required')
    if not (args.study_dir / 'LONG_TRAINING_COMPLETE').exists():
        raise ValueError('Complete both scientifically gated long training arms first')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    lock = (args.study_dir.parent / '.lr_schedule_gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    training = json.loads((args.study_dir / 'protocol.json').read_text())
    if training['smoke_only']:
        raise ValueError('Smoke study is not a long training experiment')
    for name, expected in training['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen long-study implementation changed')
    for name, expected in training['vendor_sha256'].items():
        if digest(args.repo_root / 'vqvae' / name) != expected:
            raise ValueError('Vendor implementation changed')
    samples = Path(training['samples_dir'])
    if digest(samples / 'samples_manifest.csv') != training['manifest_sha256']:
        raise ValueError('Target corpus changed')
    rows = read_manifest(samples)
    lookup = {row['sample_path']: row for row in rows['train']}
    calibration = [lookup[name] for name in training['calibration_train_only']]
    validation = sorted(rows['val'], key=lambda row: row['sample_path'])
    inputs = {row['sample_path']: digest(ROOT / row['sample_path']) for row in calibration + validation}
    for row in calibration:
        if inputs[row['sample_path']] != training['input_sha256'][row['sample_path']]:
            raise ValueError('Train calibration changed')
    candidates = {f'{variant}_{phase}': args.study_dir / variant / name for variant in VARIANTS
                  for phase, name in [('selected', 'selected.pt'), ('endpoint', 'last.pt')]}
    protocol = {'kind': 'factorized_long_development_wire_v1',
        'training_protocol_sha256': digest(args.study_dir / 'protocol.json'),
        'candidates': {label: digest(path) for label, path in candidates.items()}, 'input_sha256': inputs,
        'metadata_sha256': {name: digest(samples / name) for name in ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')},
        'code_sha256': training['code_sha256'], 'validation_windows': len(validation), 'final_holdout_accessed': False}
    out = args.output_dir or args.study_dir / 'wire_validation'
    if out.exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Long wire resume protocol changed')
    else:
        out.mkdir(); write_json(out / 'protocol.json', protocol)
    results, groups = {}, {}
    for label, path in candidates.items():
        variant = label.split('_')[0]
        result_path = out / f'{label}.json'
        if result_path.exists():
            results[label] = json.loads(result_path.read_text())
            with (out / label / 'groups.csv').open() as stream:
                groups[label] = [{k: v if k in ('family', 'configuration') else float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
        else:
            results[label], groups[label] = evaluate_codec(path, out / label, calibration, validation, samples, inputs,
                build_factorized, {'version': 'factorized-vq-study-v1', 'variant': variant, 'composite_states': 2048,
                'radices': [2048] if variant == 'flat' else [32, 64], 'nominal_bits': 11})
            write_json(result_path, finite_json(results[label]))
        print(f"factorized-wire {label} RMSE={results[label]['macro']['macro_relative_rmse']:.6f} bytes={results[label]['complete_stream_bytes']}", flush=True)
    comparisons = {}
    for phase in ('selected', 'endpoint'):
        before, after = results[f'flat_{phase}'], results[f'product_{phase}']
        gain = 100 * (1 - after['macro']['macro_relative_rmse'] / before['macro']['macro_relative_rmse'])
        delta = after['complete_stream_bytes'] - before['complete_stream_bytes']
        failures = group_regressions(groups[f'flat_{phase}'], groups[f'product_{phase}'])
        comparisons[phase] = {'rmse_gain_pct': gain, 'stream_byte_change': delta, 'configuration_regressions': failures,
            'passes_investment_gate': gain >= 5 and delta <= 0 and not failures}
    write_json(out / 'results.json', finite_json({'candidates': results, 'comparisons': comparisons,
        'confirmation_eligible': comparisons['selected']['passes_investment_gate'], 'model_promoted': False}))
    (out / 'COMPLETE').touch(); (args.study_dir / 'COMPLETE').touch()
    lock.close()


if __name__ == '__main__':
    main()
