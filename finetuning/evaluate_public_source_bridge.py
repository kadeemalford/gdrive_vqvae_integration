#!/usr/bin/env python3
"""Evaluate public-source bridges, preserving the exact audited control stream results."""
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.evaluate_lr_schedule_study import group_regressions
from finetuning.evaluate_pretraining_bridge import evaluate_candidate
from finetuning.paired_search_diagnostics import finite_json
from finetuning.representation_study import digest, read_manifest, write_json


def read_groups(path):
    with path.open() as stream:
        return [{key: value if key in ('family', 'configuration') else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one remote RTX4090 required')
    if not (args.study_dir / 'TRAINING_COMPLETE').exists():
        raise ValueError('Complete the public source training before validation')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    training = json.loads((args.study_dir / 'protocol.json').read_text())
    audit = training['control_audit']
    control = Path(audit['control_study']); old_wire = control / 'wire_validation'
    if digest(old_wire / 'protocol.json') != audit['wire_protocol_sha256']:
        raise ValueError('Control wire protocol changed')
    old_protocol = json.loads((old_wire / 'protocol.json').read_text())
    for name, expected in old_protocol['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Control replay implementation changed')
    samples = Path(training['samples_dir'])
    if digest(samples / 'samples_manifest.csv') != training['manifest_sha256']:
        raise ValueError('Target corpus changed')
    splits = read_manifest(samples)
    calibration, rng = [], random.Random(2095)
    for family in ('doped', 'multicore', 'multimode', 'single_mode'):
        rows = sorted((r for r in splits['train'] if r['family'] == family), key=lambda r: r['sample_path'])
        calibration += rng.sample(rows, 8)
    validation = sorted(splits['val'], key=lambda r: r['sample_path'])
    if training['smoke_only']:
        calibration = [max(calibration, key=lambda r: int(r['channels']))]
        validation = [max(validation, key=lambda r: int(r['channels']))]
    if [r['sample_path'] for r in calibration] != old_protocol['calibration_train_only']:
        raise ValueError('Calibration is not identical to reused controls')
    inputs = {r['sample_path']: digest(ROOT / r['sample_path']) for r in calibration + validation}
    if inputs != old_protocol['input_sha256'] or len(validation) != old_protocol['validation_windows']:
        raise ValueError('Validation/calibration inputs differ from reused controls')
    candidates = {'porotomo_selected': args.study_dir / 'porotomo/selected.pt',
                  'porotomo_endpoint': args.study_dir / 'porotomo/last.pt'}
    protocol = {'kind': 'public_source_wire_evaluation_v1',
                'training_protocol_sha256': digest(args.study_dir / 'protocol.json'),
                'control_audit': audit, 'candidates': {k: digest(v) for k, v in candidates.items()},
                'input_sha256': inputs, 'calibration_train_only': old_protocol['calibration_train_only'],
                'metadata_sha256': old_protocol['metadata_sha256'],
                'code_sha256': {name: digest(ROOT / name) for name in
                    ('finetuning/evaluate_public_source_bridge.py', 'finetuning/evaluate_pretraining_bridge.py',
                     'finetuning/evaluate_lr_schedule_study.py', 'compression/alphabet_stream.py', 'compression/study_bundle.py')},
                'control_retrained': False, 'control_reevaluated': False,
                'control_reuse': 'Exact model/stream/groups fingerprints and identical calibration/validation/replay implementation',
                'smoke_only': training['smoke_only'], 'final_holdout_accessed': False}
    for name, expected in protocol['metadata_sha256'].items():
        if digest(samples / name) != expected:
            raise ValueError('Target geometry/metadata changed')
    out = args.output_dir or args.study_dir / 'wire_validation'
    if out.exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Public source evaluation resume changed')
    else:
        out.mkdir(); write_json(out / 'protocol.json', protocol)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
    results, groups = {}, {}
    for label, expected in audit['wire_artifacts'].items():
        directory = old_wire / label; result_path = old_wire / f'{label}.json'
        if (digest(result_path) != expected['result_sha256']
                or digest(directory / 'groups.csv') != expected['groups_sha256']
                or digest(directory / 'validation.fsb') != expected['bundle_sha256']
                or digest(directory / 'decoder_model_weights.pt') != expected['deployment_model_sha256']):
            raise ValueError('Audited control artifacts changed')
        results[label] = json.loads(result_path.read_text())
        groups[label] = read_groups(directory / 'groups.csv')
    for label, path in candidates.items():
        result_path = out / f'{label}.json'
        if result_path.exists():
            results[label] = json.loads(result_path.read_text()); groups[label] = read_groups(out / label / 'groups.csv')
        else:
            results[label], groups[label] = evaluate_candidate(path, out / label, calibration, validation, samples, inputs)
            write_json(result_path, finite_json(results[label]))
        print(f"public-wire {label}: RMSE={results[label]['macro']['macro_relative_rmse']:.6f} bytes={results[label]['complete_stream_bytes']}", flush=True)
    comparisons = {}
    for phase in ('selected', 'endpoint'):
        new, baseline = results[f'porotomo_{phase}'], results[f'febus_{phase}']
        gain = 100 * (1 - new['macro']['macro_relative_rmse'] / baseline['macro']['macro_relative_rmse'])
        regressions = group_regressions(groups[f'febus_{phase}'], groups[f'porotomo_{phase}'])
        comparisons[phase] = {'rmse_gain_pct': gain,
            'stream_byte_change': new['complete_stream_bytes'] - baseline['complete_stream_bytes'],
            'configuration_regressions': regressions,
            'passes_investment_gate': gain >= 5 and new['complete_stream_bytes'] <= baseline['complete_stream_bytes'] and not regressions}
    write_json(out / 'results.json', finite_json({'candidates': results, 'comparisons': comparisons,
        'confirmation_eligible': not training['smoke_only'] and comparisons['selected']['passes_investment_gate'],
        'model_promoted': False, 'smoke_only': training['smoke_only'], 'control_retrained': False}))
    (out / 'COMPLETE').touch()
    write_json(args.study_dir / 'STATUS.json', {'stage': 'COMPLETE', 'model_promoted': False,
        'confirmation_eligible': not training['smoke_only'] and comparisons['selected']['passes_investment_gate'], 'smoke_only': training['smoke_only']})
    (args.study_dir / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
