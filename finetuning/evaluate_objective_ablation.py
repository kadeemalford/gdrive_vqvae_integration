#!/usr/bin/env python3
"""Full wire validation for objective-only continuation with fingerprinted control reuse."""
import argparse
import csv
import fcntl
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090')
    if not (args.study_dir / 'TRAINING_COMPLETE').exists():
        raise ValueError('Complete objective continuation before evaluating')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    lock = (args.study_dir.parent / '.lr_schedule_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    training = json.loads((args.study_dir / 'protocol.json').read_text())
    if digest(args.source) != training['source_sha256']:
        raise ValueError('Source checkpoint changed')
    for name, expected in training['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen objective/replay implementation changed: {name}')
    for name, expected in training['vendor_sha256'].items():
        if digest(args.repo_root / 'vqvae' / name) != expected:
            raise ValueError('Vendor implementation changed')
    audit = training['control_audit']
    old_wire = Path(audit['study']) / 'wire_validation'
    if digest(old_wire / 'protocol.json') != audit['wire_protocol_sha256']:
        raise ValueError('Control wire protocol changed')
    old_protocol = json.loads((old_wire / 'protocol.json').read_text())
    for entries in audit['wire_artifacts'].values():
        for name, expected in entries.items():
            if digest(old_wire / name) != expected:
                raise ValueError(f'Control wire artifact changed: {name}')
    samples = Path(training['samples_dir'])
    if digest(samples / 'samples_manifest.csv') != training['manifest_sha256']:
        raise ValueError('Corpus manifest changed')
    splits = read_manifest(samples)
    rng, calibration = random.Random(2095), []
    for family in ('doped', 'multicore', 'multimode', 'single_mode'):
        available = sorted((r for r in splits['train'] if r['family'] == family), key=lambda row: row['sample_path'])
        calibration += rng.sample(available, 8)
    validation = sorted(splits['val'], key=lambda row: row['sample_path'])
    inputs = {r['sample_path']: digest(ROOT / r['sample_path']) for r in calibration + validation}
    if (inputs != old_protocol['input_sha256'] or len(validation) != old_protocol['validation_windows']
            or [r['sample_path'] for r in calibration] != old_protocol['calibration_train_only']):
        raise ValueError('Historical and challenger calibration/validation samples differ')
    for name, expected in old_protocol['metadata_sha256'].items():
        if digest(samples / name) != expected:
            raise ValueError('Target geometry/side metadata changed')
    if training['smoke_only']:
        calibration = [max(calibration, key=lambda row: int(row['channels']))]
        validation = [max(validation, key=lambda row: int(row['channels']))]
    candidates = {'mse_selected': args.study_dir / 'mse/selected.pt',
                  'mse_endpoint': args.study_dir / 'mse/last.pt'}
    if training['smoke_only']:
        candidates['source'] = args.source
    protocol = {'kind': 'objective_ablation_wire_v1', 'training_protocol_sha256': digest(args.study_dir / 'protocol.json'),
        'candidates': {label: digest(path) for label, path in candidates.items()},
        'input_sha256': inputs, 'metadata_sha256': old_protocol['metadata_sha256'],
        'validation_windows': len(validation), 'calibration_train_only': [r['sample_path'] for r in calibration],
        'control_audit': audit, 'smoke_only': training['smoke_only'],
        'wire': 'FAK1/FSB1, discrete neural replay, all metadata/maps/tables charged; shared weights separate',
        'code_sha256': training['code_sha256'], 'final_holdout_accessed': False}
    out = args.output_dir or args.study_dir / 'wire_validation'
    if out.exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Objective wire evaluation resume changed')
    else:
        out.mkdir()
        write_json(out / 'protocol.json', protocol)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
    results, groups = {}, {}
    if not training['smoke_only']:
        for label in ('source', 'constant_high_selected', 'constant_high_endpoint'):
            results[label] = json.loads((old_wire / f'{label}.json').read_text())
            groups[label] = read_groups(old_wire / label / 'groups.csv')
    for label, path in candidates.items():
        result_path = out / f'{label}.json'
        if result_path.exists():
            results[label] = json.loads(result_path.read_text())
            groups[label] = read_groups(out / label / 'groups.csv')
        else:
            results[label], groups[label] = evaluate_candidate(path, out / label, calibration, validation, samples, inputs)
            write_json(result_path, finite_json(results[label]))
        print(f"objective-wire {label} RMSE={results[label]['macro']['macro_relative_rmse']:.6f} bytes={results[label]['complete_stream_bytes']}", flush=True)
    comparisons = {}
    for phase in ('selected', 'endpoint'):
        label = f'mse_{phase}'
        reference = 'source' if training['smoke_only'] else f'constant_high_{phase}'
        before, after = results[reference], results[label]
        gain = 100 * (1 - after['macro']['macro_relative_rmse'] / before['macro']['macro_relative_rmse'])
        failures = group_regressions(groups[reference], groups[label])
        byte_change = after['complete_stream_bytes'] - before['complete_stream_bytes']
        comparisons[phase] = {'reference': reference, 'rmse_gain_pct': gain,
            'stream_byte_change': byte_change, 'configuration_regressions': failures,
            'passes_investment_gate': not training['smoke_only'] and gain >= 5 and byte_change <= 0 and not failures}
    eligible = comparisons['selected']['passes_investment_gate']
    write_json(out / 'results.json', finite_json({'candidates': results, 'comparisons': comparisons,
        'confirmation_eligible': eligible, 'model_promoted': False, 'smoke_only': training['smoke_only'],
        'control_retrained': False, 'control_reevaluated': training['smoke_only']}))
    (out / 'COMPLETE').touch()
    write_json(args.study_dir / 'STATUS.json', {'stage': 'COMPLETE', 'confirmation_eligible': eligible,
        'model_promoted': False, 'smoke_only': training['smoke_only']})
    (args.study_dir / 'COMPLETE').touch()
    lock.close()


if __name__ == '__main__':
    main()
