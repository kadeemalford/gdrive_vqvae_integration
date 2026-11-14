#!/usr/bin/env python3
"""Separate practical equal-compute utility from the same-target-dose transfer contrast."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.evaluate_pretraining_bridge import evaluate_candidate
from finetuning.paired_search_diagnostics import finite_json
from finetuning.representation_study import digest, read_manifest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose one remote RTX4090')
    if not (args.study_dir / 'TRAINING_COMPLETE').exists():
        raise ValueError('Complete matched-target-dose training first')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    training = json.loads((args.study_dir / 'protocol.json').read_text())
    control = Path(training['control_audit']['control_study'])
    old = control / 'wire_validation'
    if digest(old / 'results.json') != training['control_audit']['wire_results_sha256']:
        raise ValueError('Original bridge results changed')
    old_protocol = json.loads((old / 'protocol.json').read_text())
    samples = Path(training['samples_dir']); splits = read_manifest(samples)
    calibration, rng = [], random.Random(2095)
    for family in ('doped','multicore','multimode','single_mode'):
        rows = sorted((r for r in splits['train'] if r['family'] == family),key=lambda r:r['sample_path'])
        calibration += rng.sample(rows,8)
    validation = sorted(splits['val'],key=lambda r:r['sample_path'])
    inputs = {r['sample_path']:digest(ROOT/r['sample_path']) for r in calibration+validation}
    if inputs != old_protocol['input_sha256'] or [r['sample_path'] for r in calibration] != old_protocol['calibration_train_only']:
        raise ValueError('Diagnostic target/calibration inputs differ from bridge study')
    for name, expected in old_protocol['code_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Shared wire evaluation implementation changed')
    for name, expected in old_protocol['metadata_sha256'].items():
        if digest(samples/name) != expected:
            raise ValueError('Target geometry/manifest metadata changed')
    candidates = {'selected':args.study_dir/'selected.pt','endpoint':args.study_dir/'last.pt'}
    protocol = {'kind':'matched_target_dose_wire_v1','training_protocol_sha256':digest(args.study_dir/'protocol.json'),
        'bridge_results_sha256':digest(old/'results.json'),'input_sha256':inputs,
        'candidates':{k:digest(v) for k,v in candidates.items()},
        'source_sha256':digest(Path(__file__)),
        'role':'Diagnostic only; keep primary equal-total-update comparator and investment gate unchanged',
        'final_holdout_accessed':False}
    out=args.study_dir/'wire_validation'
    if out.exists():
        if json.loads((out/'protocol.json').read_text())!=protocol:
            raise ValueError('Diagnostic wire resume protocol changed')
    else:
        out.mkdir();write_json(out/'protocol.json',protocol)
        shutil.copy2(Path(__file__),out/Path(__file__).name)
    results={}
    bridge=json.loads((old/'results.json').read_text())
    comparisons={}
    for label,path in candidates.items():
        result_path=out/f'{label}.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
        else:
            result,_=evaluate_candidate(path,out/label,calibration,validation,samples,inputs)
            write_json(result_path,finite_json(result))
        results[label]=result
        q=bridge['candidates'][f'quantx_{label}']
        comparisons[label]={'quantx_gain_vs_same_febus_dose_pct':100*(1-q['macro']['macro_relative_rmse']/result['macro']['macro_relative_rmse']),
            'quantx_stream_byte_change':q['complete_stream_bytes']-result['complete_stream_bytes'],
            'diagnostic_control_total_updates':training['updates'],
            'quantx_total_new_updates':24576,'same_final_febus_updates':training['updates']}
        print(f"target-dose-wire {label}: RMSE={result['macro']['macro_relative_rmse']:.6f} bytes={result['complete_stream_bytes']}",flush=True)
    write_json(out/'results.json',finite_json({'candidates':results,'comparisons':comparisons,
        'primary_equal_total_update_gate_unchanged':True,'model_promoted':False}))
    (out/'COMPLETE').touch();(args.study_dir/'COMPLETE').touch()
    write_json(args.study_dir/'STATUS.json',{'stage':'COMPLETE','diagnostic_only':True,'model_promoted':False})


if __name__=='__main__':
    main()
