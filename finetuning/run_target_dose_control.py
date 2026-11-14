#!/usr/bin/env python3
"""Diagnostic OOI->FEBUS control with the exact final target dose, not equal total compute."""
import argparse
import fcntl
from functools import partial
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.control_reuse import audit_reusable_febus_control
from finetuning.finetune_vqvae import load_sample, load_valid_mask, save_checkpoint
from finetuning.model_factory import build_model
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import evaluate_detailed, validation_metadata
from finetuning.representation_study import digest, write_json
from finetuning.run_lr_schedule_study import capture_rng, restore_rng, clean_json
from finetuning.run_pretraining_bridge import optimizer_steps
from finetuning.update_schedule import load_independent_optimizer_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-study', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    audit = audit_reusable_febus_control(args.control_study, ROOT, args.repo_root)
    old = json.loads((args.control_study / 'protocol.json').read_text())
    paths = [ROOT / name for name in old['final_febus_plan']]
    pilot = [ROOT / name for name in old['pilot']]
    samples = Path(old['samples_dir']); metadata = validation_metadata(pilot, samples / 'samples_manifest.csv')
    out = args.output_dir; out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 6 * 1024**3:
        raise RuntimeError('6 GiB headroom required')
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = ['finetuning/run_target_dose_control.py', 'finetuning/evaluate_target_dose_control.py',
            'finetuning/control_reuse.py', *old['code_sha256']]
    protocol = {'kind': 'target_dose_only_control_v1', 'control_audit': audit,
        'source_sha256': old['source_sha256'], 'manifest_sha256': old['manifest_sha256'],
        'seed': old['seed'] + 1, 'updates': old['final_febus_updates'], 'block_size': old['block_size'],
        'target_plan': old['final_febus_plan'], 'pilot': old['pilot'], 'samples_dir': old['samples_dir'],
        'input_sha256': old['input_sha256'], 'optimizer': 'Fresh AdamW lr=1e-4 wd=1e-5, clip1, FP32 TF32=false',
        'scope': 'Diagnostic matched final FEBUS dose; fewer total updates than F/Q/P. Not a replacement promotion comparator.',
        'code_sha256': {name: digest(ROOT / name) for name in sorted(set(code))},
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'torch': str(torch.__version__),
        'final_holdout_accessed': False}
    if protocol['gpu_uuid'] != old['gpu_uuid'] or protocol['torch'] != old['torch']:
        raise ValueError('Diagnostic control GPU/software context changed')
    if protocol['gpu_uuid'] != old['gpu_uuid'] or protocol['torch'] != old['torch']:
        raise ValueError('Diagnostic control software/GPU changed')
    if args.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Resume protocol changed')
        if (out / 'COMPLETE').exists():
            return
    else:
        out.mkdir(exist_ok=False); write_json(out / 'protocol.json', protocol)
        shutil.copy2(args.control_study / 'initial.pt', out / 'initial.pt')
        for name in protocol['code_sha256']:
            dest = out / 'code_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
    if digest(out / 'initial.pt') != protocol['source_sha256']:
        raise ValueError('Diagnostic initial weights differ from reused control')
    initial = torch.load(out / 'initial.pt', map_location='cpu', weights_only=False)
    torch.manual_seed(protocol['seed'])
    model = build_model(initial['config'], initial, 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    load = partial(load_sample, normalization='robust-escape-v1')
    masks = partial(load_valid_mask, normalization='robust-escape-v1')

    def evaluate():
        return evaluate_detailed(model, pilot, metadata, torch.device('cuda'), load,
            load_valid_mask=masks, fidelity_transform=partial(restore_for_validation,
                policy='robust-escape-v1', model=model, detailed_channels=True), spatial_weighting=True)[0]

    last = out / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['protocol_sha256'] != digest(out / 'protocol.json'):
            raise ValueError('Retained diagnostic lineage differs')
        model.load_state_dict(saved['model_state_dict']); load_independent_optimizer_state(optimizer, saved['optimizer_state_dict'])
        restore_rng(saved['rng_state']); completed = saved['updates']
        best, best_name, history = saved['best_rmse'], saved['best_name'], saved['history']
        del saved
    else:
        best, best_name, history = evaluate()['macro_relative_rmse'], 'best_initial.pt', []
        shutil.copy2(out / 'initial.pt', out / best_name); completed = 0
        torch.manual_seed(protocol['seed']); random.seed(protocol['seed']); np.random.seed(protocol['seed'])

    def payload():
        return {'config': initial['config'], 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'updates': completed,
            'best_rmse': best, 'best_name': best_name, 'history': history, 'rng_state': capture_rng(),
            'protocol_sha256': digest(out / 'protocol.json')}

    try:
        save_checkpoint(payload(), last)
        for start in range(completed, protocol['updates'], protocol['block_size']):
            if any(digest(ROOT / name) != value for name, value in protocol['code_sha256'].items()):
                raise ValueError('Frozen diagnostic code changed')
            model.train(); began = time.perf_counter(); losses = []
            for path in paths[start:start + protocol['block_size']]:
                if digest(path) != protocol['input_sha256'][str(path.relative_to(ROOT))]:
                    raise ValueError('Target input changed')
                x = torch.from_numpy(load(path))[None, None].cuda(); valid = torch.from_numpy(masks(path))[None, None].cuda()
                optimizer.zero_grad(set_to_none=True)
                recon, vq, _, _ = model(x)
                loss = .5 * F.mse_loss(recon[valid], x[valid]) + .5 * F.l1_loss(recon[valid], x[valid]) + vq
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite diagnostic loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach()))
            completed = start + protocol['block_size']
            if optimizer_steps(optimizer) != [float(completed)]:
                raise ValueError('Diagnostic optimizer exposure mismatch')
            seconds = time.perf_counter() - began; metrics = evaluate()
            if not np.isfinite(metrics['macro_relative_rmse']):
                raise RuntimeError('Nonfinite diagnostic selection metric')
            if metrics['macro_relative_rmse'] < best:
                best, best_name = metrics['macro_relative_rmse'], f'best_{completed}.pt'
                save_checkpoint({'config': initial['config'], 'model_state_dict': model.state_dict(), 'updates': completed}, out / best_name)
            history.append({'updates': completed, 'train_seconds': seconds, 'train_loss': float(np.mean(losses)), **metrics})
            save_checkpoint(payload(), last); write_json(out / 'history.json', clean_json(history))
            write_json(out / 'STATUS.json', {'stage': 'training', 'updates': completed, 'budget': protocol['updates']})
            print(f"target-dose {completed}/{protocol['updates']} RMSE={metrics['macro_relative_rmse']:.6f}",flush=True)
        shutil.copy2(out / best_name, out / 'selected.pt')
        write_json(out / 'summary.json', {'updates': completed, 'final_optimizer_steps': optimizer_steps(optimizer),
            'best_pilot_rmse': best, 'best_name': best_name,
            'selected_sha256': digest(out / 'selected.pt'), 'endpoint_sha256': digest(last)})
        (out / 'TRAINING_COMPLETE').touch()
    finally:
        lock.close()


if __name__ == '__main__':
    main()
