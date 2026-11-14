#!/usr/bin/env python3
"""MSE-only continuation against an audited, exactly dose-matched historical control."""
import argparse
import fcntl
from functools import partial
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.finetune_vqvae import load_sample, load_valid_mask, save_checkpoint
from finetuning.model_factory import build_model
from finetuning.objective_ablation import audit_lr_control, reconstruction_loss
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import evaluate_detailed, validation_metadata
from finetuning.representation_study import digest, read_manifest, write_json
from finetuning.run_lr_schedule_study import capture_rng, restore_rng, clean_json
from finetuning.run_pretraining_bridge import optimizer_steps
from finetuning.run_sampling_experiment import check_splits
from finetuning.update_schedule import load_independent_optimizer_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-protocol', type=Path, required=True)
    parser.add_argument('--control-study', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090; local training is forbidden')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    out = args.output_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(out.parent).free < 8 * 1024**3:
        raise RuntimeError('8 GiB free required; no automatic cleanup')
    audit = audit_lr_control(args.control_study, args.source, ROOT, args.repo_root)
    old = json.loads((args.control_study / 'protocol.json').read_text())
    reference = json.loads(args.source_protocol.read_text())
    if digest(args.source_protocol) != old['source_protocol_sha256']:
        raise ValueError('Initial protocol differs from historical control')
    if digest(args.samples_dir / 'samples_manifest.csv') != old['manifest_sha256']:
        raise ValueError('Target corpus differs')
    check_splits(args.samples_dir, reference['test_fold'])
    splits = read_manifest(args.samples_dir)
    train_paths = {r['sample_path'] for r in splits['train']}
    if any(name not in train_paths for block in old['sequence'] for name in block):
        raise ValueError('Historical batch plan contains non-training data')
    sequence = [[ROOT / name for name in block] for block in old['sequence']]
    pilot = [ROOT / name for name in old['pilot']]
    if args.smoke:
        sequence = [[ROOT / max(splits['train'], key=lambda r: int(r['channels']))['sample_path']], [sequence[0][1]]]
        rows = {r['sample_path']: r for r in splits['val']}
        pilot = [max(pilot, key=lambda path: int(rows[str(path.relative_to(ROOT))]['channels']))]
    blocks, block_size = len(sequence), len(sequence[0])
    budget = blocks * block_size
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    if (source['protocol_sha256'] != digest(args.source_protocol)
            or {float(v['step']) for v in source['optimizer_state_dict']['state'].values()} != {98304.}):
        raise ValueError('Original source lineage/counters differ')
    code = ['finetuning/run_objective_ablation.py', 'finetuning/evaluate_objective_ablation.py',
            'finetuning/objective_ablation.py', 'finetuning/evaluate_pretraining_bridge.py', *old['code_sha256']]
    protocol = {'kind': 'reconstruction_objective_only_ablation_v1', 'objective': 'mse',
        'source_sha256': old['source_sha256'], 'source_protocol_sha256': old['source_protocol_sha256'],
        'control_audit': audit, 'manifest_sha256': old['manifest_sha256'],
        'samples_dir': str(args.samples_dir.resolve()), 'smoke_only': args.smoke,
        'updates': budget, 'block_size': block_size,
        'sequence': [[str(path.relative_to(ROOT)) for path in block] for block in sequence],
        'pilot': [str(path.relative_to(ROOT)) for path in pilot],
        'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(pilot + sum(sequence, [])))},
        'code_sha256': {name: digest(ROOT / name) for name in sorted(set(code))},
        'vendor_sha256': old['vendor_sha256'], 'learning_rate': 1e-4,
        'optimizer': 'Inherited AdamW moments, steps, betas and wd; same constant LR, clip1, FP32 TF32=false',
        'change': 'Reconstruction term MSE instead of 0.5*MSE+0.5*L1; unchanged VQ loss coefficient',
        'interpretation': 'Practical objective change, including relative reconstruction/VQ scale; not isolated mean-versus-median mechanism or exact physical macro-RMSE optimization',
        'selection': old['selection'], 'extension': 'None; fixed matched 16384 updates',
        'confirmation_gate': '>=5% lower physical macro relative RMSE than reused mixed control, nonincreasing complete stream bytes, unchanged configuration guards',
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'torch': str(torch.__version__),
        'final_holdout_accessed': False}
    if protocol['gpu_uuid'] != old['gpu_uuid'] or protocol['torch'] != old['torch']:
        raise ValueError('Reuse requires the same GPU/software context')
    if args.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Objective continuation resume protocol changed')
        if (out / 'TRAINING_COMPLETE').exists():
            return
    else:
        out.mkdir(exist_ok=False)
        write_json(out / 'protocol.json', protocol)
        for name in protocol['code_sha256']:
            target = out / 'code_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
    config = {**source['config'], 'exact_eval_quantization': True, 'activation_checkpointing': True}
    model = build_model(config, source, 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    load = partial(load_sample, normalization='robust-escape-v1')
    masks = partial(load_valid_mask, normalization='robust-escape-v1')
    metadata = validation_metadata(pilot, args.samples_dir / 'samples_manifest.csv')

    def evaluate():
        return evaluate_detailed(model, pilot, metadata, torch.device('cuda'), load,
            load_valid_mask=masks, fidelity_transform=partial(restore_for_validation,
                policy='robust-escape-v1', model=model, detailed_channels=True), spatial_weighting=True)[0]

    folder = out / 'mse'; folder.mkdir(exist_ok=True)
    last = folder / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['objective_protocol_sha256'] != digest(out / 'protocol.json'):
            raise ValueError('Retained objective branch lineage differs')
        model.load_state_dict(saved['model_state_dict'])
        load_independent_optimizer_state(optimizer, saved['optimizer_state_dict'])
        restore_rng(saved['rng_state'])
        completed, history = saved['updates'], saved['history']
        best, best_block = saved['best_rmse'], saved['best_block']
        del saved
    else:
        load_independent_optimizer_state(optimizer, source['optimizer_state_dict'])
        restore_rng(source['finetune']['rng_state'])
        metrics = evaluate()
        best, best_block, completed = metrics['macro_relative_rmse'], 0, 0
        history = [{'block': 0, **metrics}]
        if not args.smoke:
            old_initial = json.loads((args.control_study / 'constant_high/history.json').read_text())[0]
            if not np.isclose(best, old_initial['macro_relative_rmse'], rtol=0, atol=1e-10):
                raise ValueError('Initial target pilot does not reproduce historical control')
        save_checkpoint({'config': config, 'model_state_dict': model.state_dict()}, folder / 'best_block_0.pt')
    if completed % block_size or optimizer_steps(optimizer) != [98304. + completed]:
        raise ValueError('Optimizer dose differs on resume/start')
    try:
        for block in range(completed // block_size, blocks):
            if any(digest(ROOT / name) != expected for name, expected in protocol['code_sha256'].items()):
                raise ValueError('Frozen objective code changed')
            model.train(); began = time.perf_counter()
            diagnostics = []
            for path in sequence[block]:
                if digest(path) != protocol['input_sha256'][str(path.relative_to(ROOT))]:
                    raise ValueError('Training array changed')
                x = torch.from_numpy(load(path))[None, None].cuda()
                valid = torch.from_numpy(masks(path))[None, None].cuda()
                for group in optimizer.param_groups:
                    group['lr'] = 1e-4
                optimizer.zero_grad(set_to_none=True)
                reconstruction, vq, _, perplexity = model(x)
                mse = reconstruction_loss(reconstruction, x, valid, 'mse')
                l1 = F.l1_loss(reconstruction[valid], x[valid])
                loss = mse + vq
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite MSE objective')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); completed += 1
                diagnostics.append([float(loss.detach()), float(mse.detach()), float(l1.detach()),
                                    float(vq.detach()), float(perplexity.detach()), float(norm)])
            seconds = time.perf_counter() - began
            if optimizer_steps(optimizer) != [98304. + completed]:
                raise ValueError('Objective branch optimizer dose differs')
            if {float(v['step']) for v in source['optimizer_state_dict']['state'].values()} != {98304.}:
                raise ValueError('Original optimizer in memory mutated')
            metrics = evaluate()
            if not np.isfinite(metrics['macro_relative_rmse']):
                raise RuntimeError('Nonfinite objective selection score')
            if metrics['macro_relative_rmse'] < best:
                best, best_block = metrics['macro_relative_rmse'], block + 1
                save_checkpoint({'config': config, 'model_state_dict': model.state_dict()}, folder / f'best_block_{best_block}.pt')
            means = np.mean(diagnostics, axis=0)
            history.append({'block': block + 1, 'updates': completed, 'training_seconds': seconds,
                **dict(zip(('train_loss', 'train_mse', 'train_l1', 'train_vq', 'train_perplexity', 'gradient_norm'), map(float, means))),
                'clipped_fraction': float(np.mean(np.asarray(diagnostics)[:, -1] > 1)), **metrics})
            save_checkpoint({'config': config, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'rng_state': capture_rng(),
                'updates': completed, 'best_rmse': best, 'best_block': best_block, 'history': history,
                'objective_protocol_sha256': digest(out / 'protocol.json')}, last)
            write_json(folder / 'history.json', clean_json(history))
            write_json(out / 'STATUS.json', {'stage': 'training', 'updates': completed, 'budget': budget, 'smoke_only': args.smoke})
            print(f"objective mse {completed}/{budget} pilot_RMSE={metrics['macro_relative_rmse']:.6f} train_s={seconds:.1f}", flush=True)
        if digest(args.source) != protocol['source_sha256']:
            raise ValueError('Source file changed')
        shutil.copy2(folder / f'best_block_{best_block}.pt', folder / 'selected.pt')
        write_json(folder / 'summary.json', {'updates': completed, 'best_block': best_block,
            'best_pilot_rmse': best, 'final_optimizer_steps': optimizer_steps(optimizer),
            'selected_sha256': digest(folder / 'selected.pt'), 'endpoint_sha256': digest(last)})
        write_json(out / 'STATUS.json', {'stage': 'TRAINING_COMPLETE', 'smoke_only': args.smoke})
        (out / 'TRAINING_COMPLETE').touch()
    finally:
        lock.close()


if __name__ == '__main__':
    main()
