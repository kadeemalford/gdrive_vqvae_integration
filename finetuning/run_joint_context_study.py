#!/usr/bin/env python3
"""Paired full-model context adaptation: original CNN, codebook and both contexts train."""
import argparse
import fcntl
from functools import partial
import json
import os
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
from finetuning.joint_context_vq import ARMS, initialize_joint_context, inherited_joint_optimizer, assert_joint_optimizer_steps
from finetuning.latent_context_adapter import LatentContextAdapter
from finetuning.objective_ablation import audit_lr_control
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import evaluate_detailed, validation_metadata
from finetuning.representation_study import digest, read_manifest, write_json
from finetuning.run_decoder_aware_codebooks import source_state_hash
from finetuning.run_lr_schedule_study import capture_rng, restore_rng, clean_json
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
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.backends.mha.set_fastpath_enabled(False)
    sys.path.insert(0, str(args.repo_root.resolve()))
    out = args.output_dir; out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(out.parent).free < 10 * 1024**3:
        raise RuntimeError('10 GiB headroom required')
    audit = audit_lr_control(args.control_study, args.source, ROOT, args.repo_root)
    old = json.loads((args.control_study / 'protocol.json').read_text())
    if digest(args.source_protocol) != old['source_protocol_sha256']:
        raise ValueError('Original initialization protocol differs')
    reference = json.loads(args.source_protocol.read_text())
    if digest(args.samples_dir / 'samples_manifest.csv') != old['manifest_sha256']:
        raise ValueError('Target corpus differs')
    check_splits(args.samples_dir, reference['test_fold'])
    rows = read_manifest(args.samples_dir)
    train_names = {row['sample_path'] for row in rows['train']}
    if any(name not in train_names for block in old['sequence'] for name in block):
        raise ValueError('Historical training sequence includes other splits')
    sequence = [[ROOT / name for name in block] for block in old['sequence']]
    pilot = [ROOT / name for name in old['pilot']]
    if args.smoke:
        sequence = [[ROOT / max(rows['train'], key=lambda row: int(row['channels']))['sample_path']], [sequence[0][1]]]
        val_lookup = {row['sample_path']: row for row in rows['val']}
        pilot = [max(pilot, key=lambda path: int(val_lookup[str(path.relative_to(ROOT))]['channels']))]
    blocks, block_size = len(sequence), len(sequence[0])
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    if source['protocol_sha256'] != digest(args.source_protocol):
        raise ValueError('Source checkpoint lineage differs')
    if {float(s['step']) for s in source['optimizer_state_dict']['state'].values()} != {98304.}:
        raise ValueError('Source optimizer must begin at98304')
    code = ['finetuning/run_joint_context_study.py', 'finetuning/evaluate_joint_context_study.py',
            'finetuning/joint_context_vq.py', 'finetuning/latent_context_adapter.py',
            'finetuning/objective_ablation.py', 'finetuning/run_decoder_aware_codebooks.py', *old['code_sha256']]
    counts = {kind: 2 * sum(p.numel() for p in LatentContextAdapter('shape' if kind == 'pointwise' else 'axial').parameters()) for kind in ARMS}
    if max(counts.values()) / min(counts.values()) >= 1.01:
        raise ValueError('Context control capacities differ by >=1%')
    protocol = {'kind': 'joint_analysis_synthesis_context_study_v1', 'seed': 2108, 'smoke_only': args.smoke,
        'source_sha256': old['source_sha256'], 'source_protocol_sha256': old['source_protocol_sha256'],
        'control_audit': audit, 'manifest_sha256': old['manifest_sha256'], 'samples_dir': str(args.samples_dir.resolve()),
        'arms': list(ARMS), 'extra_parameters': counts, 'updates_per_arm': blocks * block_size, 'block_size': block_size,
        'sequence': [[str(path.relative_to(ROOT)) for path in block] for block in sequence],
        'pilot': [str(path.relative_to(ROOT)) for path in pilot],
        'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(pilot + sum(sequence, [])))},
        'code_sha256': {name: digest(ROOT / name) for name in sorted(set(code))}, 'vendor_sha256': old['vendor_sha256'],
        'initialization': 'Original A epoch24 backbone/codebook identical; shape-aware pointwise and axial contexts on BOTH sides start as exact identity',
        'optimizer': 'Base AdamW moments/steps inherited independently at98304; context moments fresh; all params lr1e-4 wd1e-5, global clip1',
        'trainable': 'All encoder, codebook, decoder, analysis and synthesis context parameters; no frozen backbone',
        'objective': 'Historical 0.5 MSE+0.5 L1+VQ, ordinary-cell mask unchanged',
        'control': 'Pointwise and axial receive known shape/positions, same batches/seeds/base moments and nearly identical parameter count; no temporal history across frames',
        'native_reference': 'Historical unaugmented constant-high continuation reused as practical reference; it has fewer parameters and was not run with math-attention settings',
        'selection': old['selection'], 'extension': 'None; fixed16384 per arm',
        'confirmation_gate': 'Axial must reduce physical macro RMSE>=5% vs BOTH pointwise and native references with no more complete bytes and no configuration physical guard regression',
        'wire': 'Only IDs, known shape and original side reach decoder; exact code-vector replay then synthesis context; no analysis feature or attention KV cache transmitted',
        'cost': 'Same nominal grid/K, not presumed equal entropy bytes/FLOPs/time; additional decoder context and encoder context weights reported separately',
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'torch': str(torch.__version__),
        'numerics': 'FP32 TF32false, deterministic algorithms, CUBLAS4096:8, math attention, MHAfastpathfalse',
        'final_holdout_accessed': False}
    if protocol['gpu_uuid'] != old['gpu_uuid'] or protocol['torch'] != old['torch']:
        raise ValueError('Historical practical reference GPU/software differs')
    if args.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Joint-context resume changed')
        if (out / 'TRAINING_COMPLETE').exists():
            return
    else:
        out.mkdir(exist_ok=False); write_json(out / 'protocol.json', protocol)
        for name in protocol['code_sha256']:
            dest = out / 'code_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    metadata = validation_metadata(pilot, args.samples_dir / 'samples_manifest.csv')
    load = partial(load_sample, normalization='robust-escape-v1')
    masks = partial(load_valid_mask, normalization='robust-escape-v1')
    for kind in ARMS:
        torch.manual_seed(protocol['seed'])
        model, config = initialize_joint_context(source, kind, 'cuda')
        model.requires_grad_(True)
        base_initial_hash = source_state_hash(model.base)
        optimizer = inherited_joint_optimizer(model, source['optimizer_state_dict'])
        folder = out / kind; folder.mkdir(exist_ok=True); last = folder / 'last.pt'

        def evaluate():
            return evaluate_detailed(model, pilot, metadata, torch.device('cuda'), load,
                load_valid_mask=masks, fidelity_transform=partial(restore_for_validation,
                    policy='robust-escape-v1', model=model, detailed_channels=True), spatial_weighting=True)[0]

        if last.exists():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            if saved['joint_protocol_sha256'] != digest(out / 'protocol.json'):
                raise ValueError('Joint-context branch lineage differs')
            model.load_state_dict(saved['model_state_dict'])
            load_independent_optimizer_state(optimizer, saved['optimizer_state_dict'])
            restore_rng(saved['rng_state'])
            completed, history = saved['updates'], saved['history']
            best, best_block = saved['best_rmse'], saved['best_block']; del saved
        else:
            restore_rng(source['finetune']['rng_state'])
            initial = evaluate(); best, best_block, completed = initial['macro_relative_rmse'], 0, 0
            if not args.smoke:
                historical = json.loads((args.control_study / 'constant_high/history.json').read_text())[0]
                if not np.isclose(best, historical['macro_relative_rmse'], rtol=0, atol=1e-10):
                    raise ValueError('Identity-initialized context does not reproduce source pilot')
            history = [{'block': 0, **initial}]
            save_checkpoint({'config': config, 'model_state_dict': model.state_dict()}, folder / 'best_block_0.pt')
        if completed % block_size:
            raise ValueError('Resume only at complete block boundary')
        assert_joint_optimizer_steps(model, optimizer, completed)
        for block in range(completed // block_size, blocks):
            if any(digest(ROOT / name) != expected for name, expected in protocol['code_sha256'].items()):
                raise ValueError('Frozen joint-context code changed')
            model.train(); began = time.perf_counter(); losses, perplexities = [], []
            for path in sequence[block]:
                if digest(path) != protocol['input_sha256'][str(path.relative_to(ROOT))]:
                    raise ValueError('Training array changed')
                x = torch.from_numpy(load(path))[None, None].cuda()
                valid = torch.from_numpy(masks(path))[None, None].cuda()
                optimizer.zero_grad(set_to_none=True)
                predicted, vq, _, perplexity = model(x)
                loss = .5 * F.mse_loss(predicted[valid], x[valid]) + .5 * F.l1_loss(predicted[valid], x[valid]) + vq
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite joint-context loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); completed += 1
                losses.append(float(loss.detach())); perplexities.append(float(perplexity.detach()))
            seconds = time.perf_counter() - began
            assert_joint_optimizer_steps(model, optimizer, completed)
            if {float(s['step']) for s in source['optimizer_state_dict']['state'].values()} != {98304.}:
                raise AssertionError('Source optimizer mutated')
            metrics = evaluate()
            if not np.isfinite(metrics['macro_relative_rmse']):
                raise RuntimeError('Nonfinite joint selection metric')
            if metrics['macro_relative_rmse'] < best:
                best, best_block = metrics['macro_relative_rmse'], block + 1
                save_checkpoint({'config': config, 'model_state_dict': model.state_dict()}, folder / f'best_block_{best_block}.pt')
            history.append({'block': block + 1, 'updates': completed, 'training_seconds': seconds,
                            'train_loss': float(np.mean(losses)), 'train_perplexity': float(np.mean(perplexities)), **metrics})
            save_checkpoint({'config': config, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': capture_rng(), 'updates': completed, 'history': history, 'best_rmse': best, 'best_block': best_block,
                'joint_protocol_sha256': digest(out / 'protocol.json')}, last)
            write_json(folder / 'history.json', clean_json(history))
            write_json(out / 'STATUS.json', {'stage': 'training', 'arm': kind, 'updates': completed,
                                            'budget': blocks * block_size, 'smoke_only': args.smoke})
            print(f"joint-context {kind} {completed}/{blocks * block_size} pilot_RMSE={metrics['macro_relative_rmse']:.6f} train_s={seconds:.1f}", flush=True)
        if digest(args.source) != protocol['source_sha256']:
            raise AssertionError('Source file changed')
        shutil.copy2(folder / f'best_block_{best_block}.pt', folder / 'selected.pt')
        write_json(folder / 'summary.json', {'updates': completed, 'best_block': best_block, 'best_pilot_rmse': best,
            'base_optimizer_step': 98304 + completed, 'context_optimizer_step': completed,
            'all_parameters_trainable': all(p.requires_grad for p in model.parameters()),
            'base_weights_changed': source_state_hash(model.base) != base_initial_hash,
            'selected_sha256': digest(folder / 'selected.pt'), 'endpoint_sha256': digest(last),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
        del model, optimizer
        torch.cuda.empty_cache()
    write_json(out / 'STATUS.json', {'stage': 'TRAINING_COMPLETE', 'smoke_only': args.smoke})
    (out / 'TRAINING_COMPLETE').touch(); lock.close()


if __name__ == '__main__':
    main()
