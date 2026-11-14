#!/usr/bin/env python3
"""Three paired continuations, exact optimizer/RNG branching, fixed update budgets."""
import argparse
import csv
import fcntl
from functools import partial
import hashlib
import json
import os
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
from finetuning.finetune_vqvae import load_sample, load_valid_mask, save_checkpoint
from finetuning.model_factory import build_model
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import evaluate_detailed, stratified_validation, validation_metadata
from finetuning.representation_study import digest, read_manifest, write_json
from finetuning.run_sampling_experiment import check_splits
from finetuning.sampling import epoch_paths
from finetuning.update_schedule import UpdateSchedule, load_independent_optimizer_state

ARMS = ('constant_high', 'cosine', 'constant_mean')


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if len(state['cuda']) != 1:
        raise ValueError('Source must contain one CUDA RNG state')
    torch.cuda.set_rng_state(state['cuda'][0].cpu(), 0)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-protocol', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX 4090; CPU/local training forbidden')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    state = source['finetune']
    original_optimizer_steps = [float(v['step']) for v in source['optimizer_state_dict']['state'].values()]
    if set(original_optimizer_steps) != {98304.}:
        raise ValueError('Source optimizer must be at exactly 98304 updates')
    source_protocol = json.loads(args.source_protocol.read_text())
    if source['protocol_sha256'] != digest(args.source_protocol):
        raise ValueError('Source checkpoint/protocol lineage mismatch')
    expected = {'normalization': 'robust-escape-v1', 'sampling': 'window', 'trainable': 'full',
                'learning_rate': 1e-4, 'weight_decay': 1e-5, 'spectral_weight': 0.,
                'channel_mse_weight': 0., 'codebook_refresh': 'none', 'domain_family': None}
    if source['epoch'] != 24 or source['initialization']['arm'] != 'A' or any(state.get(k) != v for k, v in expected.items()):
        raise ValueError('Expected compact transferred A endpoint epoch24 with full optimizer/RNG')
    if source['config']['latent_geometry'] != 'spatial-4x' or 'optimizer_state_dict' not in source:
        raise ValueError('Source geometry or optimizer missing')
    if Path(state['samples_dir']).name != args.samples_dir.name:
        raise ValueError('Source corpus differs')
    if digest(args.samples_dir / 'samples_manifest.csv') != source_protocol['metadata_sha256']['samples_manifest.csv']:
        raise ValueError('Source manifest differs')
    check_splits(args.samples_dir, source_protocol['test_fold'])
    splits = read_manifest(args.samples_dir)
    train = sorted(ROOT / r['sample_path'] for r in splits['train'])
    val = sorted(ROOT / r['sample_path'] for r in splits['val'])
    metadata = validation_metadata(val, args.samples_dir / 'samples_manifest.csv')
    pilot = stratified_validation(val, metadata, 2, source_protocol['seed'])
    original_pilot = sorted(str(Path(state['samples_dir']) / 'val' / path.name) for path in pilot)
    if hashlib.sha256('\n'.join(original_pilot).encode()).hexdigest() != state['validation_hash']:
        raise ValueError('Historical pilot selection differs after path relocation')
    blocks, block_size = (2, 1) if args.smoke else (8, 2048)
    total = blocks * block_size
    order_rng = random.Random()
    order_rng.setstate(state['rng_state']['python'])
    sequence = [epoch_paths(train, {}, 'window', block_size, order_rng) for _ in range(blocks)]
    if args.smoke:
        largest = max(splits['train'], key=lambda r: int(r['channels']))
        sequence[0][0] = ROOT / largest['sample_path']
        pilot = [max(pilot, key=lambda path: int(metadata[path]['channels']))]
    out = args.output_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 8 * 1024 ** 3:
        raise RuntimeError('At least 8 GiB free required')
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code = ['finetuning/run_lr_schedule_study.py', 'finetuning/update_schedule.py',
            'finetuning/finetune_vqvae.py', 'finetuning/model_factory.py', 'finetuning/latent_geometry.py',
            'finetuning/reconstruction_objective.py', 'finetuning/physical_validation.py',
            'finetuning/robust_escape.py', 'finetuning/reversible_normalization.py',
            'finetuning/evaluate_lr_schedule_study.py', 'compression/alphabet_stream.py',
            'compression/study_bundle.py', 'finetuning/run_index_search.py']
    input_paths = sorted(set([*pilot, *(path for block in sequence for path in block)]))
    protocol = {'kind': 'paired_update_lr_continuation_v1', 'source_sha256': digest(args.source),
                'source_protocol_sha256': digest(args.source_protocol), 'source_epoch': source['epoch'],
                'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
                'samples_dir': str(args.samples_dir.resolve()), 'smoke_only': args.smoke,
                'arms': list(ARMS), 'updates_per_arm': total, 'blocks': blocks, 'block_size': block_size,
                'lr_high': 1e-4, 'lr_low': 1e-5, 'optimizer': 'inherited AdamW moments/steps/betas/weight decay, not reset',
                'source_rng': 'same saved Torch/CUDA/NumPy/Python RNG for all arms; explicit shared batch plan',
                'sequence': [[str(path.relative_to(ROOT)) for path in block] for block in sequence],
                'pilot': [str(path.relative_to(ROOT)) for path in pilot],
                'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in input_paths},
                'code_sha256': {name: digest(ROOT / name) for name in code},
                'vendor_sha256': {name: digest(args.repo_root / 'vqvae' / name) for name in ('train_vqvae.py', 'evaluate_vqvae.py')},
                'gpu': torch.cuda.get_device_name(0), 'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch': str(torch.__version__),
                'selection': 'minimum pilot macro relative RMSE, initial source eligible, endpoint retained separately',
                'full_validation': 'after all three training arms; selected and endpoint; exact discrete replay and complete streams',
                'no_test_signal_access': True, 'extension': 'none; >=1% over both constants, nonincreasing bytes and unchanged physical guards required before confirmation'}
    if args.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Resume protocol/input/code/source differs')
        if (out / 'COMPLETE').exists():
            print('Study already complete; no new updates.', flush=True)
            return
    else:
        out.mkdir(exist_ok=False)
        write_json(out / 'protocol.json', protocol)
        for name in code:
            dest = out / 'code_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        shutil.copy2(args.source_protocol, out / 'source_protocol.json')
    load = partial(load_sample, normalization='robust-escape-v1')
    masks = partial(load_valid_mask, normalization='robust-escape-v1')

    def evaluate(model):
        return evaluate_detailed(model, pilot, metadata, torch.device('cuda'), load,
            load_valid_mask=masks, fidelity_transform=partial(restore_for_validation,
                policy='robust-escape-v1', model=model, detailed_channels=True), spatial_weighting=True)[0]

    def check_immutable():
        if digest(args.source) != protocol['source_sha256']:
            raise ValueError('Source checkpoint changed')
        if any(digest(ROOT / name) != value for name, value in protocol['code_sha256'].items()):
            raise ValueError('Source code changed while running')

    try:
        for arm in ARMS:
            folder = out / arm
            folder.mkdir(exist_ok=True)
            schedule = UpdateSchedule(arm, total)
            config = {**source['config'], 'exact_eval_quantization': True, 'activation_checkpointing': True}
            model = build_model(config, source, 'cuda')
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            last = folder / 'last.pt'
            if last.exists():
                saved = torch.load(last, map_location='cpu', weights_only=False)
                if saved['schedule_protocol_sha256'] != digest(out / 'protocol.json'):
                    raise ValueError('Branch lineage mismatch')
                model.load_state_dict(saved['model_state_dict'])
                load_independent_optimizer_state(optimizer, saved['optimizer_state_dict'])
                schedule.load_state_dict(saved['schedule_state'])
                restore_rng(saved['finetune']['rng_state'])
                best, best_block = saved['selection']['best_rmse'], saved['selection']['best_block']
                history = saved['history']
                if schedule.completed_updates % block_size:
                    raise ValueError('Only complete-block snapshots may resume')
                del saved
            else:
                load_independent_optimizer_state(optimizer, source['optimizer_state_dict'])
                if any(float(v['step']) != 98304 for v in optimizer.state.values()):
                    raise ValueError('Branch optimizer initial step differs from source')
                restore_rng(state['rng_state'])
                initial = evaluate(model)
                best, best_block, history = initial['macro_relative_rmse'], 0, [{'block': 0, **initial}]
                save_checkpoint({'config': config, 'model_state_dict': model.state_dict(), 'epoch': 0,
                                 'source_sha256': protocol['source_sha256']}, folder / 'best_block_0.pt')

            def payload(block):
                return {'config': config, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': block, 'schedule_protocol_sha256': digest(out / 'protocol.json'),
                        'schedule_state': schedule.state_dict(), 'selection': {'best_rmse': best, 'best_block': best_block},
                        'history': history, 'finetune': {**state, 'rng_state': capture_rng()}}

            for block in range(schedule.completed_updates // block_size, blocks):
                check_immutable()
                model.train()
                started = time.perf_counter()
                losses, rates = [], []
                for index, path in enumerate(sequence[block]):
                    expected_step = block * block_size + index
                    if schedule.completed_updates != expected_step:
                        raise ValueError('Schedule update mismatch')
                    if digest(path) != protocol['input_sha256'][str(path.relative_to(ROOT))]:
                        raise ValueError('Training array changed')
                    x = torch.from_numpy(load(path))[None, None].cuda()
                    valid = torch.from_numpy(masks(path))[None, None].cuda()
                    lr = schedule.prepare_update(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    recon, vq_loss, _, _ = model(x)
                    loss = .5 * F.mse_loss(recon[valid], x[valid]) + .5 * F.l1_loss(recon[valid], x[valid]) + vq_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    schedule.advance()
                    losses.append(float(loss.detach())); rates.append(lr)
                training_seconds = time.perf_counter() - started
                metrics = evaluate(model)
                if not np.isfinite(metrics['macro_relative_rmse']):
                    raise RuntimeError('Nonfinite selection score')
                improved = metrics['macro_relative_rmse'] < best
                if improved:
                    best, best_block = metrics['macro_relative_rmse'], block + 1
                row = {'block': block + 1, 'updates': schedule.completed_updates,
                       'training_seconds': training_seconds, 'train_loss': float(np.mean(losses)),
                       'lr_first': rates[0], 'lr_last': rates[-1], **metrics}
                history.append(row)
                if improved:
                    save_checkpoint({'config': config, 'model_state_dict': model.state_dict(), 'epoch': block + 1,
                                     'source_sha256': protocol['source_sha256']}, folder / f'best_block_{block+1}.pt')
                save_checkpoint(payload(block + 1), last)
                if any(float(v['step']) != 98304 for v in source['optimizer_state_dict']['state'].values()):
                    raise ValueError('In-memory source optimizer was mutated')
                if any(float(v['step']) != 98304 + schedule.completed_updates for v in optimizer.state.values()):
                    raise ValueError('Branch optimizer step exceeds paired exposure')
                write_json(folder / 'history.json', clean_json(history))
                write_json(out / 'STATUS.json', {'stage': 'training', 'arm': arm,
                    'completed_arm_updates': schedule.completed_updates, 'budget_per_arm': total,
                    'block': block + 1, 'best_pilot_rmse': best, 'smoke_only': args.smoke})
                print(f"{arm} block={block+1}/{blocks} updates={schedule.completed_updates} RMSE={metrics['macro_relative_rmse']:.6f} LR={rates[-1]:.8g} train_s={training_seconds:.1f}", flush=True)
            shutil.copy2(folder / f'best_block_{best_block}.pt', folder / 'selected.pt')
            final_optimizer_steps = sorted({float(v['step']) for v in optimizer.state.values()})
            if final_optimizer_steps != [98304. + total]:
                raise ValueError('Final optimizer exposure differs from paired budget')
            write_json(folder / 'summary.json', {'arm': arm, 'updates': schedule.completed_updates,
                'best_block': best_block, 'best_pilot_rmse': best, 'source_optimizer_preserved': True,
                'initial_optimizer_steps': original_optimizer_steps, 'final_optimizer_steps': final_optimizer_steps,
                'schedule_state': schedule.state_dict(), 'selected_sha256': digest(folder / 'selected.pt'),
                'last_sha256': digest(last)})
            del model, optimizer
            torch.cuda.empty_cache()
        write_json(out / 'STATUS.json', {'stage': 'TRAINING_COMPLETE', 'updates': 3 * total,
            'smoke_only': args.smoke, 'full_validation_complete': False})
        (out / 'TRAINING_COMPLETE').touch()
    finally:
        lock.close()


if __name__ == '__main__':
    main()
