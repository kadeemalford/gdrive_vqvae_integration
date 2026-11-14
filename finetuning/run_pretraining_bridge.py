#!/usr/bin/env python3
"""Bounded OOI->FEBUS versus OOI->QuantX phase->FEBUS, with matched final batches."""
import argparse
from collections import defaultdict
import csv
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
from finetuning.finetune_vqvae import load_sample, load_valid_mask, save_checkpoint
from finetuning.model_factory import build_model
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import evaluate_detailed, stratified_validation, validation_metadata
from finetuning.representation_study import digest, read_manifest, write_json
from finetuning.run_lr_schedule_study import capture_rng, restore_rng, clean_json
from finetuning.run_sampling_experiment import check_splits
from finetuning.sampling import epoch_paths
from finetuning.update_schedule import load_independent_optimizer_state

ARMS = ('febus', 'quantx')


def balanced_source_plan(rows, steps, seed):
    """Equal source-group dose, then uniform acquisition, then source window."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['family']][row['source_id']].append(row['sample_name'])
    families = sorted(grouped)
    if len(families) != 4 or steps % 4:
        raise ValueError('Four source groups and divisible update budget required')
    rng, result = random.Random(seed), []
    for step in range(steps):
        group = grouped[families[step % 4]]
        source_id = rng.choice(sorted(group))
        result.append(rng.choice(sorted(group[source_id])))
    return result


def optimizer_steps(optimizer):
    return sorted({float(s['step']) for s in optimizer.state.values()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offshore', type=Path, required=True)
    p.add_argument('--reference-protocol', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--quantx-dir', type=Path, required=True)
    p.add_argument('--quantx-preparation', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    reference = json.loads(args.reference_protocol.read_text())
    if digest(args.offshore) != reference['source_sha256']:
        raise ValueError('Expected original offshore weights, not FEBUS adapted source')
    if digest(args.samples_dir / 'samples_manifest.csv') != reference['metadata_sha256']['samples_manifest.csv']:
        raise ValueError('FEBUS manifest differs from audited cohort')
    if not (args.quantx_preparation / 'COMPLETE').exists():
        raise ValueError('Complete audited raw-phase preparation required')
    preparation = json.loads((args.quantx_preparation / 'summary.json').read_text())
    qmanifest = args.quantx_dir / 'source_manifest.csv'
    if digest(qmanifest) != preparation['manifest_sha256']:
        raise ValueError('QuantX preparation manifest changed')
    with qmanifest.open() as stream:
        qrows = list(csv.DictReader(stream))
    if any(int(r['time_samples']) != 1000 for r in qrows):
        raise ValueError('Unexpected QuantX temporal window')
    qhashes = {r['sample_name']: r['sha256'] for r in qrows}
    if any(digest(args.quantx_dir / name) != value for name, value in qhashes.items()):
        raise ValueError('Prepared QuantX array changed')
    check_splits(args.samples_dir, reference['test_fold'])
    splits = read_manifest(args.samples_dir)
    train = sorted(ROOT / r['sample_path'] for r in splits['train'])
    val = sorted(ROOT / r['sample_path'] for r in splits['val'])
    metadata = validation_metadata(val, args.samples_dir / 'samples_manifest.csv')
    pilot = stratified_validation(val, metadata, 2, reference['seed'])
    seed, block_size = 2096, 1 if args.smoke else 2048
    source_blocks, target_blocks = (4, 2) if args.smoke else (8, 4)
    source_budget, target_budget = source_blocks * block_size, target_blocks * block_size
    rng = random.Random(seed)
    source_f = [path for _ in range(source_blocks) for path in epoch_paths(train, {}, 'window', block_size, rng)]
    source_q = balanced_source_plan(qrows, source_budget, seed)
    rng = random.Random(seed + 1)
    final_paths = [path for _ in range(target_blocks) for path in epoch_paths(train, {}, 'window', block_size, rng)]
    if args.smoke:
        source_f[0] = ROOT / max(splits['train'], key=lambda row: int(row['channels']))['sample_path']
        pilot = [max(pilot, key=lambda path: int(metadata[path]['channels']))]
    out = args.output_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(out.parent).free < 8 * 1024**3:
        raise RuntimeError('At least 8 GiB disk headroom required')
    if args.resume:
        initial = torch.load(out / 'initial.pt', map_location='cpu', weights_only=False)
    else:
        out.mkdir(exist_ok=False)
        original = torch.load(args.offshore, map_location='cpu', weights_only=False)
        config = {k: original['config'][k] for k in
                  ('base_channels', 'embedding_dim', 'num_embeddings', 'downsample_levels', 'commitment_cost')}
        config.update(architecture='native-v1', latent_geometry='spatial-4x',
                      exact_eval_quantization=True, activation_checkpointing=True)
        initial = {'config': config, 'model_state_dict': original['model_state_dict'],
                   'offshore_sha256': digest(args.offshore), 'epoch': 0}
        save_checkpoint(initial, out / 'initial.pt')
        del original
    code = ['finetuning/run_pretraining_bridge.py', 'finetuning/update_schedule.py',
            'finetuning/model_factory.py', 'finetuning/finetune_vqvae.py',
            'finetuning/reconstruction_objective.py', 'finetuning/physical_validation.py',
            'finetuning/run_lr_schedule_study.py', 'finetuning/representation_study.py',
            'finetuning/evaluate_pretraining_bridge.py', 'finetuning/evaluate_lr_schedule_study.py',
            'compression/alphabet_stream.py', 'compression/study_bundle.py']
    used = sorted(set(source_f + final_paths + pilot))
    protocol = {'kind': 'raw_phase_pretraining_bridge_v1', 'smoke_only': args.smoke,
                'seed': seed, 'source_sha256': digest(out / 'initial.pt'), 'offshore_sha256': digest(args.offshore),
                'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
                'quantx_manifest_sha256': digest(qmanifest), 'quantx_preparation_sha256': digest(args.quantx_preparation / 'protocol.json'),
                'samples_dir': str(args.samples_dir.resolve()), 'quantx_dir': str(args.quantx_dir.resolve()),
                'arms': list(ARMS), 'source_updates': source_budget, 'final_febus_updates': target_budget,
                'block_size': block_size, 'learning_rate': 1e-4,
                'optimizer': 'Fresh full-model AdamW lr=1e-4 wd=1e-5 at start AND transition in both arms; clip=1; FP32/TF32=false',
                'source_febus_plan': [str(path.relative_to(ROOT)) for path in source_f],
                'source_quantx_plan': source_q,
                'final_febus_plan': [str(path.relative_to(ROOT)) for path in final_paths],
                'pilot': [str(path.relative_to(ROOT)) for path in pilot],
                'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in used},
                'quantx_input_sha256': qhashes,
                'code_sha256': {name: digest(ROOT / name) for name in code},
                'vendor_sha256': {name: digest(args.repo_root / 'vqvae' / name) for name in
                                  ('train_vqvae.py', 'evaluate_vqvae.py')},
                'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'torch': str(torch.__version__),
                'objective': '0.5*MSE+0.5*L1+VQ; same architecture/bits; original recorded QuantX phase, not calibrated FEBUS strain-rate',
                'selection': 'Global source initialization eligible; phase1 endpoint eligible; phase2 pilot minimum at every complete block, endpoints separate',
                'guards': '>=5% target RMSE gain vs FEBUS-only matched updates, complete bytes no larger, unchanged physical configuration guards',
                'exposure': 'Matched additional updates and identical final FEBUS batch sequence; not matched FLOPs or total FEBUS exposure',
                'extension': 'None', 'final_holdout_accessed': False,
                'limitations': ['12 previously audited QuantX recordings only, no independent source holdout',
                               'Source QuantX windows 100ms/70-100 loci; target FEBUS 50ms/native retained tiles',
                               'Source normalization retains drift/noise/all loci, no validated useful-fiber mask',
                               'Development FEBUS validation reused; no independent final generalization claim']}
    if args.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Bridge resume protocol/data/source/code changed')
        if (out / 'COMPLETE').exists():
            return
    else:
        write_json(out / 'protocol.json', protocol)
        for name in code:
            dest = out / 'code_snapshot' / name
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    load = partial(load_sample, normalization='robust-escape-v1')
    masks = partial(load_valid_mask, normalization='robust-escape-v1')

    def evaluate(model):
        return evaluate_detailed(model, pilot, metadata, torch.device('cuda'), load,
            load_valid_mask=masks, fidelity_transform=partial(restore_for_validation,
                policy='robust-escape-v1', model=model, detailed_channels=True), spatial_weighting=True)[0]

    try:
        for arm in ARMS:
            folder = out / arm; folder.mkdir(exist_ok=True)
            torch.manual_seed(seed)
            model = build_model(initial['config'], initial, 'cuda')
            last = folder / 'last.pt'
            saved = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
            if saved:
                if saved['bridge_protocol_sha256'] != digest(out / 'protocol.json'):
                    raise ValueError('Bridge branch lineage differs')
                model.load_state_dict(saved['model_state_dict'])
                best, best_name, history = saved['best_rmse'], saved['best_name'], saved['history']
            else:
                baseline = evaluate(model)
                if not np.isfinite(baseline['macro_relative_rmse']):
                    raise RuntimeError('Nonfinite initial FEBUS selection score')
                best, best_name, history = baseline['macro_relative_rmse'], 'best_initial.pt', []
                save_checkpoint(initial, folder / best_name)
            for phase, budget in [('source', source_budget), ('febus', target_budget)]:
                if saved and saved['phase'] == 'febus' and phase == 'source':
                    continue
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
                if saved and saved['phase'] == phase:
                    load_independent_optimizer_state(optimizer, saved['optimizer_state_dict'])
                    restore_rng(saved['rng_state']); completed = saved['phase_updates']
                else:
                    torch.manual_seed(seed if phase == 'source' else seed + 1)
                    np.random.seed(seed if phase == 'source' else seed + 1)
                    random.seed(seed if phase == 'source' else seed + 1)
                    completed = 0

                def payload():
                    return {'config': initial['config'], 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 'phase': phase,
                            'phase_updates': completed, 'rng_state': capture_rng(), 'history': history,
                            'best_rmse': best, 'best_name': best_name,
                            'bridge_protocol_sha256': digest(out / 'protocol.json')}

                save_checkpoint(payload(), last)
                for start in range(completed, budget, block_size):
                    if any(digest(ROOT / name) != value for name, value in protocol['code_sha256'].items()):
                        raise ValueError('Bridge code changed during run')
                    model.train(); started = time.perf_counter(); losses, perplexities = [], []
                    for step in range(start, start + block_size):
                        if phase == 'source' and arm == 'quantx':
                            name = source_q[step]; path = args.quantx_dir / name
                            if digest(path) != qhashes[name]:
                                raise ValueError('QuantX training input changed')
                            with np.load(path, allow_pickle=False) as data:
                                values = data['values']
                            x = torch.from_numpy(values)[None, None].cuda()
                            valid = torch.ones_like(x, dtype=torch.bool)
                        else:
                            path = source_f[step] if phase == 'source' else final_paths[step]
                            if digest(path) != protocol['input_sha256'][str(path.relative_to(ROOT))]:
                                raise ValueError('FEBUS training input changed')
                            x = torch.from_numpy(load(path))[None, None].cuda()
                            valid = torch.from_numpy(masks(path))[None, None].cuda()
                        optimizer.zero_grad(set_to_none=True)
                        recon, vq, _, perplexity = model(x)
                        loss = .5 * F.mse_loss(recon[valid], x[valid]) + .5 * F.l1_loss(recon[valid], x[valid]) + vq
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite bridge loss')
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step(); losses.append(float(loss.detach())); perplexities.append(float(perplexity.detach()))
                    completed = start + block_size
                    if optimizer_steps(optimizer) != [float(completed)]:
                        raise ValueError('Stage optimizer not reset/isolated correctly')
                    train_seconds = time.perf_counter() - started
                    metrics = evaluate(model) if phase == 'febus' or completed == budget else {}
                    if metrics and not np.isfinite(metrics['macro_relative_rmse']):
                        raise RuntimeError('Nonfinite FEBUS selection score')
                    if metrics and metrics['macro_relative_rmse'] < best:
                        best = metrics['macro_relative_rmse']; best_name = f'best_{phase}_{completed}.pt'
                        save_checkpoint({'config': initial['config'], 'model_state_dict': model.state_dict(),
                                         'phase': phase, 'updates': completed}, folder / best_name)
                    history.append({'phase': phase, 'updates': completed, 'train_seconds': train_seconds,
                                    'train_loss': float(np.mean(losses)), 'mean_perplexity': float(np.mean(perplexities)), **metrics})
                    save_checkpoint(payload(), last)
                    write_json(folder / 'history.json', clean_json(history))
                    write_json(out / 'STATUS.json', {'stage': phase, 'arm': arm, 'stage_updates': completed,
                                                     'stage_budget': budget, 'smoke_only': args.smoke})
                    print(f"bridge {arm}/{phase}: {completed}/{budget} train_s={train_seconds:.1f} pilot={metrics.get('macro_relative_rmse')}", flush=True)
                if phase == 'source':
                    save_checkpoint({'config': initial['config'], 'model_state_dict': model.state_dict(),
                                     'phase': phase, 'updates': completed}, folder / 'source_endpoint.pt')
                saved = None
            shutil.copy2(folder / best_name, folder / 'selected.pt')
            write_json(folder / 'summary.json', {'source_updates': source_budget, 'final_febus_updates': target_budget,
                'final_optimizer_steps': optimizer_steps(optimizer), 'best_pilot_rmse': best, 'best_name': best_name,
                'selected_sha256': digest(folder / 'selected.pt'), 'endpoint_sha256': digest(last)})
            del optimizer, model
            torch.cuda.empty_cache()
        (out / 'TRAINING_COMPLETE').touch()
        write_json(out / 'STATUS.json', {'stage': 'TRAINING_COMPLETE', 'full_validation_complete': False,
                                        'smoke_only': args.smoke, 'total_updates': 2 * (source_budget + target_budget)})
    finally:
        lock.close()


if __name__ == '__main__':
    main()
