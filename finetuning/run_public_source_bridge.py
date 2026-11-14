#!/usr/bin/env python3
"""Train one public-source bridge while reusing an audited FEBUS-only control."""
import argparse
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
from finetuning.control_reuse import audit_reusable_febus_control
from finetuning.finetune_vqvae import load_sample, load_valid_mask, save_checkpoint
from finetuning.model_factory import build_model
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import evaluate_detailed, validation_metadata
from finetuning.representation_study import digest, read_manifest, write_json
from finetuning.run_lr_schedule_study import capture_rng, restore_rng, clean_json
from finetuning.run_pretraining_bridge import balanced_source_plan, optimizer_steps
from finetuning.update_schedule import load_independent_optimizer_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-study', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--source-preparation', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one remote RTX4090 must be exposed')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    audit = audit_reusable_febus_control(args.control_study, ROOT, args.repo_root, allow_smoke=args.smoke)
    control = json.loads((args.control_study / 'protocol.json').read_text())
    if control['smoke_only'] != args.smoke:
        raise ValueError('Smoke/scientific control mismatch')
    if not (args.source_preparation / 'COMPLETE').is_file():
        raise ValueError('Do not train on an incomplete source preparation')
    preparation = json.loads((args.source_preparation / 'protocol.json').read_text())
    summary = json.loads((args.source_preparation / 'summary.json').read_text())
    manifest = args.source_dir / 'source_manifest.csv'
    if preparation['kind'] != 'porotomo_multiday_source_preparation_v2' or digest(manifest) != summary['manifest_sha256']:
        raise ValueError('Unexpected public source preparation/manifest')
    with manifest.open() as stream:
        source_rows = list(csv.DictReader(stream))
    source_hashes = {r['sample_name']: r['sha256'] for r in source_rows}
    if any(digest(args.source_dir / name) != value for name, value in source_hashes.items()):
        raise ValueError('Public source array changed')
    seed, block = control['seed'], control['block_size']
    source_budget, target_budget = control['source_updates'], control['final_febus_updates']
    source_plan = balanced_source_plan(source_rows, source_budget, seed)
    final_plan = [ROOT / path for path in control['final_febus_plan']]
    samples = Path(control['samples_dir'])
    splits = read_manifest(samples)
    training_paths = {r['sample_path'] for r in splits['train']}
    if any(path not in training_paths for path in control['source_febus_plan'] + control['final_febus_plan']):
        raise ValueError('Control sequences contain non-training target data')
    pilot = [ROOT / name for name in control['pilot']]
    metadata = validation_metadata(pilot, samples / 'samples_manifest.csv')
    out = args.output_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(out.parent).free < 6 * 1024**3:
        raise RuntimeError('6 GiB free required; no automatic cleanup')
    if not args.resume:
        out.mkdir(exist_ok=False)
        shutil.copy2(args.control_study / 'initial.pt', out / 'initial.pt')
    if digest(out / 'initial.pt') != audit['initial_sha256']:
        raise ValueError('Bridge and reused control initial weights differ')
    code = ['finetuning/run_public_source_bridge.py', 'finetuning/evaluate_public_source_bridge.py',
            'finetuning/control_reuse.py', *control['code_sha256']]
    protocol = {'kind': 'public_strain_rate_source_bridge_v1', 'source_label': 'porotomo',
                'smoke_only': args.smoke, 'control_audit': audit,
                'source_sha256': audit['initial_sha256'], 'offshore_sha256': control['offshore_sha256'],
                'manifest_sha256': control['manifest_sha256'], 'samples_dir': str(samples.resolve()),
                'source_manifest_sha256': digest(manifest), 'source_preparation_sha256': digest(args.source_preparation / 'protocol.json'),
                'source_updates': source_budget, 'final_febus_updates': target_budget, 'block_size': block,
                'seed': seed, 'learning_rate': 1e-4,
                'optimizer': control['optimizer'], 'source_plan': source_plan,
                'final_febus_plan': control['final_febus_plan'], 'pilot': control['pilot'],
                'source_input_sha256': source_hashes, 'target_input_sha256': control['input_sha256'],
                'code_sha256': {name: digest(ROOT / name) for name in sorted(set(code))},
                'vendor_sha256': control['vendor_sha256'],
                'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'torch': str(torch.__version__),
                'selection': control['selection'], 'extension': 'None',
                'guards': '>=5% target RMSE gain vs audited FEBUS-only control, complete bytes nonincreasing, unchanged per-configuration physical guards',
                'source_product': 'PoroTomo native waveform proportional to strain-rate; source windows 1s, target windows 50ms; no absolute calibration equivalence',
                'source_groups': 'Four directory-day strata, not fiber families or independent validation acquisitions',
                'exposure': 'Reused control and challenger have matched additional updates and identical final FEBUS batch sequence; not matched FLOPs/source shapes',
                'final_holdout_accessed': False}
    if protocol['gpu_uuid'] != control['gpu_uuid'] or protocol['torch'] != control['torch']:
        raise ValueError('Control reuse requires the same GPU/software context')
    if args.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Public-source resume protocol changed')
        if (out / 'COMPLETE').exists():
            return
    else:
        write_json(out / 'protocol.json', protocol)
        write_json(out / 'REUSED_CONTROL.json', audit)
        for name in protocol['code_sha256']:
            target = out / 'code_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, target)
    initial = torch.load(out / 'initial.pt', map_location='cpu', weights_only=False)
    torch.manual_seed(seed)
    model = build_model(initial['config'], initial, 'cuda')
    load = partial(load_sample, normalization='robust-escape-v1')
    masks = partial(load_valid_mask, normalization='robust-escape-v1')

    def evaluate():
        result = evaluate_detailed(model, pilot, metadata, torch.device('cuda'), load,
            load_valid_mask=masks, fidelity_transform=partial(restore_for_validation,
                policy='robust-escape-v1', model=model, detailed_channels=True), spatial_weighting=True)[0]
        if not np.isfinite(result['macro_relative_rmse']):
            raise RuntimeError('Nonfinite target selection metric')
        return result

    folder = out / 'porotomo'; folder.mkdir(exist_ok=True)
    last = folder / 'last.pt'
    saved = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
    if saved:
        if saved['bridge_protocol_sha256'] != digest(out / 'protocol.json'):
            raise ValueError('Retained bridge lineage differs')
        model.load_state_dict(saved['model_state_dict'])
        best, best_name, history = saved['best_rmse'], saved['best_name'], saved['history']
    else:
        best, best_name, history = evaluate()['macro_relative_rmse'], 'best_initial.pt', []
        shutil.copy2(out / 'initial.pt', folder / best_name)
    try:
        for phase, budget in [('source', source_budget), ('febus', target_budget)]:
            if saved and saved['phase'] == 'febus' and phase == 'source':
                continue
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            if saved and saved['phase'] == phase:
                load_independent_optimizer_state(optimizer, saved['optimizer_state_dict'])
                restore_rng(saved['rng_state']); completed = saved['phase_updates']
            else:
                stage_seed = seed if phase == 'source' else seed + 1
                torch.manual_seed(stage_seed); random.seed(stage_seed); np.random.seed(stage_seed)
                completed = 0

            def payload():
                return {'config': initial['config'], 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'phase': phase, 'phase_updates': completed,
                        'rng_state': capture_rng(), 'history': history, 'best_rmse': best, 'best_name': best_name,
                        'bridge_protocol_sha256': digest(out / 'protocol.json')}

            save_checkpoint(payload(), last)
            for start in range(completed, budget, block):
                if any(digest(ROOT / name) != value for name, value in protocol['code_sha256'].items()):
                    raise ValueError('Frozen bridge code changed')
                model.train(); started = time.perf_counter(); losses, perplexities = [], []
                for step in range(start, start + block):
                    if phase == 'source':
                        name = source_plan[step]; path = args.source_dir / name
                        if digest(path) != source_hashes[name]:
                            raise ValueError('Public source changed')
                        with np.load(path, allow_pickle=False) as values:
                            x = torch.from_numpy(values['values'])[None, None].cuda()
                        valid = torch.ones_like(x, dtype=torch.bool)
                    else:
                        path = final_plan[step]
                        if digest(path) != control['input_sha256'][str(path.relative_to(ROOT))]:
                            raise ValueError('Final FEBUS batch changed')
                        x = torch.from_numpy(load(path))[None, None].cuda()
                        valid = torch.from_numpy(masks(path))[None, None].cuda()
                    optimizer.zero_grad(set_to_none=True)
                    recon, vq, _, perplexity = model(x)
                    loss = .5 * F.mse_loss(recon[valid], x[valid]) + .5 * F.l1_loss(recon[valid], x[valid]) + vq
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite source-bridge loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); losses.append(float(loss.detach())); perplexities.append(float(perplexity.detach()))
                completed = start + block
                if optimizer_steps(optimizer) != [float(completed)]:
                    raise ValueError('Public bridge optimizer reset/counter mismatch')
                seconds = time.perf_counter() - started
                metrics = evaluate() if phase == 'febus' or completed == budget else {}
                if metrics and metrics['macro_relative_rmse'] < best:
                    best, best_name = metrics['macro_relative_rmse'], f'best_{phase}_{completed}.pt'
                    save_checkpoint({'config': initial['config'], 'model_state_dict': model.state_dict(),
                                     'phase': phase, 'updates': completed}, folder / best_name)
                history.append({'phase': phase, 'updates': completed, 'train_seconds': seconds,
                                'train_loss': float(np.mean(losses)), 'mean_perplexity': float(np.mean(perplexities)), **metrics})
                save_checkpoint(payload(), last)
                write_json(folder / 'history.json', clean_json(history))
                write_json(out / 'STATUS.json', {'stage': phase, 'stage_updates': completed, 'budget': budget,
                                                'smoke_only': args.smoke, 'control_retrained': False})
                print(f"porotomo/{phase} {completed}/{budget} train_s={seconds:.1f} pilot={metrics.get('macro_relative_rmse')}", flush=True)
            if phase == 'source':
                save_checkpoint({'config': initial['config'], 'model_state_dict': model.state_dict(),
                                 'phase': phase, 'updates': completed}, folder / 'source_endpoint.pt')
            saved = None
        shutil.copy2(folder / best_name, folder / 'selected.pt')
        write_json(folder / 'summary.json', {'source_updates': source_budget, 'final_febus_updates': target_budget,
            'final_optimizer_steps': optimizer_steps(optimizer), 'best_pilot_rmse': best, 'best_name': best_name,
            'selected_sha256': digest(folder / 'selected.pt'), 'endpoint_sha256': digest(last)})
        write_json(out / 'STATUS.json', {'stage': 'TRAINING_COMPLETE', 'control_retrained': False,
                                        'smoke_only': args.smoke})
        (out / 'TRAINING_COMPLETE').touch()
    finally:
        lock.close()


if __name__ == '__main__':
    main()
