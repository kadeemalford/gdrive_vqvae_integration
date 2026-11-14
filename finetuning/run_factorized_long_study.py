#!/usr/bin/env python3
"""Gated 8k pilot -> 65k paired full-model training for flat versus product VQ."""
import argparse
import fcntl
from functools import partial
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
from finetuning.codec_wire_evaluation import evaluate_codec
from finetuning.factorized_vq import VARIANTS, initialize_factorized, build_factorized, long_training_gate, training_check_metadata
from finetuning.finetune_vqvae import load_sample, load_valid_mask, save_checkpoint
from finetuning.paired_search_diagnostics import finite_json
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import evaluate_detailed
from finetuning.representation_study import digest, read_manifest, pilot_rows, load_packet, write_json
from finetuning.run_alphabet_factorization import lloyd
from finetuning.run_lr_schedule_study import capture_rng, restore_rng, clean_json
from finetuning.run_spatial_output_study import split_train_files
from finetuning.update_schedule import load_independent_optimizer_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-protocol', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('pilot', 'long'), default='pilot')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one remote RTX4090 required; no local experiments')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(args.repo_root.resolve()))
    out = args.output_dir; out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(out.parent).free < 12 * 1024**3:
        raise RuntimeError('12 GiB headroom required')
    reference = json.loads(args.source_protocol.read_text())
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    if (source['protocol_sha256'] != digest(args.source_protocol) or source.get('epoch') != 24
            or source.get('initialization', {}).get('arm') != 'A'
            or digest(args.samples_dir / 'samples_manifest.csv') != reference['metadata_sha256']['samples_manifest.csv']):
        raise ValueError('Expected original fold0 transferred A epoch24/corpus')
    seed = 2110
    rows = read_manifest(args.samples_dir)
    selected = pilot_rows(rows['train'], seed=seed)
    fit_ids, check_ids = split_train_files(selected, seed=seed)
    fitting = [selected[i] for i in fit_ids]; checks = [selected[i] for i in check_ids]
    held_files = {row['drive_id'] for row in checks}
    fit = [row for row in rows['train'] if row['drive_id'] not in held_files]
    rng = random.Random(seed)
    pilot_budget, maximum, block_size = (2, 2, 1) if args.smoke else (8192, 65536, 2048)
    sequence = [rng.choice(fit)['sample_path'] for _ in range(maximum)]
    calibration = []
    for family in ('doped', 'multicore', 'multimode', 'single_mode'):
        candidates = sorted((row for row in fit if row['family'] == family), key=lambda row: row['sample_path'])
        calibration += rng.sample(candidates, 8)
    if args.smoke:
        maximum_row = max(fit, key=lambda row: int(row['channels']))
        sequence[0] = maximum_row['sample_path']
        fitting = fitting[:4] + [maximum_row]
        checks = [max(checks, key=lambda row: int(row['channels']))]
        calibration = [maximum_row]
    used_names = sorted(set(sequence + [row['sample_path'] for row in fitting + checks + calibration]))
    code = ['finetuning/run_factorized_long_study.py', 'finetuning/evaluate_factorized_long_study.py',
            'finetuning/factorized_vq.py', 'finetuning/codec_wire_evaluation.py',
            'finetuning/run_alphabet_factorization.py', 'finetuning/run_spatial_output_study.py',
            'finetuning/run_lr_schedule_study.py', 'finetuning/run_index_search.py',
            'finetuning/model_factory.py', 'finetuning/latent_geometry.py',
            'finetuning/finetune_vqvae.py', 'finetuning/reconstruction_objective.py',
            'finetuning/physical_validation.py', 'finetuning/representation_study.py',
            'finetuning/robust_escape.py', 'finetuning/reversible_normalization.py',
            'finetuning/update_schedule.py', 'compression/alphabet_stream.py', 'compression/study_bundle.py',
            'finetuning/evaluate_lr_schedule_study.py', 'finetuning/paired_search_diagnostics.py']
    protocol = {'kind': 'joint_factorized_vq_long_study_v1', 'seed': seed, 'smoke_only': args.smoke,
        'source_sha256': digest(args.source), 'source_protocol_sha256': digest(args.source_protocol),
        'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'), 'samples_dir': str(args.samples_dir.resolve()),
        'variants': list(VARIANTS), 'pilot_updates': pilot_budget, 'maximum_updates': maximum, 'block_size': block_size,
        'sequence': sequence, 'checks_train_only': [row['sample_path'] for row in checks],
        'fitting_train_only': [row['sample_path'] for row in fitting], 'calibration_train_only': [row['sample_path'] for row in calibration],
        'input_sha256': {name: digest(ROOT / name) for name in used_names},
        'code_sha256': {name: digest(ROOT / name) for name in code},
        'vendor_sha256': {name: digest(args.repo_root / 'vqvae' / name) for name in ('train_vqvae.py', 'evaluate_vqvae.py')},
        'representation': 'Flat2048 versus Cartesian32x64, both2048 composite IDs/11 nominal bits, same spatial-4x grid. Product books train jointly with entire encoder/decoder.',
        'product_initialization': 'Up to256 source encoder vectors per fitting window; full128-dimensional orthogonal balanced PCA, no whitening/truncation; deterministic12-iteration Lloyd separately for32/64 books. No validation features.',
        'optimizer': 'Fresh AdamW in BOTH arms lr1e-4 wd1e-5 clip1; all model parameters train, no selective source moments inherited',
        'objective': 'Historical0.5MSE+0.5L1+VQ; original ordinary-cell mask, no filtering or new data',
        'selection': 'Minimum train-file check macro physical relative RMSE, initial candidate eligible; original source historically saw these files',
        'long_gate': 'Product improves>=1% vs own initialization and>=0.2% in final pilot block, within3% of flat endpoint RMSE, selected real frame bytes<=105% of flat selected, mean perplexity>=16. Screening gate only, not codec acceptance.',
        'final_gate': '>=5% lower physical macro relative RMSE than matched flat, complete bytes nonincreasing and unchanged configuration guards; further independent-cohort confirmation required',
        'phase_transition': 'Pilot stops after8192 each. Long phase must be separately launched after gate audit; same optimizer/RNG/sequence continues to65536 each, without further extensions.',
        'limitations': ['Factorized book has fewer free centroid parameters; this is a representation contrast, not capacity equivalence',
                       'The flat dictionary is already adapted; product initialization is fitted anew from training features',
                       'Training-file checks have been seen by original source and are not independent acquisitions',
                       'Nominal11-bit parity is not equality of measured entropy or complete bytes'],
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'torch': str(torch.__version__),
        'final_holdout_accessed': False}
    continuing = args.resume or args.phase == 'long'
    if continuing:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Factorized study resume source/inputs/code changed')
    else:
        out.mkdir(exist_ok=False); write_json(out / 'protocol.json', protocol)
        for name in code:
            dest = out / 'code_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    if args.phase == 'pilot' and (out / 'PILOT_COMPLETE').exists():
        return
    if args.phase == 'long':
        gate = json.loads((out / 'LONG_GATE.json').read_text())
        if (args.smoke or not gate['eligible'] or not long_training_gate(**gate['evidence'])
                or gate['study_protocol_sha256'] != digest(out / 'protocol.json')
                or not (out / 'PILOT_COMPLETE').exists()):
            raise ValueError('Long training is not authorized by the scientific pilot gate')
        if (out / 'LONG_TRAINING_COMPLETE').exists():
            return
    assets_path = out / 'product_assets.pt'
    if not assets_path.exists():
        torch.manual_seed(seed)
        teacher, _ = initialize_factorized(source, 'flat', 'cuda')
        teacher.eval().requires_grad_(False); pool = []
        generator = np.random.default_rng(seed)
        with torch.no_grad():
            for row in fitting:
                if digest(ROOT / row['sample_path']) != protocol['input_sha256'][row['sample_path']]:
                    raise ValueError('Fitting sample changed')
                _, packet = load_packet(row); x = torch.from_numpy(packet.values)[None, None].cuda()
                h, w = packet.values.shape
                z = teacher.encoder(F.pad(x, (0, (-w) % 16, 0, (-h) % 16)))
                flat = z.permute(0, 2, 3, 1).reshape(-1, 128).cpu()
                ids = generator.choice(len(flat), min(256, len(flat)), replace=False)
                pool.append(flat[torch.from_numpy(ids)])
        values = torch.cat(pool).double()
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite fitting vectors')
        center = values.mean(dim=0); centered = values - center
        covariance = centered.T @ centered / max(1, len(values) - 1)
        _, basis = torch.linalg.eigh(covariance)
        basis = basis.flip(1)
        for column in range(128):
            if basis[basis[:, column].abs().argmax(), column] < 0:
                basis[:, column] *= -1
        order = list(range(0, 128, 2)) + list(range(1, 128, 2))
        basis = basis[:, order].float(); center = center.float()
        rotated = (values.float() - center) @ basis
        assets = {'left': lloyd(rotated[:, :64], 32, seed=seed),
                  'right': lloyd(rotated[:, 64:], 64, seed=seed), 'basis': basis, 'center': center}
        save_checkpoint(assets, assets_path)
        write_json(out / 'preparation.json', {'assets_sha256': digest(assets_path), 'fitting_vectors': len(values),
            'fitting_windows': len(fitting), 'basis_dimensions_retained': 128, 'parameter_updates_to_teacher': 0})
        del teacher, pool, values, rotated
        torch.cuda.empty_cache()
    prep = json.loads((out / 'preparation.json').read_text())
    if digest(assets_path) != prep['assets_sha256']:
        raise ValueError('Prepared quantizer assets changed')
    assets = torch.load(assets_path, map_location='cpu', weights_only=False)
    check_paths = [ROOT / row['sample_path'] for row in checks]
    metadata = training_check_metadata(checks, ROOT)
    load = partial(load_sample, normalization='robust-escape-v1'); masks = partial(load_valid_mask, normalization='robust-escape-v1')
    target = pilot_budget if args.phase == 'pilot' else maximum
    for variant in VARIANTS:
        torch.manual_seed(seed)
        model, config = initialize_factorized(source, variant, 'cuda', assets)
        model.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        folder = out / variant; folder.mkdir(exist_ok=True); last = folder / 'last.pt'

        def evaluate():
            return evaluate_detailed(model, check_paths, metadata, torch.device('cuda'), load,
                load_valid_mask=masks, fidelity_transform=partial(restore_for_validation,
                    policy='robust-escape-v1', model=model, detailed_channels=True), spatial_weighting=True)[0]

        if last.exists():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            if saved['study_protocol_sha256'] != digest(out / 'protocol.json'):
                raise ValueError('Branch lineage changed')
            model.load_state_dict(saved['model_state_dict']); load_independent_optimizer_state(optimizer, saved['optimizer_state_dict'])
            restore_rng(saved['rng_state']); completed, best, history = saved['updates'], saved['best_rmse'], saved['history']
            best_step = saved['best_step']; del saved
            if {float(v['step']) for v in optimizer.state.values()} != {float(completed)}:
                raise ValueError('Restored optimizer age differs')
        else:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            initial = evaluate(); best, best_step, completed = initial['macro_relative_rmse'], 0, 0
            if not np.isfinite(best):
                raise RuntimeError('Nonfinite initial physical RMSE')
            history = [{'updates': 0, **initial}]
            save_checkpoint({'config': config, 'model_state_dict': model.state_dict()}, folder / 'best.pt')
        for start in range(completed, target, block_size):
            if any(digest(ROOT / name) != expected for name, expected in protocol['code_sha256'].items()):
                raise ValueError('Frozen study code changed')
            model.train(); began = time.perf_counter(); losses, perplexities = [], []
            for name in sequence[start:start + block_size]:
                path = ROOT / name
                if digest(path) != protocol['input_sha256'][name]:
                    raise ValueError('Training sample changed')
                x = torch.from_numpy(load(path))[None, None].cuda(); valid = torch.from_numpy(masks(path))[None, None].cuda()
                optimizer.zero_grad(set_to_none=True)
                predicted, vq, _, perplexity = model(x)
                loss = .5 * F.mse_loss(predicted[valid], x[valid]) + .5 * F.l1_loss(predicted[valid], x[valid]) + vq
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite factorized loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); completed += 1
                losses.append(float(loss.detach())); perplexities.append(float(perplexity.detach()))
            seconds = time.perf_counter() - began
            if ({float(v['step']) for v in optimizer.state.values()} != {float(completed)}
                    or len(optimizer.state) != sum(1 for _ in model.parameters())):
                raise ValueError('Not all parameters received the matched optimizer dose')
            metrics = evaluate()
            if not np.isfinite(metrics['macro_relative_rmse']):
                raise RuntimeError('Nonfinite physical selection RMSE')
            if metrics['macro_relative_rmse'] < best:
                best, best_step = metrics['macro_relative_rmse'], completed
                save_checkpoint({'config': config, 'model_state_dict': model.state_dict()}, folder / 'best.pt')
            history.append({'updates': completed, 'training_seconds': seconds, 'train_loss': float(np.mean(losses)),
                            'train_perplexity': float(np.mean(perplexities)), **metrics})
            save_checkpoint({'config': config, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': capture_rng(), 'updates': completed, 'best_rmse': best, 'best_step': best_step,
                'history': history, 'study_protocol_sha256': digest(out / 'protocol.json')}, last)
            write_json(folder / 'history.json', clean_json(history))
            write_json(out / 'STATUS.json', {'stage': args.phase, 'variant': variant, 'updates': completed, 'target': target})
            print(f"factorized-{args.phase} {variant} {completed}/{target} RMSE={metrics['macro_relative_rmse']:.6f} train_s={seconds:.1f}", flush=True)
        selected_name = 'selected_pilot.pt' if args.phase == 'pilot' else 'selected.pt'
        if variant == 'product':
            for name in ('basis', 'center'):
                if not torch.equal(getattr(model.quantizer.embedding, name).detach().cpu(), assets[name]):
                    raise AssertionError('Fixed product transform changed')
        shutil.copy2(folder / 'best.pt', folder / selected_name)
        write_json(folder / f'{args.phase}_summary.json', {'updates': completed, 'best_step': best_step, 'best_rmse': best,
            'selected_sha256': digest(folder / selected_name), 'endpoint_sha256': digest(last)})
        del model, optimizer
        torch.cuda.empty_cache()
    if digest(args.source) != protocol['source_sha256']:
        raise ValueError('Original source changed')
    if args.phase == 'pilot':
        wire = out / 'pilot_wire'; wire.mkdir(exist_ok=True)
        wire_protocol = {'study_protocol_sha256': digest(out / 'protocol.json'),
            'selected_sha256': {variant: digest(out / variant / 'selected_pilot.pt') for variant in VARIANTS}}
        if (wire / 'protocol.json').exists():
            if json.loads((wire / 'protocol.json').read_text()) != wire_protocol:
                raise ValueError('Pilot replay checkpoints changed')
        else:
            write_json(wire / 'protocol.json', wire_protocol)
        wire_results = {}
        for variant in VARIANTS:
            result_path = wire / f'{variant}.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                result, _ = evaluate_codec(out / variant / 'selected_pilot.pt', wire / variant, calibration, checks,
                    args.samples_dir, protocol['input_sha256'], build_factorized,
                    {'version': 'factorized-vq-study-v1', 'variant': variant, 'composite_states': 2048,
                     'radices': [2048] if variant == 'flat' else [32, 64], 'nominal_bits': 11})
            wire_results[variant] = result
            write_json(wire / f'{variant}.json', finite_json(result))
        product_history = json.loads((out / 'product/history.json').read_text())
        flat_history = json.loads((out / 'flat/history.json').read_text())
        evidence = {'initial_product': product_history[0]['macro_relative_rmse'],
            'previous_product': product_history[-2]['macro_relative_rmse'], 'final_product': product_history[-1]['macro_relative_rmse'],
            'flat_rmse': flat_history[-1]['macro_relative_rmse'],
            'product_bytes': wire_results['product']['frame_bytes'], 'flat_bytes': wire_results['flat']['frame_bytes'],
            'perplexity': product_history[-1]['train_perplexity']}
        write_json(out / 'LONG_GATE.json', {'eligible': not args.smoke and long_training_gate(**evidence),
            'study_protocol_sha256': digest(out / 'protocol.json'),
            'evidence': evidence, 'pilot_wire': finite_json(wire_results), 'smoke_only': args.smoke,
            'meaning': 'Screening only, not acceptance or measured validation superiority'})
        (out / 'PILOT_COMPLETE').touch()
        write_json(out / 'STATUS.json', {'stage': 'PILOT_COMPLETE', 'long_eligible': not args.smoke and long_training_gate(**evidence)})
        print('LONG_GATE', json.dumps({'eligible': not args.smoke and long_training_gate(**evidence), **evidence}), flush=True)
    else:
        (out / 'LONG_TRAINING_COMPLETE').touch()
        write_json(out / 'STATUS.json', {'stage': 'LONG_TRAINING_COMPLETE', 'updates_per_arm': maximum})
    lock.close()


if __name__ == '__main__':
    main()
