#!/usr/bin/env python3
"""Bounded warm-start grid/alphabet pilot with train-file selection and real bytes."""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from finetuning.model_factory import new_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, read_manifest, pilot_rows, load_packet, distortion, score, write_json
from finetuning.run_index_search import nearest, vectors
from finetuning.run_spatial_output_study import split_train_files, macro_loss
from finetuning.run_physical_guard_search import physical_admit

ARMS = {'compact2048': ('spatial-4x', 2048), 'compact64': ('spatial-4x', 64),
        'dense64': ('spatial4-temporal2', 64), 'dense32': ('spatial4-temporal2', 32)}


def lloyd(vectors, k, seed=2090, iterations=12):
    """Deterministic farthest-first seeds, then fixed-budget train-only Lloyd steps."""
    generator = torch.Generator(device=vectors.device).manual_seed(seed)
    first = torch.randint(len(vectors), (), generator=generator, device=vectors.device)
    centers = [vectors[first]]
    distance = torch.full((len(vectors),), float('inf'), device=vectors.device)
    for _ in range(1, k):
        distance = torch.minimum(distance, (vectors - centers[-1]).square().sum(1))
        centers.append(vectors[distance.argmax()])
    centers = torch.stack(centers)
    for _ in range(iterations):
        distances = vectors.square().sum(1, keepdim=True) - 2 * vectors @ centers.T + centers.square().sum(1)
        assigned = distances.argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, assigned, vectors)
        counts = torch.bincount(assigned, minlength=k)
        centers = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers)
    if not torch.isfinite(centers).all():
        raise RuntimeError('Nonfinite initialization')
    return centers


def atomic_save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one remote RTX 4090 via CUDA_VISIBLE_DEVICES')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed, budget = 2090, 2 if args.smoke else 1024
    splits = read_manifest(args.samples_dir)
    pilots = pilot_rows(splits['train'], seed=seed)
    _, check_indices = split_train_files(pilots, seed=seed)
    checks = [pilots[i] for i in check_indices]
    holdout = {r['drive_id'] for r in checks}
    fits = [r for r in splits['train'] if r['drive_id'] not in holdout]
    calibration = pilot_rows(fits, seed=seed)
    validation = pilot_rows(splits['val'], seed=seed)
    # All validation is already-consumed development; never call it fresh holdout.
    if args.smoke:
        checks = [max(checks, key=lambda r: int(r['channels']))]
        calibration = [max(calibration, key=lambda r: int(r['channels'])), min(calibration, key=lambda r: int(r['channels']))]
        validation = [max(validation, key=lambda r: int(r['channels']))]
    rng = random.Random(seed)
    sequence = [fits[rng.randrange(len(fits))] for _ in range(budget)]
    if args.smoke:
        sequence[0] = max(fits, key=lambda r: int(r['channels']))
    source_hash = digest(args.source)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    assert saved['config']['num_embeddings'] == 2048
    assert saved['config']['latent_geometry'] == 'spatial-4x'
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_alphabet_factorization.py', 'compression/alphabet_stream.py',
            'finetuning/model_factory.py', 'finetuning/latent_geometry.py', 'finetuning/representation_study.py',
            'finetuning/run_spatial_output_study.py', 'finetuning/paired_search_diagnostics.py',
            'finetuning/run_physical_guard_search.py', 'finetuning/run_index_search.py']
    used = {r['sample_path']: r for r in [*sequence, *checks, *calibration, *validation]}
    protocol = {'kind': 'warmstart_grid_alphabet_v1', 'seed': seed, 'smoke_only': args.smoke,
        'budget_per_arm': budget, 'arms': ARMS, 'gpu': torch.cuda.get_device_name(),
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'torch': str(torch.__version__),
        'source_sha256': source_hash, 'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'code_sha256': {f: digest(ROOT / f) for f in code},
        'vendor_sha256': {f: digest(args.repo_root / 'vqvae' / f) for f in ('train_vqvae.py', 'evaluate_vqvae.py')},
        'sequence': [r['sample_path'] for r in sequence], 'internal_train_check': [r['sample_path'] for r in checks],
        'calibration_train': [r['sample_path'] for r in calibration], 'validation': [r['sample_path'] for r in validation],
        'input_sha256': {f: digest(ROOT / f) for f in sorted(used)},
        'optimizer': 'fresh AdamW lr=1e-4 wd=1e-5 batch=1 clip=1 FP32 TF32=false activation checkpointing',
        'objective': 'same 0.5*MSE+0.5*L1+VQ, no rate or physical auxiliary',
        'initialization': 'identical source nonquantizer weights; original2048 kept, reduced alphabets train-only farthest-first+12 Lloyd steps, <=256 vectors/calibration window; no claim of matched pretraining',
        'selection': 'minimum macro normalized distortion on train-file checks at 0/256/512/1024; validation only once per selected arm',
        'wire': 'FAK1 exact fixed-bit or marginal-range fallback; selected-model tables fit on train calibration only, smoothed +1; same framing/side for every arm',
        'rate': 'dense32/dense64 bracket nominal compact2048 payload; actual shape padding/entropy/side/registry charged, not exact bitrate equality by design',
        'guard': 'diagnostic per-window fallback vs selected compact2048 with bytes and physical_admit; same metrics, no tolerances relaxed; route byte + both shared models counted separately',
        'extension': 'no automatic extension in this runner; only consider another preregistered stage after >=5% physically guarded RMSE at nonincreasing routed frame bytes and no systematic group regression',
        'limitations': ['short warm-start adaptation, not scratch/convergence comparison', 'one seed/fold, already-consumed development validation',
                        'prior source model saw train-check files in historical pretraining', 'grid changes alter context/padding and compute',
                        'no EDAS/SMF access, no final promotion or operational fidelity claim']}
    write_json(out / 'protocol.json', protocol)
    for f in code:
        dest = out / 'code_snapshot' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    for f in protocol['vendor_sha256']:
        dest = out / 'code_snapshot/vendor/vqvae' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.repo_root / 'vqvae' / f, dest)

    def infer(model, row):
        raw, packet = load_packet(row)
        target = torch.from_numpy(packet.values)[None, None].cuda()
        h, w = raw.shape
        z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
        indices = nearest(z, model.quantizer.embedding.weight)
        latent = tuple(z.shape[-2:])
        output = model.decoder(vectors(indices, model.quantizer.embedding.weight, latent))[..., :h, :w]
        return raw, packet, target, output, indices, latent

    def evaluate_checks(model):
        model.eval()
        values = []
        with torch.no_grad():
            for row in checks:
                _, packet, target, output, _, _ = infer(model, row)
                valid = torch.from_numpy(packet.valid)[None, None].cuda()
                values.append((row['family'], row['configuration'], float(distortion(output, target, valid))))
        return macro_loss(values)

    baseline, results, all_records = {}, {}, {}
    for arm, (geometry, k) in ARMS.items():
        folder = out / arm
        folder.mkdir()
        torch.manual_seed(seed)
        config = {**saved['config'], 'latent_geometry': geometry, 'num_embeddings': k,
                  'activation_checkpointing': True, 'exact_eval_quantization': True}
        model = new_model(config, 'cuda')
        state = {key: value for key, value in saved['model_state_dict'].items() if key != 'quantizer.embedding.weight'}
        incompatible = model.load_state_dict(state, strict=False)
        assert incompatible.missing_keys == ['quantizer.embedding.weight'] and not incompatible.unexpected_keys
        with torch.no_grad():
            if k == 2048:
                model.quantizer.embedding.weight.copy_(saved['model_state_dict']['quantizer.embedding.weight'])
            else:
                model.eval()
                pool = []
                for row in calibration:
                    _, packet = load_packet(row)
                    tensor = torch.from_numpy(packet.values)[None, None].cuda()
                    h, w = tensor.shape[-2:]
                    z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
                    flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
                    positions = torch.linspace(0, len(flat) - 1, min(256, len(flat)), device='cuda').long()
                    pool.append(flat[positions].clone())
                model.quantizer.embedding.weight.copy_(lloyd(torch.cat(pool), k))
                del pool, z, flat, tensor
        initial = evaluate_checks(model)
        best_loss, best_step = initial, 0
        best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        history = [{'step': 0, 'train_check_distortion': initial}]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        train_seconds = 0.
        for step, row in enumerate(sequence, 1):
            _, packet = load_packet(row)
            tensor = torch.from_numpy(packet.values)[None, None].cuda()
            valid = torch.from_numpy(packet.valid)[None, None].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            prediction, vq, _, _ = model(tensor)
            loss = distortion(prediction, tensor, valid) + vq
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - tick
            del prediction, vq, loss, tensor, valid
            if step in (256, 512, 1024, budget):
                check = evaluate_checks(model)
                history.append({'step': step, 'train_check_distortion': check, 'last_gradient_norm': float(norm)})
                if check < best_loss:
                    best_loss, best_step = check, step
                    best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                write_json(folder / 'history.json', history)
                atomic_save(folder / 'last.pt', {'model_state_dict': model.state_dict(), 'config': config,
                    'optimizer': optimizer.state_dict(), 'step': step, 'source_sha256': source_hash,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()})
                print(json.dumps({'arm': arm, 'step': step, 'train_check_distortion': check, 'best_step': best_step}), flush=True)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        del optimizer
        model.load_state_dict(best)
        model.eval().requires_grad_(False)
        atomic_save(folder / 'selected.pt', {'model_state_dict': best, 'config': config, 'step': best_step, 'source_sha256': source_hash})
        del best
        checkpoint_hash = digest(folder / 'selected.pt')
        counts = np.zeros(k, dtype=np.int64)
        with torch.no_grad():
            for row in calibration:
                _, _, _, _, indices, _ = infer(model, row)
                counts += np.bincount(indices.cpu().numpy(), minlength=k)
        np.save(folder / 'counts.npy', counts)
        stream = AlphabetStream(counts, checkpoint_hash)
        descriptor = {'format': 'FAK1', 'checkpoint_sha256': checkpoint_hash, 'counts_sha256': digest(folder / 'counts.npy'),
                      'alphabet': k, 'geometry': geometry, 'identifier': stream.identifier.hex(), 'config': config}
        write_json(folder / 'descriptor.json', descriptor)
        records, metrics, guarded_metrics, framed, guarded_bytes = [], [], [], 0, 0
        with torch.no_grad():
            for i, row in enumerate(validation):
                raw, packet, _, output, indices, latent = infer(model, row)
                array = output[0, 0].cpu().numpy()
                metric = finite_json(score(row, raw, packet, array))
                frame = stream.encode(indices.cpu().numpy(), raw.shape, latent, packet.side_stream)
                recovered, shape, ls, side = stream.decode(frame)
                np.testing.assert_array_equal(recovered, indices.cpu().numpy())
                assert shape == raw.shape and ls == latent and side == packet.side_stream
                replay = model.decoder(vectors(torch.from_numpy(recovered.astype(np.int64)).cuda(), model.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]]
                torch.testing.assert_close(replay, output, rtol=0, atol=0)
                if arm == 'compact2048':
                    baseline[row['sample_path']] = metric, frame
                before, source_frame = baseline[row['sample_path']]
                eligible = len(frame) <= len(source_frame) and (arm == 'compact2048' or physical_admit(before, metric))
                selected = frame if eligible else source_frame
                metrics.append(metric)
                guarded_metrics.append(metric if eligible else before)
                framed += len(frame)
                guarded_bytes += len(selected) + 1
                (folder / f'{i:05d}.candidate.fak').write_bytes(frame)
                (folder / f'{i:05d}.guarded.route').write_bytes(bytes([int(eligible)]) + selected)
                records.append({'sample_path': row['sample_path'], 'metrics': metric, 'source': before,
                    'guarded': metric if eligible else before, 'admitted': bool(eligible),
                    'frame_bytes': len(frame), 'source_frame_bytes': len(source_frame), 'positions': len(recovered),
                    'nominal_fixed_bits': len(recovered) * stream.bits, 'latent': latent})
        macro, groups = aggregate_metrics(metrics)
        guarded_macro, guarded_groups = aggregate_metrics(guarded_metrics)
        auxiliary = (folder / 'counts.npy').stat().st_size + (folder / 'descriptor.json').stat().st_size
        weight_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
        results[arm] = {'selected_step': best_step, 'updates': budget, 'initial_train_check_distortion': initial,
            'selected_train_check_distortion': best_loss, 'train_compute_seconds': train_seconds, 'train_and_checks_seconds': elapsed,
            'peak_cuda_allocated_bytes': peak, 'waveform_tensor_bytes': weight_bytes,
            'macro': macro, 'groups': groups, 'guarded_macro': guarded_macro, 'guarded_groups': guarded_groups,
            'candidate_frame_bytes': framed, 'guarded_routed_frame_bytes': guarded_bytes,
            'shared_table_descriptor_bytes': auxiliary, 'admitted_windows': sum(r['admitted'] for r in records),
            'train_observed_codes': int((counts > 0).sum())}
        write_json(folder / 'records.json', records)
        all_records[arm] = records
        write_json(out / 'partial_results.json', finite_json(results))
        del model
        torch.cuda.empty_cache()
    transport = [{key: r[key] for key in ('sample_path', 'drive_id', 'family', 'configuration', 'spatial_region_index')} for r in validation]
    write_json(out / 'transport.json', transport)
    control = results['compact2048']
    for arm, result in results.items():
        result['guarded_rmse_gain_vs_control'] = 1 - result['guarded_macro']['macro_relative_rmse'] / control['macro']['macro_relative_rmse']
        result['raw_rmse_gain_vs_control'] = 1 - result['macro']['macro_relative_rmse'] / control['macro']['macro_relative_rmse']
        result['candidate_stream_bytes'] = result['candidate_frame_bytes'] + result['shared_table_descriptor_bytes'] + (out / 'transport.json').stat().st_size
        # Source-only route control also pays one byte per frame. Fallback needs both registries.
        result['guarded_stream_bytes'] = result['guarded_routed_frame_bytes'] + result['shared_table_descriptor_bytes'] + (control['shared_table_descriptor_bytes'] if arm != 'compact2048' else 0) + (out / 'transport.json').stat().st_size
        result['fallback_waveform_tensor_bytes'] = result['waveform_tensor_bytes'] + (control['waveform_tensor_bytes'] if arm != 'compact2048' else 0)
        result['advance_gate_met'] = (arm != 'compact2048' and result['guarded_rmse_gain_vs_control'] >= .05
            and result['guarded_stream_bytes'] <= control['guarded_stream_bytes'])
    assert digest(args.source) == source_hash
    write_json(out / 'results.json', finite_json({'parameter_updates': len(ARMS) * budget,
        'smoke_only': args.smoke, 'gpu': torch.cuda.get_device_name(), 'validation_windows': len(validation), 'arms': results}))
    (out / 'COMPLETE').touch()
    print(json.dumps({a: {'rmse': r['macro']['macro_relative_rmse'], 'bytes': r['candidate_frame_bytes'],
        'guarded_gain': r['guarded_rmse_gain_vs_control'], 'advance': r['advance_gate_met']} for a, r in results.items()}), flush=True)


if __name__ == '__main__':
    main()
