#!/usr/bin/env python3
"""Bounded local GPU study of decoder-available spatial output correction."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.spatial_correction_stream import pack_frame, decode_frame
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json, guarded_admit
from finetuning.representation_study import digest, distortion, load_packet, read_manifest, score, write_json
from finetuning.run_index_search import FrozenIndexStream, vectors
from finetuning.spatial_output_correction import SpatialOutputCorrection


def split_train_files(rows, seed=2084):
    grouped = defaultdict(set)
    for row in rows:
        if row['split'] != 'train':
            raise ValueError('Internal split must use train files only')
        grouped[row['family'], row['configuration']].add(row['drive_id'])
    rng, holdout = random.Random(seed), set()
    for key, ids in sorted(grouped.items()):
        files = sorted(ids)
        if len(files) < 2:
            raise ValueError('At least two files per configuration required')
        rng.shuffle(files)
        holdout.update(files[:max(1, len(files) // 5)])
    fit = [i for i, row in enumerate(rows) if row['drive_id'] not in holdout]
    check = [i for i, row in enumerate(rows) if row['drive_id'] in holdout]
    assert not ({rows[i]['drive_id'] for i in fit} & {rows[i]['drive_id'] for i in check})
    return fit, check


def macro_loss(rows):
    groups = defaultdict(list)
    for family, configuration, value in rows:
        groups[family, configuration].append(value)
    families = defaultdict(list)
    for (family, _), values in groups.items():
        families[family].append(float(np.mean(values)))
    return float(np.mean([np.mean(v) for v in families.values()]))


def channel_audit(target, prediction, valid):
    target, prediction = target.astype(np.float64), prediction.astype(np.float64)
    count = valid.sum(axis=1)
    mean = np.where(valid, target, 0).sum(axis=1) / np.maximum(count, 1)
    variance = np.where(valid, (target - mean[:, None]) ** 2, 0).sum(axis=1) / np.maximum(count, 1)
    error = np.where(valid, (target - prediction) ** 2, 0).sum(axis=1) / np.maximum(count, 1)
    std = np.sqrt(variance)
    floor = max(float(np.max(std)) * 1e-3, 1e-12)
    weak = std < floor
    boundary = (np.arange(len(std)) < 8) | (np.arange(len(std)) >= len(std) - 8)
    result = {'std_floor': floor, 'channels': len(std), 'weak_channels': int(weak.sum())}
    for name, mask in [('edge_weak', boundary & weak), ('edge_nonweak', boundary & ~weak),
                       ('interior_weak', ~boundary & weak), ('interior_nonweak', ~boundary & ~weak)]:
        eligible = mask & (count > 0)
        ep = float(np.sum(error[eligible] * count[eligible]))
        tp = float(np.sum(variance[eligible] * count[eligible]))
        n = int(count[eligible].sum())
        result[name] = {'channels': int(eligible.sum()), 'cells': n, 'error_power': ep,
                       'target_centered_power': tp, 'mse': ep / n if n else None,
                       'relative_rmse': float(np.sqrt(ep / tp)) if tp > 1e-20 else None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This training study requires the local CUDA GPU')
    if '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('This campaign is restricted to the local RTX 4060')
    if args.steps != 1024 and not args.smoke:
        raise ValueError('Production budget is fixed at 1024 updates per arm')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(2084)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_hash = digest(args.source)
    paired = json.loads((args.paired_dir / 'protocol.json').read_text())
    assert (args.paired_dir / 'COMPLETE').exists()
    assert source_hash == paired['checkpoint_sha256']['source']
    assert digest(args.calibration_dir / 'counts.npy') == paired['counts_sha256']
    assert digest(args.samples_dir / 'samples_manifest.csv') == paired['manifest_sha256']
    splits = read_manifest(args.samples_dir)
    lookup = {r['sample_path']: r for values in splits.values() for r in values}
    selected = {s: [lookup[p] for p in names] for s, names in paired['selection'].items()}
    if args.smoke:
        # Preserve file-level stratification even in smoke; use a small update budget.
        args.steps = min(args.steps, 8)
    fit, check = split_train_files(selected['train'])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    paths = ['finetuning/run_spatial_output_study.py', 'finetuning/spatial_output_correction.py',
             'compression/spatial_correction_stream.py', 'finetuning/paired_search_diagnostics.py',
             'finetuning/representation_study.py', 'finetuning/model_factory.py']
    protocol = {'kind': 'decoder_available_spatial_correction_v1', 'seed': 2084,
        'gpu': torch.cuda.get_device_name(), 'device': 'cuda', 'smoke_only': args.smoke,
        'torch_version': str(torch.__version__), 'source_sha256': source_hash,
        'math_policy': 'FP32, deterministic cuDNN, TF32 disabled for matmul and cuDNN',
        'paired_protocol_sha256': digest(args.paired_dir / 'protocol.json'),
        'code_sha256': {p: digest(ROOT / p) for p in paths},
        'fit': [selected['train'][i]['sample_path'] for i in fit],
        'internal_holdout': [selected['train'][i]['sample_path'] for i in check],
        'validation': [r['sample_path'] for r in selected['val']],
        'steps_per_arm': args.steps, 'arms': ['pointwise', 'spatial'], 'hidden': 16,
        'initialization': 'identical weights; residual last layer zero; source exactly eligible',
        'optimizer': 'Adam lr=0.001; batch=1; clip_grad_norm=1; FP32',
        'objective': '0.5 ordinary normalized MSE + 0.5 ordinary normalized L1; no relative denominator in training',
        'selection': 'lowest strict-source-guarded macro normalized distortion on file-disjoint internal TRAIN holdout; checkpoints at 0,256,512,1024; validation only after both arms selected',
        'extension': 'none; 1024 updates per arm, no automatic continuation',
        'investment_gate': '>=5% guarded macro RMSE reduction vs matched pointwise control at equal framed payload, with concurrent metrics and deployment weight costs disclosed; otherwise no scale-up',
        'decoded_features': 'five decoded spatial taps plus clipped distances to each spatial edge; pointwise repeats center five times; same nominal network parameter budget, not equal effective feature rank',
        'side_contract': 'source indices/side unchanged; FSC1 envelope 13 bytes per frame for every arm including source; correction weights and registry separately counted; fallback mode is explicit',
        'test_access': False, 'external_holdout_used': False}
    write_json(out / 'protocol.json', protocol)
    for path in paths:
        dest = out / 'code_snapshot' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
    stream = FrozenIndexStream(np.load(args.calibration_dir / 'counts.npy', allow_pickle=False), source_hash)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    base = build_model({**saved['config'], 'exact_eval_quantization': True}, saved, 'cuda').eval().requires_grad_(False)
    del saved
    cache, audits = {}, []
    cache_dir = out / 'cache'
    cache_dir.mkdir()
    with torch.no_grad():
        for split, rows in selected.items():
            cache[split] = []
            for index, row in enumerate(rows):
                assert row['split'] == split
                assert digest(ROOT / row['sample_path']) == paired['sample_sha256'][row['sample_path']]
                raw, packet = load_packet(row)
                frame = (args.paired_dir / split / f'{index:05d}.source.frame').read_bytes()
                indices, shape, latent, side = stream.decode(frame)
                assert shape == raw.shape and side == packet.side_stream
                pred = base.decoder(vectors(torch.from_numpy(indices.astype(np.int64)).cuda(),
                    base.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]][0, 0].cpu().numpy()
                before = finite_json(score(row, raw, packet, pred))
                np.savez_compressed(cache_dir / f'{split}_{index:05d}.npz', prediction=pred,
                                    target=packet.values, valid=packet.valid)
                cache[split].append((torch.from_numpy(pred)[None, None], torch.from_numpy(packet.values)[None, None],
                                     torch.from_numpy(packet.valid)[None, None], before, frame))
                if split == 'train':
                    audits.append({'sample_path': row['sample_path'], 'family': row['family'], 'configuration': row['configuration'],
                                   **channel_audit(packet.values, pred, packet.valid)})
    write_json(out / 'channel_audit_train.json', finite_json({'windows': audits,
        'weak_rule': 'channel std < max(0.001*max_channel_std_in_window,1e-12); diagnostic only; no channels removed'}))
    del base
    torch.cuda.empty_cache()
    rng = random.Random(2084)
    sequence = [rng.choice(fit) for _ in range(args.steps)]
    write_json(out / 'exposure.json', {'sample_paths': [selected['train'][i]['sample_path'] for i in sequence]})

    def internal_evaluate(model):
        model.eval()
        losses, accepted = [], 0
        with torch.no_grad():
            for i in check:
                row = selected['train'][i]
                x, target, valid, before, _ = cache['train'][i]
                prediction = model(x.cuda())
                raw, packet = load_packet(row)
                after = finite_json(score(row, raw, packet, prediction[0, 0].cpu().numpy()))
                eligible = guarded_admit(before, after)
                chosen = prediction if eligible else x.cuda()
                accepted += int(eligible)
                losses.append((row['family'], row['configuration'], float(distortion(chosen, target.cuda(), valid.cuda()))))
        return macro_loss(losses), accepted

    history, selections, trained = {}, {}, {}
    for arm in ('pointwise', 'spatial'):
        torch.manual_seed(2084)
        model = SpatialOutputCorrection(arm == 'spatial').cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        best_loss, accepted = internal_evaluate(model)
        best_step = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history[arm] = [{'step': 0, 'internal_guarded_distortion': best_loss, 'accepted_windows': accepted}]
        start = time.perf_counter()
        for step, index in enumerate(sequence, 1):
            x, target, valid, _, _ = cache['train'][index]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = distortion(model(x.cuda()), target.cuda(), valid.cuda())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite correction training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step in (256, 512, 1024, args.steps):
                value, accepted = internal_evaluate(model)
                history[arm].append({'step': step, 'internal_guarded_distortion': value, 'accepted_windows': accepted,
                                     'last_training_loss': float(loss.detach())})
                if value < best_loss:
                    best_loss, best_step = value, step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                            'spatial': model.spatial, 'hidden': model.hidden, 'source_sha256': source_hash}, out / f'{arm}_last.pt')
                write_json(out / 'history.json', history)
                print(json.dumps({'arm': arm, 'step': step, 'internal_distortion': value, 'best_step': best_step}), flush=True)
        model.load_state_dict(best_state)
        model.eval()
        checkpoint = {'state_dict': best_state, 'step': best_step, 'spatial': model.spatial,
                      'hidden': model.hidden, 'source_sha256': source_hash}
        torch.save(checkpoint, out / f'{arm}_selected.pt')
        selections[arm] = {'selected_step': best_step, 'completed_updates': args.steps,
                           'internal_guarded_distortion': best_loss, 'seconds': time.perf_counter() - start,
                           'parameters': sum(p.numel() for p in model.parameters()),
                           'weight_bytes': (out / f'{arm}_selected.pt').stat().st_size}
        trained[arm] = model
    write_json(out / 'selection.json', selections)
    records, scored = [], defaultdict(list)
    for index, row in enumerate(selected['val']):
        x, _, _, before, payload = cache['val'][index]
        raw, packet = load_packet(row)
        before = finite_json(score(row, raw, packet, x[0, 0].numpy()))
        record = {'sample_path': row['sample_path'], 'drive_id': row['drive_id'], 'source': before}
        scored['source'].append(before)
        for arm, model in trained.items():
            with torch.no_grad():
                prediction = model(x.cuda())[0, 0].cpu().numpy()
            after = finite_json(score(row, raw, packet, prediction))
            eligible = guarded_admit(before, after)
            record[arm] = after
            record[arm + '_guarded'] = after if eligible else before
            record[arm + '_mode'] = eligible
            scored[arm].append(after)
            scored[arm + '_guarded'].append(record[arm + '_guarded'])
            folder = out / arm
            folder.mkdir(exist_ok=True)
            (folder / f'{index:05d}.fsc').write_bytes(pack_frame(payload, eligible))
        folder = out / 'source'
        folder.mkdir(exist_ok=True)
        (folder / f'{index:05d}.fsc').write_bytes(pack_frame(payload, False))
        records.append(record)
    write_json(out / 'validation_records.json', records)
    np.save(out / 'counts.npy', np.load(args.calibration_dir / 'counts.npy', allow_pickle=False))
    maps = [json.loads(line) for line in (args.samples_dir / 'channel_maps.jsonl').read_text().splitlines()]
    ids = {r['channel_map_id'] for r in selected['val']}
    write_json(out / 'maps.json', [m for m in maps if m['channel_map_id'] in ids])
    write_json(out / 'transport.json', [{'frame': f'{i:05d}.fsc', 'sample_path': r['sample_path'],
        'drive_id': r['drive_id'], 'block_index': int(r['block_index']), 'time_start': int(r['time_start']),
        'channel_map_id': r['channel_map_id']} for i, r in enumerate(selected['val'])])
    common = sum((out / name).stat().st_size for name in ('counts.npy', 'maps.json', 'transport.json'))
    results = {'gpu': torch.cuda.get_device_name(), 'parameter_updates': args.steps * 2,
               'selection': selections, 'validation_windows': len(records), 'arms': {}, 'common_auxiliary_bytes': common,
               'source_frame_bytes_unwrapped': sum(len(item[4]) for item in cache['val']),
               'limitations': ['development only; selected on internal train-file holdout, not validation',
                   'base decoder/codebook weights shared and excluded; correction weights counted separately',
                   'per-window fallback protects RMSE/correlation/p90, not universal physical fidelity',
                   'cached source outputs, not a test of end-to-end representation or higher-rate residual coding',
                   'internal train-file split does not guarantee acquisition independence']}
    for arm, rows in scored.items():
        macro, groups = aggregate_metrics(rows)
        results['arms'][arm] = {'macro': macro, 'groups': groups}
    source_rmse = results['arms']['source']['macro']['macro_relative_rmse']
    for arm in ('source', 'pointwise', 'spatial'):
        folder = out / arm
        registry = {'format': 'FSC1', 'source_sha256': source_hash,
                    'counts_sha256': digest(out / 'counts.npy'), 'correction_sha256': None if arm == 'source' else digest(out / f'{arm}_selected.pt')}
        write_json(folder / 'registry.json', registry)
        size = sum(p.stat().st_size for p in folder.glob('*.fsc'))
        key = arm if arm == 'source' else arm + '_guarded'
        results['arms'][key].update(frame_bytes=size, stream_bytes=size + common + (folder / 'registry.json').stat().st_size,
            correction_weights_bytes=0 if arm == 'source' else selections[arm]['weight_bytes'],
            rmse_gain_vs_source=1 - results['arms'][key]['macro']['macro_relative_rmse'] / source_rmse)
    results['spatial_gain_vs_pointwise_guarded'] = 1 - results['arms']['spatial_guarded']['macro']['macro_relative_rmse'] / results['arms']['pointwise_guarded']['macro']['macro_relative_rmse']
    results['investment_gate_met'] = results['spatial_gain_vs_pointwise_guarded'] >= .05
    assert digest(args.source) == source_hash
    # Independent CPU decode uses only shared models, frames and tables, no target input.
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    base = build_model({**saved['config'], 'exact_eval_quantization': True}, saved, 'cpu').eval().requires_grad_(False)
    replay_rows = []
    for arm in trained:
        selected_checkpoint = torch.load(out / f'{arm}_selected.pt', map_location='cpu', weights_only=False)
        assert selected_checkpoint['source_sha256'] == source_hash
        model = SpatialOutputCorrection(selected_checkpoint['spatial'], selected_checkpoint['hidden'])
        model.load_state_dict(selected_checkpoint['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        positive = [i for i, record in enumerate(records) if record[arm + '_mode']]
        replay_indices = sorted(set([0, len(records) // 2, len(records) - 1, *positive[:1]]))
        for index in replay_indices:
            restored = decode_frame((out / arm / f'{index:05d}.fsc').read_bytes(), stream, base, model, 'cpu')
            row = selected['val'][index]
            raw, packet = load_packet(row)
            assert restored.shape == raw.shape and np.isfinite(restored).all()
            np.testing.assert_array_equal(restored[~packet.valid], raw[~packet.valid])
            replay_rows.append({'arm': arm, 'index': index, 'finite': True, 'escapes_exact': True})
    results['independent_cpu_replay'] = replay_rows
    results['cpu_gpu_bit_identity_claimed'] = False
    write_json(out / 'results.json', finite_json(results))
    (out / 'COMPLETE').touch()
    print(json.dumps({'gate': results['investment_gate_met'], 'gain_vs_pointwise': results['spatial_gain_vs_pointwise_guarded'],
        'arms': {a: results['arms'][a]['macro']['macro_relative_rmse'] for a in results['arms']}}), flush=True)


if __name__ == '__main__':
    main()
