#!/usr/bin/env python3
"""Local GPU paired full-VQ coadaptation with and without a frozen-table rate term."""
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
from compression.spatial_correction_stream import pack_frame, decode_model_frame
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import finite_json, aggregate_metrics, guarded_admit
from finetuning.representation_study import digest, load_packet, read_manifest, distortion, score, write_json
from finetuning.run_index_search import FrozenIndexStream, vectors
from finetuning.run_encoder_rd_study import gradient_norm
from finetuning.run_spatial_output_study import split_train_files, macro_loss
from finetuning.vq_encoder_rd import frozen_forward


def atomic_checkpoint(path, contents):
    temp = path.with_suffix('.tmp')
    torch.save(contents, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('This training campaign requires the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(2085)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    budget = 2 if args.smoke else 1024
    paired = json.loads((args.paired_dir / 'protocol.json').read_text())
    assert (args.paired_dir / 'COMPLETE').exists()
    source_hash = digest(args.source)
    calibration = json.loads((args.calibration_dir / 'calibration.json').read_text())
    assert source_hash == paired['checkpoint_sha256']['source'] == calibration['checkpoint_sha256']
    assert digest(args.samples_dir / 'samples_manifest.csv') == paired['manifest_sha256'] == calibration['manifest_sha256']
    assert digest(args.calibration_dir / 'counts.npy') == paired['counts_sha256']
    splits = read_manifest(args.samples_dir)
    lookup = {r['sample_path']: r for rows in splits.values() for r in rows}
    pilot_train = [lookup[p] for p in paired['selection']['train']]
    _, check = split_train_files(pilot_train)
    check_rows = [pilot_train[i] for i in check]
    holdout_ids = {r['drive_id'] for r in check_rows}
    fit_rows = [r for r in splits['train'] if r['drive_id'] not in holdout_ids]
    assert not (holdout_ids & {r['drive_id'] for r in fit_rows})
    validation = [lookup[p] for p in paired['selection']['val']]
    calibration_rows = []
    for family in sorted({r['family'] for r in fit_rows}):
        rows = [r for r in fit_rows if r['family'] == family]
        calibration_rows.extend([min(rows, key=lambda r: (int(r['channels']), r['sample_path'])),
                                 max(rows, key=lambda r: (int(r['channels']), r['sample_path']))])
    rng = random.Random(2085)
    sequence = [rng.randrange(len(fit_rows)) for _ in range(budget)]
    # Smoke includes a maximal shape regardless of the production draw sequence.
    if args.smoke:
        sequence[0] = max(range(len(fit_rows)), key=lambda i: int(fit_rows[i]['channels']))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_joint_vq_rate_study.py', 'finetuning/vq_encoder_rd.py',
            'finetuning/run_spatial_output_study.py', 'finetuning/model_factory.py',
            'finetuning/paired_search_diagnostics.py', 'finetuning/run_index_search.py',
            'compression/spatial_correction_stream.py', 'finetuning/representation_study.py']
    protocol = {'kind': 'full_vq_frozen_rate_pair_v1', 'seed': 2085, 'smoke_only': args.smoke,
        'device': 'cuda', 'gpu': torch.cuda.get_device_name(), 'torch_version': str(torch.__version__),
        'source_sha256': source_hash, 'calibration_sha256': digest(args.calibration_dir / 'calibration.json'),
        'counts_sha256': digest(args.calibration_dir / 'counts.npy'), 'temperature': calibration['temperature'],
        'code_sha256': {p: digest(ROOT / p) for p in code}, 'budget_per_arm': budget,
        'train_candidates': len(fit_rows), 'internal_holdout': [r['sample_path'] for r in check_rows],
        'sequence': [fit_rows[i]['sample_path'] for i in sequence],
        'sequence_sha256': {fit_rows[i]['sample_path']: digest(ROOT / fit_rows[i]['sample_path']) for i in sequence},
        'validation': [r['sample_path'] for r in validation], 'base_parameters_frozen': False,
        'trainable_scope': 'encoder, decoder and codebook in BOTH arms; prior probabilities frozen and shared',
        'optimizer': 'fresh AdamW lr=1e-5, weight_decay=1e-5, clip=1, batch=1, FP32, activation checkpointing, TF32 disabled',
        'rate_coefficient_rule': '0.1 * median train-only ||grad(D+VQ)|| / ||grad(R)|| over full parameter scope; clipped [1e-4,10]',
        'rate_calibration_samples': [r['sample_path'] for r in calibration_rows],
        'selection': 'lowest source-byte-and-metric-guarded macro normalized distortion on internal TRAIN files at 0/256/512/1024; no validation selection',
        'wire_contract': 'FSC1 mode 0=source model/source indices, mode 1=candidate model/candidate indices; frozen alphabet tables; both waveform models required for fallback and bound by outer registry',
        'extension': 'none; no automatic LR/temperature/sampler/hyperprior sweep',
        'investment_gate': '>=5% guarded RMSE reduction vs full-VQ control at nonincreasing framed bytes, concurrent metrics inspected; waveform weight deployment cost disclosed',
        'limitations': ['train-file holdout, not independent acquisitions; development validation reused',
                        'short fixed budget, not a convergence or family impossibility test',
                        'same selected-checkpoint policy; selected step can differ between arms',
                        'weights of the two waveform models are shared deployment cost, not free']}
    write_json(out / 'protocol.json', protocol)
    for path in code:
        dest = out / 'code_snapshot' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
    stream = FrozenIndexStream(np.load(args.calibration_dir / 'counts.npy', allow_pickle=False), source_hash)
    costs = -torch.log2(torch.from_numpy(stream.probabilities).cuda())
    temperature = calibration['temperature']

    def new_model():
        saved = torch.load(args.source, map_location='cpu', weights_only=False)
        config = {**saved['config'], 'activation_checkpointing': True, 'exact_eval_quantization': True}
        model = build_model(config, saved, 'cuda')
        del saved
        return model, config

    model, config = new_model()
    model.train().requires_grad_(True)
    parameters = list(model.parameters())
    gradient_rows = []
    for row in calibration_rows:
        assert row['split'] == 'train'
        assert row['drive_id'] not in holdout_ids
        _, packet = load_packet(row)
        tensor = torch.from_numpy(packet.values)[None, None].cuda()
        valid = torch.from_numpy(packet.valid)[None, None].cuda()
        prediction, vq, rate, _, _, _ = frozen_forward(model, tensor, costs, temperature, True)
        dnorm = gradient_norm(distortion(prediction, tensor, valid) + vq, parameters, retain_graph=True)
        rnorm = gradient_norm(rate, parameters)
        if not np.isfinite([dnorm, rnorm]).all() or rnorm <= 0:
            raise RuntimeError('Invalid full-scope rate calibration')
        gradient_rows.append({'sample_path': row['sample_path'], 'd_vq_grad_l2': dnorm, 'rate_grad_l2': rnorm})
        del prediction, vq, rate, tensor, valid
    coefficient = min(10., max(1e-4, .1 * float(np.median([r['d_vq_grad_l2'] / r['rate_grad_l2'] for r in gradient_rows]))))
    write_json(out / 'calibration.json', {'temperature': temperature, 'coefficient': coefficient,
                                        'gradients': gradient_rows, 'parameter_updates': 0})
    model.eval().requires_grad_(False)
    baselines = {}
    with torch.no_grad():
        for split, rows in [('train', check_rows), ('val', validation)]:
            indices = {name: i for i, name in enumerate(paired['selection'][split])}
            for row in rows:
                raw, packet = load_packet(row)
                frame = (args.paired_dir / split / f'{indices[row["sample_path"]]:05d}.source.frame').read_bytes()
                original, shape, latent, side = stream.decode(frame)
                assert shape == raw.shape and side == packet.side_stream
                prediction = model.decoder(vectors(torch.from_numpy(original.astype(np.int64)).cuda(),
                    model.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]][0, 0].cpu().numpy()
                baselines[row['sample_path']] = (prediction, finite_json(score(row, raw, packet, prediction)), frame)
    del parameters, model
    torch.cuda.empty_cache()

    def evaluate(model, rows, collect=False):
        model.eval()
        losses, records = [], []
        with torch.no_grad():
            for row in rows:
                raw, packet = load_packet(row)
                tensor = torch.from_numpy(packet.values)[None, None].cuda()
                valid = torch.from_numpy(packet.valid)[None, None].cuda()
                before_prediction, before, source_frame = baselines[row['sample_path']]
                prediction, _, rate, indices, latent, distances = frozen_forward(model, tensor, costs, temperature, False)
                array = prediction[0, 0].cpu().numpy()
                candidate_frame = stream.encode(indices, raw.shape, latent, packet.side_stream)
                recovered, shape, latent_shape, side = stream.decode(candidate_frame)
                np.testing.assert_array_equal(recovered, indices.cpu().numpy())
                assert shape == raw.shape and tuple(latent_shape) == latent and side == packet.side_stream
                after = finite_json(score(row, raw, packet, array))
                eligible = len(candidate_frame) <= len(source_frame) and guarded_admit(before, after)
                chosen = prediction if eligible else torch.from_numpy(before_prediction)[None, None].cuda()
                losses.append((row['family'], row['configuration'], float(distortion(chosen, tensor, valid))))
                if collect:
                    soft = float((torch.softmax(-distances / temperature, dim=1) * costs).sum() / tensor.numel())
                    records.append({'sample_path': row['sample_path'], 'source': before, 'candidate': after,
                        'guarded': after if eligible else before, 'use_candidate': bool(eligible),
                        'source_frame_bytes': len(source_frame), 'candidate_frame_bytes': len(candidate_frame),
                        'selected_frame': pack_frame(candidate_frame if eligible else source_frame, eligible),
                        'source_frame': pack_frame(source_frame, False), 'hard_bpc': float(rate), 'soft_bpc': soft})
        return macro_loss(losses), records

    histories, selections = {}, {}
    for arm in ('control', 'rate'):
        model, config = new_model()
        model.train().requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
        best_loss, _ = evaluate(model, check_rows)
        best_step = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        histories[arm] = [{'step': 0, 'internal_guarded_distortion': best_loss}]
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for step, index in enumerate(sequence, 1):
            row = fit_rows[index]
            _, packet = load_packet(row)
            tensor = torch.from_numpy(packet.values)[None, None].cuda()
            valid = torch.from_numpy(packet.valid)[None, None].cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, vq, rate, _, _, _ = frozen_forward(model, tensor, costs, temperature, arm == 'rate')
            loss = distortion(prediction, tensor, valid) + vq + (coefficient * rate if arm == 'rate' else 0)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite joint-VQ training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            del prediction, loss, vq, rate, tensor, valid
            if step in (256, 512, 1024, budget):
                value, _ = evaluate(model, check_rows)
                histories[arm].append({'step': step, 'internal_guarded_distortion': value})
                if value < best_loss:
                    best_loss, best_step = value, step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                atomic_checkpoint(out / f'{arm}_last.pt', {'model_state_dict': model.state_dict(), 'config': config,
                    'step': step, 'optimizer': optimizer.state_dict(), 'source_sha256': source_hash,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()})
                write_json(out / 'history.json', histories)
                print(json.dumps({'arm': arm, 'step': step, 'internal_distortion': value, 'best_step': best_step}), flush=True)
        atomic_checkpoint(out / f'{arm}_selected.pt', {'model_state_dict': best_state, 'config': config,
            'step': best_step, 'source_sha256': source_hash})
        selections[arm] = {'selected_step': best_step, 'completed_updates': budget, 'internal_guarded_distortion': best_loss,
                           'seconds': time.perf_counter() - start, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'selected_waveform_weight_bytes': (out / f'{arm}_selected.pt').stat().st_size}
        del optimizer, model, best_state
        torch.cuda.empty_cache()
    write_json(out / 'selection.json', selections)
    results = {'parameter_updates': 2 * budget, 'gpu': torch.cuda.get_device_name(), 'selection': selections,
               'rate_coefficient': coefficient, 'temperature': temperature, 'validation': {}}
    for arm in ('control', 'rate'):
        saved = torch.load(out / f'{arm}_selected.pt', map_location='cpu', weights_only=False)
        model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
        _, rows = evaluate(model, validation, True)
        folder = out / arm
        folder.mkdir()
        for i, record in enumerate(rows):
            (folder / f'{i:05d}.fsc').write_bytes(record.pop('selected_frame'))
            source_frame = record.pop('source_frame')
            if arm == 'control':
                (folder / f'{i:05d}.source.fsc').write_bytes(source_frame)
        write_json(folder / 'records.json', rows)
        write_json(folder / 'registry.json', {'format': 'FSC1', 'kind': 'full_model_routing',
            'mode_zero_model_sha256': source_hash, 'mode_one_model_sha256': digest(out / f'{arm}_selected.pt'),
            'inner_table_identity': 'frozen-source-alphabet; inner checkpoint field is not candidate waveform identity'})
        summary = {}
        for label in ('source', 'candidate', 'guarded'):
            macro, groups = aggregate_metrics([r[label] for r in rows])
            summary[label] = {'macro': macro, 'groups': groups}
        summary['selected_frame_bytes'] = sum((folder / f'{i:05d}.fsc').stat().st_size for i in range(len(rows)))
        summary['source_frame_bytes'] = sum(r['source_frame_bytes'] + 13 for r in rows)
        summary['candidate_payload_bytes'] = sum(r['candidate_frame_bytes'] for r in rows)
        summary['admitted_windows'] = sum(r['use_candidate'] for r in rows)
        summary['rmse_gain_vs_source'] = 1 - summary['guarded']['macro']['macro_relative_rmse'] / summary['source']['macro']['macro_relative_rmse']
        summary['hard_bpc_unweighted_window_mean'] = float(np.mean([r['hard_bpc'] for r in rows]))
        summary['soft_bpc_unweighted_window_mean'] = float(np.mean([r['soft_bpc'] for r in rows]))
        results['validation'][arm] = summary
        del saved, model
        torch.cuda.empty_cache()
    np.save(out / 'counts.npy', np.load(args.calibration_dir / 'counts.npy', allow_pickle=False))
    maps = [json.loads(line) for line in (args.samples_dir / 'channel_maps.jsonl').read_text().splitlines()]
    ids = {r['channel_map_id'] for r in validation}
    write_json(out / 'maps.json', [m for m in maps if m['channel_map_id'] in ids])
    write_json(out / 'transport.json', [{'frame': f'{i:05d}.fsc', 'sample_path': r['sample_path'], 'drive_id': r['drive_id'],
        'block_index': int(r['block_index']), 'time_start': int(r['time_start']), 'channel_map_id': r['channel_map_id']}
        for i, r in enumerate(validation)])
    common = sum((out / name).stat().st_size for name in ('counts.npy', 'maps.json', 'transport.json'))
    for arm in ('control', 'rate'):
        results['validation'][arm]['stream_bytes'] = results['validation'][arm]['selected_frame_bytes'] + common + (out / arm / 'registry.json').stat().st_size
    control, rate = results['validation']['control'], results['validation']['rate']
    results['rate_gain_vs_control'] = 1 - rate['guarded']['macro']['macro_relative_rmse'] / control['guarded']['macro']['macro_relative_rmse']
    results['investment_gate_met'] = results['rate_gain_vs_control'] >= .05 and rate['selected_frame_bytes'] <= control['selected_frame_bytes']
    results['common_auxiliary_bytes'] = common
    results['base_weights_shared_excluded_bytes'] = args.source.stat().st_size
    results['candidate_weights_additional_shared_bytes'] = {arm: (out / f'{arm}_selected.pt').stat().st_size for arm in ('control', 'rate')}
    results['limitations'] = protocol['limitations']
    # Replay serialized full models in CPU with explicit outer mode routing.
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    base = build_model(saved['config'], saved, 'cpu').eval().requires_grad_(False)
    del saved
    replay = []
    for arm in ('control', 'rate'):
        saved = torch.load(out / f'{arm}_selected.pt', map_location='cpu', weights_only=False)
        candidate = build_model(saved['config'], saved, 'cpu').eval().requires_grad_(False)
        records = json.loads((out / arm / 'records.json').read_text())
        positive = [i for i, r in enumerate(records) if r['use_candidate']]
        for i in sorted(set([0, len(records) - 1, *positive[:1]])):
            restored = decode_model_frame((out / arm / f'{i:05d}.fsc').read_bytes(), stream, base, candidate, 'cpu')
            raw, packet = load_packet(validation[i])
            assert restored.shape == raw.shape and np.isfinite(restored).all()
            np.testing.assert_array_equal(restored[~packet.valid], raw[~packet.valid])
            replay.append({'arm': arm, 'index': i, 'finite': True, 'escapes_exact': True})
        del candidate, saved
    results['independent_cpu_replay'] = replay
    results['cpu_gpu_bit_identity_claimed'] = False
    assert digest(args.source) == source_hash
    write_json(out / 'results.json', finite_json(results))
    (out / 'COMPLETE').touch()
    print(json.dumps({'rate_gain_vs_control': results['rate_gain_vs_control'], 'gate': results['investment_gate_met'],
                      'selection': selections}), flush=True)


if __name__ == '__main__':
    main()
