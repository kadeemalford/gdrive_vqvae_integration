#!/usr/bin/env python3
"""Paired encoder-only VQ pilots with frozen reconstruction and real frame ceilings."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.finetune_vqvae import save_checkpoint
from finetuning.model_factory import build_model
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import (digest, distortion, load_packet, pilot_rows,
    read_manifest, score, tensor_packet, write_json)
from finetuning.run_index_search import FrozenIndexStream
from finetuning.summarize_index_search import admit_search
from finetuning.vq_encoder_rd import (PairedTileSampler, configure_encoder_only, frozen_forward)


def load_model(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = {**saved['config'], 'exact_eval_quantization': True}
    model = build_model(config, saved, 'cuda')
    return model, config


def gradient_norm(loss, parameters, retain_graph=False):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))


def calibration_rows(train):
    rows = pilot_rows(train, seed=2081)
    chosen = {}
    for family in sorted({r['family'] for r in rows}):
        values = [r for r in rows if r['family'] == family]
        for row in (min(values, key=lambda r: (int(r['channels']), r['sample_path'])),
                    max(values, key=lambda r: (int(r['channels']), r['sample_path']))):
            chosen[row['sample_path']] = row
    return list(chosen.values())


def prepare(args, splits):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    model, config = load_model(args.checkpoint)
    model.eval().requires_grad_(False)
    counts = np.zeros(2048, dtype=np.int64)
    gaps = []
    uniform_costs = torch.ones(2048, device='cuda')
    train = pilot_rows(splits['train'], seed=2081)
    with torch.no_grad():
        for row in train:
            _, packet = load_packet(row)
            tensor, _ = tensor_packet(packet)
            _, _, _, indices, _, distances = frozen_forward(model, tensor, uniform_costs, 1., False)
            counts += np.bincount(indices.cpu().numpy(), minlength=2048)
            pair = distances.topk(2, dim=1, largest=False).values
            gaps.append(float((pair[:, 1] - pair[:, 0]).median()))
    temperature = max(1e-5, float(np.median(gaps)) * args.temperature_scale)
    stream = FrozenIndexStream(counts, digest(args.checkpoint))
    np.save(output / 'counts.npy', counts)
    costs = -torch.log2(torch.from_numpy(stream.probabilities).cuda())
    parameters = configure_encoder_only(model)
    model.encoder.train()
    diagnostics = []
    for row in calibration_rows(splits['train']):
        _, packet = load_packet(row)
        tensor, valid = tensor_packet(packet)
        prediction, vq, rate, _, _, _ = frozen_forward(model, tensor, costs, temperature)
        d = distortion(prediction, tensor, valid) + vq
        d_norm = gradient_norm(d, parameters, retain_graph=True)
        r_norm = gradient_norm(rate, parameters)
        if not (math.isfinite(d_norm) and math.isfinite(r_norm) and r_norm > 0):
            raise RuntimeError('Invalid train-only rate-gradient calibration')
        diagnostics.append({'sample_path': row['sample_path'], 'channels': int(row['channels']),
                            'distortion_vq_gradient_l2': d_norm, 'rate_gradient_l2': r_norm})
    ratio = float(np.median([r['distortion_vq_gradient_l2'] / r['rate_gradient_l2'] for r in diagnostics]))
    coefficient = min(10., max(1e-4, .1 * ratio))
    write_json(output / 'calibration.json', {'kind': 'frozen_vq_encoder_rd_calibration_v1',
        'seed': 2081, 'checkpoint_sha256': digest(args.checkpoint),
        'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'temperature': temperature, 'temperature_scale': args.temperature_scale,
        'temperature_rule': 'scale * median train-window median nearest-two squared-distance gap; floor 1e-5',
        'rate_coefficient': coefficient,
        'coefficient_rule': '0.1 * median train ||grad(D+VQ)||/||grad(R)||; bounded [1e-4,10]',
        'train_table_windows': [r['sample_path'] for r in train], 'gradient_diagnostics': diagnostics,
        'config': config, 'source_sha256': digest(Path(__file__)), 'no_test_inference': True})
    (output / 'COMPLETE').touch()
    print(json.dumps({'temperature': temperature, 'rate_coefficient': coefficient}), flush=True)


def evaluate(model, baseline, rows, stream, costs, temperature, directory, samples):
    directory.mkdir(exist_ok=False)
    model.eval(); baseline.eval()
    measured = {name: [] for name in ('baseline', 'candidate', 'guarded')}
    records, transport = [], []
    source_bytes = candidate_bytes = guarded_bytes = cells = accepted = 0
    with torch.no_grad():
        for index, row in enumerate(rows):
            raw, packet = load_packet(row)
            tensor, _ = tensor_packet(packet)
            a, _, _, original, latent, _ = frozen_forward(baseline, tensor, costs, temperature, False)
            b, _, _, indices, candidate_latent, _ = frozen_forward(model, tensor, costs, temperature, False)
            assert latent == candidate_latent and original.numel() == indices.numel()
            old_frame = stream.encode(original, raw.shape, latent, packet.side_stream)
            new_frame = stream.encode(indices, raw.shape, latent, packet.side_stream)
            recovered, shape, recovered_latent, side = stream.decode(new_frame)
            np.testing.assert_array_equal(recovered, indices.cpu().numpy())
            assert side == packet.side_stream and shape == raw.shape and tuple(recovered_latent) == latent
            q = model.quantizer.embedding(torch.from_numpy(recovered.astype(np.int64)).cuda())
            q = q.view(1, *latent, model.embedding_dim).permute(0, 3, 1, 2).contiguous()
            replay = baseline.decoder(q)[..., :shape[0], :shape[1]]
            torch.testing.assert_close(replay, b, rtol=0, atol=0)
            before = score(row, raw, packet, a[0, 0].cpu().numpy())
            after = score(row, raw, packet, replay[0, 0].cpu().numpy())
            eligible = len(new_frame) <= len(old_frame) and admit_search(before, after)
            chosen = after if eligible else before
            selected_frame = new_frame if eligible else old_frame
            accepted += int(eligible and not np.array_equal(recovered, original.cpu().numpy()))
            for name, metrics in (('baseline', before), ('candidate', after), ('guarded', chosen)):
                measured[name].append(metrics)
            for suffix, frame in (('baseline', old_frame), ('candidate', new_frame), ('fnp', selected_frame)):
                (directory / f'{index:05d}.{suffix}').write_bytes(frame)
            records.append({'sample_path': row['sample_path'], 'baseline': before, 'candidate': after,
                'guarded': chosen, 'use_candidate': bool(eligible), 'baseline_frame_bytes': len(old_frame),
                'candidate_frame_bytes': len(new_frame), 'frame_bytes': len(selected_frame)})
            transport.append({'frame': f'{index:05d}.fnp', 'sample_path': row['sample_path'],
                'drive_id': row['drive_id'], 'block_index': int(row['block_index']),
                'time_start': int(row['time_start']), 'channel_map_id': row['channel_map_id']})
            source_bytes += len(old_frame); candidate_bytes += len(new_frame)
            guarded_bytes += len(selected_frame); cells += raw.size
            if (index + 1) % 512 == 0:
                print(json.dumps({'validation_windows': index + 1, 'total': len(rows)}), flush=True)
    identifiers = {r['channel_map_id'] for r in rows}
    maps = [json.loads(line) for line in (samples / 'channel_maps.jsonl').read_text().splitlines()]
    write_json(directory / 'maps.json', [m for m in maps if m['channel_map_id'] in identifiers])
    write_json(directory / 'transport.json', transport)
    (directory / 'huffman_table.json').write_bytes(stream.huffman.serialized)
    (directory / 'range_descriptor.json').write_bytes(stream.codec.serialized)
    (directory / 'range_probabilities.f32').write_bytes(stream.probabilities.tobytes())
    auxiliary = sum((directory / name).stat().st_size for name in
        ('maps.json', 'transport.json', 'huffman_table.json', 'range_descriptor.json', 'range_probabilities.f32'))
    result = {'windows': len(rows), 'input_cells': cells, 'accepted_changed_windows': accepted,
        'shared_tables_maps_transport_bytes': auxiliary, 'exact_original_decoder_replay': True,
        'waveform_weights_shared_excluded': True, 'policy': 'per-window nonincreasing real bytes, RMSE, p90 and nondecreasing correlation'}
    for name, size in (('baseline', source_bytes), ('candidate', candidate_bytes), ('guarded', guarded_bytes)):
        macro, groups = aggregate_validation(measured[name])
        result[name] = {'macro': macro, 'groups': groups, 'frame_bytes': size,
            'total_stream_bytes': size + auxiliary, 'total_bpc': 8 * (size + auxiliary) / cells,
            'ratio_float32': 4 * cells / (size + auxiliary)}
    write_json(directory / 'records.json', records)
    write_json(directory / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--temperature-scale', type=float, default=1., help='Train-only calibration multiplier; never selected on validation')
    parser.add_argument('--rate', action='store_true')
    parser.add_argument('--sampling', choices=('window', 'configuration-cell'), default='window')
    parser.add_argument('--initial-steps', type=int, default=8192)
    parser.add_argument('--maximum-steps', type=int, default=16384)
    parser.add_argument('--block-steps', type=int, default=2048)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.temperature_scale) or args.temperature_scale <= 0:
        raise ValueError('Positive finite temperature scale required')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2081)
    splits = read_manifest(args.samples_dir)
    if args.prepare:
        prepare(args, splits)
        return
    if args.calibration_dir is None or not (args.calibration_dir / 'COMPLETE').exists():
        raise ValueError('Completed train-only calibration required')
    if not (args.maximum_steps >= args.initial_steps >= 2 * args.block_steps > 0):
        raise ValueError('Invalid bounded update budget')
    calibration = json.loads((args.calibration_dir / 'calibration.json').read_text())
    assert calibration['checkpoint_sha256'] == digest(args.checkpoint)
    assert calibration['manifest_sha256'] == digest(args.samples_dir / 'samples_manifest.csv')
    coefficient = calibration['rate_coefficient'] if args.rate else 0.
    temperature = calibration['temperature']
    counts = np.load(args.calibration_dir / 'counts.npy', allow_pickle=False)
    stream = FrozenIndexStream(counts, calibration['checkpoint_sha256'])
    costs = -torch.log2(torch.from_numpy(stream.probabilities).cuda())
    selected = pilot_rows(splits['val'], per_region=2, seed=2081)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=args.resume)
    source_files = [Path(__file__), ROOT / 'finetuning/vq_encoder_rd.py', ROOT / 'finetuning/representation_study.py',
        ROOT / 'finetuning/run_index_search.py', ROOT / 'finetuning/summarize_index_search.py',
        ROOT / 'compression/symbol_stream.py', ROOT / 'compression/neural_symbol_stream.py']
    protocol = {'kind': 'frozen_decoder_encoder_rd_v1', 'seed': 2081, 'checkpoint_sha256': digest(args.checkpoint),
        'manifest_sha256': calibration['manifest_sha256'], 'calibration_sha256': digest(args.calibration_dir / 'calibration.json'),
        'rate_coefficient': coefficient, 'temperature': temperature, 'sampling': args.sampling,
        'sampling_control': 'identical configuration draws, historical configuration update probabilities; change only within-config CDF',
        'initial_steps': args.initial_steps, 'maximum_steps': args.maximum_steps, 'block_steps': args.block_steps,
        'extension': 'once if last-block guarded macro RMSE improves >=0.2% and guarded gain vs source >=0.2%',
        'optimizer': 'encoder-only AdamW lr=1e-5 weight_decay=1e-5; clip=1; FP32; batch=1',
        'frozen': ['decoder', 'codebook', 'geometry', 'normalization', 'probability_tables'],
        'objective': '0.5*MSE+0.5*L1 ordinary + historical VQ + coefficient*hard marginal bpc with full-softmax distance gradient',
        'selection': 'minimum guarded macro RMSE; initial source is eligible; raw results separately retained',
        'guards': 'per-window real bytes/RMSE/channel-p90 no higher, correlation no lower; no tolerance relaxation',
        'investment_target': '>=5% guarded macro RMSE gain vs matched control with no more complete bytes; confirm separately',
        'validation_pilot': [r['sample_path'] for r in selected], 'no_test_inference': True,
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in source_files},
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0)}
    if args.resume:
        assert json.loads((output / 'protocol.json').read_text()) == protocol, 'Protocol changed'
        if (output / 'COMPLETE').exists():
            return
    else:
        write_json(output / 'protocol.json', protocol)
        for path in source_files:
            destination = output / 'code_snapshot' / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    baseline, config = load_model(args.checkpoint)
    baseline.eval().requires_grad_(False)
    model, _ = load_model(args.checkpoint)
    parameters = configure_encoder_only(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-5)
    sampler = PairedTileSampler(splits['train'], args.sampling, 2081)
    step = 0; target_steps = args.initial_steps
    group_counts = Counter(); size_counts = Counter(); exposed_cells = 0
    if args.resume:
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict']); optimizer.load_state_dict(saved['optimizer'])
        sampler.rng.setstate(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        step, target_steps = saved['step'], saved['target_steps']
        group_counts.update(saved['group_counts']); size_counts.update(saved['size_counts'])
        exposed_cells = saved['exposed_cells']
        del saved
    else:
        initial = evaluate(model, baseline, selected, stream, costs, temperature, output / 'pilot_000000', args.samples_dir)
        write_json(output / 'history.json', [{'step': 0, 'guarded_rmse': initial['guarded']['macro']['macro_relative_rmse'],
            'raw_rmse': initial['candidate']['macro']['macro_relative_rmse'], 'guarded_bpc': initial['guarded']['total_bpc']}])
        save_checkpoint({'config': config, 'model_state_dict': model.state_dict(), 'step': 0}, output / 'best.pt')
    history = json.loads((output / 'history.json').read_text())
    best = min(r['guarded_rmse'] for r in history)
    raw_best = min(r['raw_rmse'] for r in history)
    started = time.monotonic()
    objectives = []
    while step < target_steps:
        if shutil.disk_usage(output).free < 20 * 2**30:
            raise RuntimeError('Less than 20 GiB free')
        model.eval(); model.encoder.train()
        row = sampler.draw()
        _, packet = load_packet(row)
        tensor, valid = tensor_packet(packet)
        optimizer.zero_grad(set_to_none=True)
        predicted, vq, rate, _, _, _ = frozen_forward(model, tensor, costs, temperature, bool(coefficient))
        d = distortion(predicted, tensor, valid)
        loss = d + vq + coefficient * rate
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite encoder objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        group_counts['/'.join(sampler.key(row))] += 1
        height = int(row['channels'])
        size_counts['1-3' if height <= 3 else '4-32' if height <= 32 else '33-128' if height <= 128 else '129-512'] += 1
        exposed_cells += tensor.numel()
        objectives.append(float(loss.detach()))
        if step % 256 == 0:
            print(json.dumps({'step': step, 'target': target_steps, 'objective': float(np.mean(objectives[-256:])),
                'seconds': time.monotonic() - started}), flush=True)
        if step % args.block_steps == 0 or step == target_steps:
            result = evaluate(model, baseline, selected, stream, costs, temperature,
                              output / f'pilot_{step:06d}', args.samples_dir)
            rmse = result['guarded']['macro']['macro_relative_rmse']
            raw_rmse = result['candidate']['macro']['macro_relative_rmse']
            entry = {'step': step, 'guarded_rmse': rmse, 'raw_rmse': raw_rmse,
                'guarded_bpc': result['guarded']['total_bpc'], 'raw_bpc': result['candidate']['total_bpc'],
                'accepted_changed_windows': result['accepted_changed_windows'], 'exposed_cells': exposed_cells,
                'seconds': time.monotonic() - started}
            history.append(entry); write_json(output / 'history.json', history)
            if rmse < best:
                best = rmse
                save_checkpoint({'config': config, 'model_state_dict': model.state_dict(), 'step': step}, output / 'best.pt')
            if raw_rmse < raw_best:
                raw_best = raw_rmse
                save_checkpoint({'config': config, 'model_state_dict': model.state_dict(), 'step': step}, output / 'best_raw.pt')
            if step == args.initial_steps:
                trend = 1 - rmse / history[-2]['guarded_rmse']
                gain = 1 - rmse / history[0]['guarded_rmse']
                extend = trend >= .002 and gain >= .002
                target_steps = args.maximum_steps if extend else args.initial_steps
                write_json(output / 'extension_decision.json', {'extend': extend,
                    'last_block_gain': trend, 'gain_vs_source': gain, 'target_steps': target_steps})
            save_checkpoint({'config': config, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'target_steps': target_steps, 'sampler_rng': sampler.rng.getstate(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'group_counts': dict(group_counts), 'size_counts': dict(size_counts), 'exposed_cells': exposed_cells}, output / 'last.pt')
            print(json.dumps({'pilot': entry}), flush=True)
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    baseline_state = baseline.state_dict()
    for name, parameter in model.named_parameters():
        if not name.startswith('encoder.'):
            torch.testing.assert_close(parameter, baseline_state[name], rtol=0, atol=0)
    result = evaluate(model, baseline, splits['val'], stream, costs, temperature, output / 'full_validation', args.samples_dir)
    result.update({'selected_step': saved['step'], 'completed_steps': step,
        'checkpoint_sha256': digest(output / 'best.pt'), 'source_checkpoint_sha256': digest(args.checkpoint),
        'rate_coefficient': coefficient, 'sampling': args.sampling, 'frozen_parameters_bit_exact': True,
        'exposure': {'configuration_updates': dict(group_counts), 'size_updates': dict(size_counts),
                     'cells': exposed_cells}, 'no_test_inference': True})
    write_json(output / 'results.json', result)
    (output / 'COMPLETE').touch()
    print(json.dumps({'complete': str(output), 'selected_step': saved['step'],
        'guarded_rmse': result['guarded']['macro']['macro_relative_rmse']}), flush=True)


if __name__ == '__main__':
    main()
