#!/usr/bin/env python3
"""Bounded E2 training pilot; exact scalar streams and common physical validation."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.scalar_stream import ScalarStream
from finetuning.scalar_rd import ScalarRDCodec
from finetuning.finetune_vqvae import save_checkpoint
from finetuning.model_factory import build_model
from finetuning.representation_study import (digest, distortion, load_packet, pilot_rows,
    read_manifest, score, tensor_packet, write_json)
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.latent_geometry import latent_positions


def evaluate(model, selected, output, samples):
    model.eval()
    output.mkdir(exist_ok=False)
    stream = ScalarStream.from_scales(model.scales.detach().cpu().numpy())
    (output / 'prior.f32').write_bytes(stream.serialized)
    # The receiver can load the exact transmitted table, rather than reproducing floating-point CDFs.
    receiver = ScalarStream(np.frombuffer((output / 'prior.f32').read_bytes(), dtype='<f4').reshape(-1, 512))
    rows, records, transport = [], [], []
    total_cells = total_bytes = estimated_bits = 0
    for index, row in enumerate(selected):
        raw, packet = load_packet(row)
        target, valid = tensor_packet(packet)
        with torch.no_grad():
            prediction, rate, quantized = model(target)
            symbols = quantized[0].cpu().numpy()
            frame = stream.encode(symbols, raw.shape, packet.side_stream, row['channel_map_id'])
            recovered, shape, side, map_id = receiver.decode(frame)
            np.testing.assert_array_equal(symbols, recovered)
            assert side == packet.side_stream and shape == raw.shape and map_id == row['channel_map_id']
            replay = model.synthesis(torch.from_numpy(recovered.copy())[None].cuda().float(), shape)
            torch.testing.assert_close(replay, prediction, rtol=0, atol=0)
        filename = f'{index:05d}.srd'
        (output / filename).write_bytes(frame)
        measured = score(row, raw, packet, replay[0, 0].cpu().numpy())
        measured['estimated_latent_bpc'] = float(rate)
        measured['frame_bpc'] = 8 * len(frame) / raw.size
        rows.append(measured)
        transport.append({'frame': filename, 'sample_path': row['sample_path'], 'drive_id': row['drive_id'],
                          'block_index': int(row['block_index']), 'time_start': int(row['time_start']),
                          'channel_map_id': row['channel_map_id']})
        records.append({'sample_path': row['sample_path'], 'frame_bytes': len(frame), 'metrics': measured})
        total_cells += raw.size; total_bytes += len(frame); estimated_bits += float(rate) * raw.size
    identifiers = {r['channel_map_id'] for r in selected}
    maps = [json.loads(line) for line in (samples / 'channel_maps.jsonl').read_text().splitlines()]
    write_json(output / 'maps.json', [m for m in maps if m['channel_map_id'] in identifiers])
    write_json(output / 'transport.json', transport)
    auxiliary = sum((output / name).stat().st_size for name in ('maps.json', 'transport.json', 'prior.f32'))
    macro, groups = aggregate_validation(rows)
    result = {'windows': len(rows), 'input_cells': total_cells, 'frame_bytes': total_bytes,
              'shared_tables_maps_transport_bytes': auxiliary, 'total_stream_bytes': total_bytes + auxiliary,
              'estimated_latent_bpc': estimated_bits / total_cells,
              'frame_bpc': 8 * total_bytes / total_cells,
              'total_bpc': 8 * (total_bytes + auxiliary) / total_cells,
              'ratio_float32': 4 * total_cells / (total_bytes + auxiliary),
              'macro': macro, 'groups': groups, 'exact_scalar_and_waveform_replay': True,
              'waveform_weights_shared_not_in_stream_bytes': True}
    write_json(output / 'records.json', records)
    write_json(output / 'results.json', result)
    return result


def reference(checkpoint_path, selected, destination):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = build_model({**checkpoint['config'], 'exact_eval_quantization': True}, checkpoint, 'cuda').eval()
    rows = []
    bits = cells = 0
    with torch.no_grad():
        for row in selected:
            raw, packet = load_packet(row)
            target, _ = tensor_packet(packet)
            prediction, _, _, _ = model(target)
            rows.append(score(row, raw, packet, prediction[0, 0].cpu().numpy()))
            bits += latent_positions(model, *raw.shape) * 11 + 8 * len(packet.side_stream)
            cells += raw.size
    macro, groups = aggregate_validation(rows)
    result = {'checkpoint': str(checkpoint_path), 'sha256': digest(checkpoint_path),
              'macro': macro, 'groups': groups, 'nominal_index_side_bpc': bits / cells,
              'not_a_wire_matched_comparison': True}
    write_json(destination, result)
    del model, checkpoint
    torch.cuda.empty_cache()
    return result


def smoke(model, maximum, output):
    raw, packet = load_packet(maximum)
    target, valid = tensor_packet(packet)
    model.train()
    _, rate, _ = model(target)
    gradients = torch.autograd.grad(rate, tuple(model.encoder.parameters()), allow_unused=True)
    norm = sum(float(g.abs().sum()) for g in gradients if g is not None)
    assert norm > 0 and math.isfinite(norm), 'Rate gradient must reach the encoder'
    model.eval()
    with torch.no_grad():
        for h in (1, 3, 111, 512):
            x = target[..., :h, :]
            prediction, bpc, symbols = model(x)
            replay = model.synthesis(symbols, x.shape[-2:])
            torch.testing.assert_close(prediction, replay, rtol=0, atol=0)
            assert prediction.shape == x.shape and torch.isfinite(bpc)
    write_json(output / 'smoke.json', {'rate_encoder_gradient_l1': norm,
        'shapes': [[h, 1000] for h in (1, 3, 111, 512)], 'updates': 0,
        'parameters': sum(p.numel() for p in model.parameters()),
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rate-weight', type=float, required=True)
    parser.add_argument('--latent-channels', type=int, default=16)
    parser.add_argument('--prior-learning-rate', type=float, default=1e-4)
    parser.add_argument('--initialize-from', type=Path, help='Weights-only warm start; fresh optimizer and bounded new budget')
    parser.add_argument('--initial-steps', type=int, default=4096)
    parser.add_argument('--maximum-steps', type=int, default=8192)
    parser.add_argument('--block-steps', type=int, default=1024)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.rate_weight) or args.rate_weight <= 0:
        raise ValueError('Rate weight must be finite and positive')
    if args.latent_channels < 1 or not math.isfinite(args.prior_learning_rate) or args.prior_learning_rate <= 0:
        raise ValueError('Latent channels and prior learning rate must be positive')
    assert args.maximum_steps >= args.initial_steps >= 2 * args.block_steps > 0
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2078)
    np.random.seed(2078)
    rng = random.Random(2078)
    splits = read_manifest(args.samples_dir)
    selected = pilot_rows(splits['val'])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=args.resume)
    sources = [Path(__file__), ROOT / 'finetuning/scalar_rd.py', ROOT / 'compression/scalar_stream.py',
               ROOT / 'finetuning/representation_study.py']
    protocol = {'kind': 'e2_factorized_scalar_v1', 'seed': 2078, 'config': {'base_channels': 32,
        'latent_channels': args.latent_channels, 'geometry': 'spatial-4x'}, 'rate_weight': args.rate_weight,
        'initial_steps': args.initial_steps, 'maximum_steps': args.maximum_steps, 'block_steps': args.block_steps,
        'extension': 'once: last pilot RMSE improves >=0.5% relative to preceding block at initial boundary',
        'optimizer': 'AdamW lr=1e-4 weight_decay=1e-5; clip=1; FP32; batch=1',
        'prior_learning_rate': args.prior_learning_rate,
        'initialization': ({'path': str(args.initialize_from), 'sha256': digest(args.initialize_from),
                            'optimizer_reused': False} if args.initialize_from else {'kind': 'random'}),
        'objective': '0.5*MSE+0.5*L1 on ordinary cells + rate_weight*estimated_latent_bits/retained_cells',
        'training_quantization': 'independent uniform(-0.5,0.5) noise; exact rounding for validation',
        'sampling': 'uniform window; existing tile-weight discrepancy preserved as a separate question',
        'normalization': 'robust-escape-v1; all ordinary samples and exact side escapes retained',
        'train_windows': len(splits['train']), 'validation_pilot': [r['sample_path'] for r in selected],
        'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'reference_sha256': digest(args.reference_checkpoint), 'no_test_inference': True,
        'selection': 'minimum pilot macro ordinary relative RMSE + report measured rate; no final promotion',
        'investment_target': '>=5% RMSE gain at no more complete bytes or >=10% byte saving at equivalent fidelity; requires later matched confirmation',
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0)}
    if args.resume:
        assert json.loads((output / 'protocol.json').read_text()) == protocol
        if (output / 'COMPLETE').exists():
            return
    else:
        write_json(output / 'protocol.json', protocol)
        for path in sources:
            destination = output / 'code_snapshot' / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    model = ScalarRDCodec(**protocol['config']).cuda()
    if args.initialize_from:
        initialized = torch.load(args.initialize_from, map_location='cpu', weights_only=True)
        if initialized['config'] != model.config:
            raise ValueError('Warm-start scalar architecture/configuration mismatch')
        model.load_state_dict(initialized['model_state_dict'], strict=True)
        del initialized
    transforms = [p for name, p in model.named_parameters() if name != 'log_scale']
    optimizer = torch.optim.AdamW([{'params': transforms},
        {'params': [model.log_scale], 'lr': args.prior_learning_rate}], lr=1e-4, weight_decay=1e-5)
    if not args.resume:
        maximum = next(r for r in splits['train'] if int(r['channels']) == 512)
        smoke(model, maximum, output)
        reference(args.reference_checkpoint, selected, output / 'vq_reference.json')
        write_json(output / 'history.json', [])
    step = 0
    best = float('inf')
    target_steps = args.initial_steps
    if args.resume:
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict']); optimizer.load_state_dict(saved['optimizer'])
        step, best, target_steps = saved['step'], saved['best'], saved['target_steps']
        rng.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    started = time.monotonic()
    train_losses, train_rates = [], []
    while step < target_steps:
        if shutil.disk_usage(output).free < 20 * 2**30:
            raise RuntimeError('Less than 20 GiB free; refusing further training')
        model.train()
        row = rng.choice(splits['train'])
        _, packet = load_packet(row)
        tensor, valid = tensor_packet(packet)
        optimizer.zero_grad(set_to_none=True)
        prediction, rate, _ = model(tensor)
        d = distortion(prediction, tensor, valid)
        loss = d + args.rate_weight * rate
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        train_losses.append(float(d.detach())); train_rates.append(float(rate.detach()))
        step += 1
        if step % 128 == 0:
            print(json.dumps({'step': step, 'target': target_steps, 'train_distortion': float(np.mean(train_losses[-128:])),
                              'train_bpc': float(np.mean(train_rates[-128:])), 'elapsed_seconds': time.monotonic() - started}), flush=True)
        if step % args.block_steps == 0 or step == target_steps:
            result = evaluate(model, selected, output / f'pilot_{step:06d}', args.samples_dir)
            rmse = result['macro']['macro_relative_rmse']
            history = json.loads((output / 'history.json').read_text())
            entry = {'step': step, 'train_distortion': float(np.mean(train_losses)),
                     'train_estimated_bpc': float(np.mean(train_rates)), 'rmse': rmse,
                     'correlation': result['macro']['macro_correlation'], 'total_bpc': result['total_bpc'],
                     'elapsed_seconds': time.monotonic() - started}
            history.append(entry)
            write_json(output / 'history.json', history)
            if rmse < best:
                best = rmse
                save_checkpoint({'config': model.config, 'model_state_dict': model.state_dict(),
                                 'step': step, 'pilot': result}, output / 'best.pt')
            if step == args.initial_steps and target_steps == args.initial_steps:
                extend = len(history) >= 2 and 1 - rmse / history[-2]['rmse'] >= .005
                target_steps = args.maximum_steps if extend else args.initial_steps
                write_json(output / 'extension_decision.json', {'extend': extend, 'target_steps': target_steps,
                    'relative_last_block_gain': 1 - rmse / history[-2]['rmse'] if len(history) >= 2 else 0})
            save_checkpoint({'config': model.config, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'best': best, 'target_steps': target_steps, 'python_rng': rng.getstate(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, output / 'last.pt')
            train_losses, train_rates = [], []
            print(json.dumps({'pilot': entry}), flush=True)
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    full_path = output / 'full_validation'
    if full_path.exists():
        raise RuntimeError('Full validation exists; inspect interrupted output instead of overwriting')
    result = evaluate(model, splits['val'], full_path, args.samples_dir)
    result['selected_step'] = saved['step']
    result['completed_steps'] = step
    result['shared_waveform_weight_tensor_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
    result['best_checkpoint_sha256'] = digest(output / 'best.pt')
    write_json(output / 'results.json', result)
    (output / 'COMPLETE').touch()
    print(json.dumps({'complete': str(output), 'rmse': result['macro']['macro_relative_rmse'],
                      'bpc': result['total_bpc']}), flush=True)


if __name__ == '__main__':
    main()
