#!/usr/bin/env python3
"""Bounded exact-code E1 search, with E0 train-only structure diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.model_factory import build_model
from finetuning.representation_study import (digest, distortion, load_packet, pilot_rows,
    read_manifest, score, tensor_packet, write_json)
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.summarize_index_search import admit_search


def nearest(z, embedding):
    flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    distances = flat.square().sum(1, keepdim=True) - 2 * flat @ embedding.T + embedding.square().sum(1)
    return distances.argmin(1)


def vectors(indices, embedding, shape):
    return F.embedding(indices, embedding).view(1, *shape, embedding.shape[1]).permute(0, 3, 1, 2).contiguous()


def search(model, target, valid, steps, learning_rate, stream=None, side=b'', frame_path=None,
           candidate_guard=None):
    h, w = target.shape[-2:]
    with torch.no_grad():
        z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
        embedding = model.quantizer.embedding.weight.detach()
        original = nearest(z, embedding)
        initial = vectors(original, embedding, z.shape[-2:])
        baseline = model.decoder(initial)[..., :h, :w]
        best_loss = float(distortion(baseline, target, valid))
        best, best_indices = baseline.clone(), original.clone()
        budget = len(stream.encode(original, (h, w), z.shape[-2:], side)) if stream else None
        costs = (-torch.log2(torch.from_numpy(stream.probabilities).to(target.device))
                 if stream else None)
    # Optimized continuous vectors are a search aid only, never transmitted.
    proposal = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([proposal], lr=learning_rate)
    accepted = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = model.decoder(proposal)[..., :h, :w]
        loss = distortion(predicted, target, valid)
        loss.backward()
        assert torch.isfinite(proposal.grad).all()
        optimizer.step()
        if (step + 1) % 4 == 0 or step + 1 == steps:
            with torch.no_grad():
                indices = nearest(proposal, embedding)
                candidates = [indices]
                if stream:
                    delta = costs[indices] - costs[original]
                    # A rate-aware alternative chooses cheap code changes first.
                    # This is only a search heuristic: the actual stream enforces the cap.
                    cheap = delta <= 0
                    limited = torch.where(cheap, indices, original)
                    available = -float(delta[cheap].sum())
                    positive = torch.where(delta > 0)[0]
                    order = positive[torch.argsort(delta[positive])]
                    prefix = torch.cumsum(delta[order], 0) <= max(0., available - 32.)
                    limited[order[prefix]] = indices[order[prefix]]
                    candidates.append(limited)
                for candidate in candidates:
                    if stream and len(stream.encode(candidate, (h, w), z.shape[-2:], side)) > budget:
                        continue
                    exact = model.decoder(vectors(candidate, embedding, z.shape[-2:]))[..., :h, :w]
                    value = float(distortion(exact, target, valid))
                    if value < best_loss:
                        if candidate_guard is not None and not candidate_guard(
                                baseline[0, 0].cpu().numpy(), exact[0, 0].cpu().numpy()):
                            continue
                        best_loss, best, best_indices = value, exact.clone(), candidate.clone()
                        accepted.append(step + 1)
    with torch.no_grad():
        replay = model.decoder(vectors(best_indices, embedding, z.shape[-2:]))[..., :h, :w]
        torch.testing.assert_close(replay, best, rtol=0, atol=0)
        final_frame = stream.encode(best_indices, (h, w), z.shape[-2:], side) if stream else None
        if stream:
            assert len(final_frame) <= budget
            decoded, decoded_shape, decoded_latent, decoded_side = stream.decode(final_frame)
            np.testing.assert_array_equal(decoded, best_indices.cpu().numpy())
            assert decoded_shape == (h, w) and decoded_latent == tuple(z.shape[-2:]) and decoded_side == side
            if frame_path is not None:
                frame_path.write_bytes(final_frame)
                frame_path.with_suffix('.baseline').write_bytes(stream.encode(original, (h, w), z.shape[-2:], side))
    return baseline[0, 0].cpu().numpy(), best[0, 0].cpu().numpy(), {
        'indices': int(original.numel()), 'fixed_bits': int(original.numel() * 11),
        'changed_fraction': float((original != best_indices).float().mean()),
        'accepted_steps': accepted, 'exact_discrete_replay': True,
        **({'baseline_frame_bytes': budget, 'searched_frame_bytes': len(final_frame)} if stream else {})}


class FrozenIndexStream:
    """Training-fitted marginal range coding with the existing fixed/Huffman fallback."""
    def __init__(self, counts, checkpoint_hash):
        from compression.symbol_stream import SymbolCodec
        from compression.neural_symbol_stream import NeuralSymbolCodec
        probabilities = (counts + 1.) / (counts + 1.).sum()
        self.probabilities = probabilities.astype(np.float32)
        self.huffman = SymbolCodec.from_counts(counts, np.tile(counts, (17, 1)), checkpoint_hash)
        descriptor = {'huffman_table_sha256': self.huffman.identifier.hex(),
                      'kind': 'e1-frozen-marginal-range-v1', 'checkpoint_sha256': checkpoint_hash}
        self.codec = NeuralSymbolCodec(self.huffman, self.probabilities, descriptor)

    def encode(self, indices, shape, latent, side):
        original, _ = self.huffman.encode(indices.detach().cpu().numpy().astype(np.uint16), shape, latent, side)
        return self.codec.encode(original)[0]

    def decode(self, frame):
        return self.codec.decode(frame)


def structure(values):
    centered = values.astype(np.float64) - values.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    total = max(float(power.sum()), 1e-20)
    result = {f'temporal_cosine_lag{lag}': float((centered[:, lag:] * centered[:, :-lag]).sum() /
              max(np.linalg.norm(centered[:, lag:]) * np.linalg.norm(centered[:, :-lag]), 1e-20))
              for lag in (1, 10, 100)}
    result['spatial_adjacent_cosine'] = float((centered[1:] * centered[:-1]).sum() /
        max(np.linalg.norm(centered[1:]) * np.linalg.norm(centered[:-1]), 1e-20))
    frequencies = np.fft.rfftfreq(values.shape[1], 1 / 20000)
    for lo, hi in ((0, 1000), (1000, 5000), (5000, 10001)):
        result[f'power_{lo}_{hi}_hz'] = float(power[:, (frequencies >= lo) & (frequencies < hi)].sum() / total)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wire-budget', action='store_true')
    parser.add_argument('--validation-seed', type=int, default=2078)
    parser.add_argument('--exclude-validation-protocol', type=Path)
    parser.add_argument('--extra-exclusion-protocol', type=Path, action='append', default=[])
    parser.add_argument('--tail-guard', action='store_true')
    parser.add_argument('--guard-during-search', action='store_true',
                        help='Reject each improving discrete candidate that violates source metric guards')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2078)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    splits = read_manifest(args.samples_dir)
    selected = {s: pilot_rows(v) for s, v in splits.items()}
    excluded = set()
    if args.exclude_validation_protocol:
        excluded = set(json.loads(args.exclude_validation_protocol.read_text())['validation'])
    for path in args.extra_exclusion_protocol:
        excluded.update(json.loads(path.read_text())['validation'])
    selected['val'] = pilot_rows([r for r in splits['val'] if r['sample_path'] not in excluded],
                                  seed=args.validation_seed)
    assert not (excluded & {r['sample_path'] for r in selected['val']})
    # Eight train windows calibrate the search; validation never chooses settings.
    train = selected['train']
    calibration = []
    for family in sorted({r['family'] for r in train}):
        choices = [r for r in train if r['family'] == family]
        calibration.extend([min(choices, key=lambda r: int(r['channels'])), max(choices, key=lambda r: int(r['channels']))])
    write_json(args.output_dir / 'protocol.json', {
        'kind': 'e0_e1_v1', 'seed': 2078, 'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': digest(args.checkpoint), 'source_sha256': digest(Path(__file__)),
        'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'validation_seed': args.validation_seed, 'excluded_validation_windows': sorted(excluded),
        'tail_guard_predefined': args.tail_guard,
        'guard_during_search': args.guard_during_search,
        'guard_calibration': 'train LR calibration unchanged; constraints applied to every validation candidate' if args.guard_during_search else None,
        'guard_source_sha256': digest(ROOT / 'finetuning/summarize_index_search.py'),
        'steps': 48, 'train_learning_rates': [.01, .05], 'selection': 'train mean relative temporal distortion gain',
        'calibration': [r['sample_path'] for r in calibration],
        'validation': [r['sample_path'] for r in selected['val']], 'no_test_inference': True,
        'rate_contract': ('per-window real frozen marginal/fixed/Huffman frame ceiling; same tables/maps/weights'
                          if args.wire_budget else 'fixed 11 bits/index plus unchanged side; no entropy-stream parity claimed')})
    for path in (Path(__file__), ROOT / 'finetuning/representation_study.py', ROOT / 'finetuning/summarize_index_search.py'):
        destination = args.output_dir / 'code_snapshot' / 'finetuning' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = {**checkpoint['config'], 'exact_eval_quantization': True}
    model = build_model(config, checkpoint, 'cuda').eval().requires_grad_(False)
    stream = None
    if args.wire_budget:
        counts = np.zeros(2048, dtype=np.int64)
        with torch.no_grad():
            for row in train:
                _, packet = load_packet(row)
                tensor, _ = tensor_packet(packet)
                h, w = tensor.shape[-2:]
                z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
                indices = nearest(z, model.quantizer.embedding.weight)
                counts += np.bincount(indices.cpu().numpy(), minlength=2048)
        stream = FrozenIndexStream(counts, digest(args.checkpoint))
        np.save(args.output_dir / 'train_counts.npy', counts)
        (args.output_dir / 'huffman_table.json').write_bytes(stream.huffman.serialized)
        (args.output_dir / 'range_descriptor.json').write_bytes(stream.codec.serialized)
        (args.output_dir / 'range_probabilities.f32').write_bytes(stream.probabilities.tobytes())
    diagnostics = []
    for row in train:
        raw, packet = load_packet(row)
        diagnostics.append({'sample_path': row['sample_path'], 'family': row['family'],
                            'channels': raw.shape[0], **structure(packet.values)})
    weighting = {}
    for split, rows in splits.items():
        cells = sum(int(r['channels']) * int(r['time_samples']) for r in rows)
        small = [r for r in rows if int(r['channels']) <= 3]
        weighting[split] = {'windows': len(rows), 'cells': cells, 'small_windows': len(small),
            'small_window_fraction': len(small) / len(rows),
            'small_cell_fraction': sum(int(r['channels']) * int(r['time_samples']) for r in small) / cells}
    write_json(args.output_dir / 'signal_diagnostics.json', {'weighting': weighting, 'windows': diagnostics,
        'limitations': '50 ms train windows only; no continuous HDF5, pooled file quantiles or independence claim'})
    calibration_results = []
    for lr in (.01, .05):
        gains = []
        for row in calibration:
            raw, packet = load_packet(row)
            target, valid = tensor_packet(packet)
            a, b, details = search(model, target, valid, 48, lr, stream, packet.side_stream)
            before = .5 * np.mean((a[packet.valid] - packet.values[packet.valid]) ** 2) + .5 * np.mean(np.abs(a[packet.valid] - packet.values[packet.valid]))
            after = .5 * np.mean((b[packet.valid] - packet.values[packet.valid]) ** 2) + .5 * np.mean(np.abs(b[packet.valid] - packet.values[packet.valid]))
            gains.append(float(1 - after / max(before, 1e-20)))
        calibration_results.append({'lr': lr, 'mean_gain': float(np.mean(gains)), 'gains': gains})
    lr = max(calibration_results, key=lambda r: r['mean_gain'])['lr']
    write_json(args.output_dir / 'calibration.json', {'results': calibration_results, 'selected_lr': lr})
    scored = {'baseline': [], 'searched': []}
    records = []
    for i, row in enumerate(selected['val']):
        raw, packet = load_packet(row)
        target, valid = tensor_packet(packet)
        start = time.monotonic()
        cached_before = []
        def candidate_guard(original, candidate):
            if not cached_before:
                cached_before.append(score(row, raw, packet, original))
            return admit_search(cached_before[0], score(row, raw, packet, candidate))
        a, b, details = search(model, target, valid, 48, lr, stream, packet.side_stream,
                               args.output_dir / f'{i:05d}.searched' if stream else None,
                               candidate_guard if args.guard_during_search else None)
        before, after = score(row, raw, packet, a), score(row, raw, packet, b)
        details['guard_fallback'] = False
        if args.tail_guard and not admit_search(before, after):
            after = before.copy()
            details['guard_fallback'] = True
            details['changed_fraction'] = 0.
            details['accepted_steps'] = []
            if stream:
                baseline_path = args.output_dir / f'{i:05d}.baseline'
                shutil.copy2(baseline_path, args.output_dir / f'{i:05d}.searched')
                details['searched_frame_bytes'] = details['baseline_frame_bytes']
        scored['baseline'].append(before); scored['searched'].append(after)
        records.append({'sample_path': row['sample_path'], 'seconds': time.monotonic() - start,
                        **details, 'baseline': before, 'searched': after})
        write_json(args.output_dir / 'records.json', records)
        print(json.dumps({'window': i + 1, 'total': len(selected['val']), 'changed': details['changed_fraction']}), flush=True)
    result = {name: dict(zip(('macro', 'groups'), aggregate_validation(values))) for name, values in scored.items()}
    result['validation_windows'] = len(records)
    result['fixed_rate_only'] = not args.wire_budget
    result['tail_guard_predefined'] = args.tail_guard
    result['guard_during_search'] = args.guard_during_search
    result['guard_fallback_windows'] = sum(r['guard_fallback'] for r in records)
    if args.wire_budget:
        result['baseline_frame_bytes'] = sum(r['baseline_frame_bytes'] for r in records)
        result['searched_frame_bytes'] = sum(r['searched_frame_bytes'] for r in records)
        result['shared_tables_bytes'] = sum((args.output_dir / name).stat().st_size for name in
            ('huffman_table.json', 'range_descriptor.json', 'range_probabilities.f32'))
    write_json(args.output_dir / 'results.json', result)
    (args.output_dir / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
