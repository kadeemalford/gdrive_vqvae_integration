#!/usr/bin/env python3
"""Frozen full-zone normalization partitions; no row is removed from the bundle."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.affine_restoration import restore
from compression.numeric_restoration import encode_numeric
from compression.partition_bundle import pack_bundle, decode_bundle, exact_nonfinite
from finetuning.affine_codec_study import score_restored
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, write_json
from finetuning.run_external_smf_stress import pooled_score
from finetuning.run_index_search import FrozenIndexStream, nearest, vectors
from finetuning.run_physical_guard_search import physical_admit


def normalization_partitions(raw, removed):
    status = np.zeros(raw.shape[0], dtype=np.uint8)
    status[np.asarray(removed, dtype=int)] = 1
    status[~np.isfinite(raw).any(axis=1)] = 2
    cuts = [0, *(np.flatnonzero(np.diff(status.astype(int))) + 1).tolist(), len(status)]
    parts = []
    for begin, end in zip(cuts[:-1], cuts[1:]):
        for start in range(begin, end, 512):
            parts.append((start, min(start + 512, end), int(status[start])))
    assert sum(end - begin for begin, end, _ in parts) == raw.shape[0]
    return parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-experiment', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--compact', type=Path, required=True)
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = json.loads((args.baseline_experiment / 'protocol.json').read_text())
    decisions_path = args.baseline_experiment / 'coverage_seam_diagnostic.json'
    decisions = {r['drive_id']: r['fixed_policy_decision'] for r in json.loads(decisions_path.read_text())['files']}
    samples = json.loads((args.samples_dir / 'samples.json').read_text())
    hashes = {a: digest(getattr(args, a)) for a in ('compact', 'dense')}
    assert hashes == baseline['source_checkpoint_sha256']
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_full_coverage_partition.py', 'compression/partition_bundle.py',
            'compression/numeric_restoration.py', 'compression/affine_restoration.py',
            'finetuning/affine_codec_study.py', 'finetuning/run_physical_guard_search.py', 'finetuning/run_index_search.py']
    write_json(out / 'protocol.json', {'kind': 'full_zone_frozen_partition_v1', 'parameter_updates': 0,
        'gpu': torch.cuda.get_device_name(), 'source_checkpoint_sha256': hashes,
        'baseline_results_sha256': digest(args.baseline_experiment / 'results.json'),
        'fixed_decision_sha256': digest(decisions_path), 'code_sha256': {s: digest(ROOT / s) for s in code},
        'coverage': 'all 551 spatial rows x 20000 samples per six SMF files, no dropped amplitude bands or nonfinite rows',
        'partition': 'fixed saved amplitude boundaries define separate contiguous normalization domains, not exclusion; wholly nonfinite rows exact-only',
        'models': 'same frozen weights, same old-train symbol tables, no affine gain search or external fitting of shared parameters',
        'wire': 'FPB1 bundles cover every row exactly once; neural FNP fragments and exact-only nonfinite zlib-6 raw-bit fragments; all routing/CRC bytes charged',
        'control': 'same FPB1 container wrapping the original full-zone 512+39 frames',
        'symbol_budget': 'assert equal neural index count before admitting partition candidates',
        'guard': 'per full 551x1000 bundle nonincreasing total bytes/RMSE/p90/FFT magnitude/RMS-distance, nondecreasing correlation/phase',
        'fallback': 'original bundle always eligible; original/routed/candidate results all retained',
        'investment_gate': '>=5% guarded RMSE gain at nonincreasing bytes, concurrent file/pooled metrics inspected; no promotion from one linked external-development group',
        'limitations': ['external DEVELOPMENT reuse, not fresh confirmation',
            'guard scope is full-window bundle, not independent original 512/39 tile guards; regional metrics remain audited',
            'joint context/normalization/padding and numeric framing change; rate decomposition required',
            'amplitude heuristic does not prove a band invalid; every band is coded',
            'no four-family acceptance or independent-acquisition claim']})
    for name in code:
        dest = out / 'code_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    summaries, all_files, transport = {}, [], []
    with torch.no_grad():
        for arm in ('compact', 'dense'):
            saved = torch.load(getattr(args, arm), map_location='cpu', weights_only=False)
            model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
            del saved
            stream = FrozenIndexStream(np.load(args.baseline_experiment / arm / 'counts.npy', allow_pickle=False), hashes[arm])
            folder = out / arm
            folder.mkdir()
            np.save(folder / 'counts.npy', np.load(args.baseline_experiment / arm / 'counts.npy', allow_pickle=False))
            write_json(folder / 'descriptor.json', {'format': 'FPB1', 'source_checkpoint_sha256': hashes[arm],
                'counts_sha256': digest(folder / 'counts.npy'), 'exact_type': 'only all-nonfinite raw little-endian float32 bits, zlib-6',
                'new_waveform_weights_bytes': 0})
            scores, totals, records = {k: [] for k in ('source', 'candidate', 'guarded')}, {k: 0 for k in ('source', 'candidate', 'guarded')}, []
            for index, sample in enumerate(samples):
                assert digest(args.samples_dir / sample['path']) == sample['sha256']
                with np.load(args.samples_dir / sample['path'], allow_pickle=False) as data:
                    raw_full = data['data']
                with np.load(args.baseline_experiment / arm / f'f{index:02d}.direct.reconstruction.npz', allow_pickle=False) as data:
                    original_full = data['data']
                parts = normalization_partitions(raw_full, decisions[sample['drive_id']]['removed_source_channel_indices'])
                outputs = {'source': original_full, 'candidate': np.empty_like(raw_full), 'guarded': np.empty_like(raw_full)}
                valid_full = np.zeros(raw_full.shape, dtype=bool)
                for window in range(20):
                    raw = raw_full[:, window * 1000:(window + 1) * 1000]
                    packet = encode_numeric(raw)
                    valid_full[:, window * 1000:(window + 1) * 1000] = packet.valid
                    row = {'family': 'single_mode', 'configuration': 'External SMF/' + sample['orientation']}
                    old, source_positions = [], 0
                    for region, begin in enumerate((0, 512)):
                        frame = (args.baseline_experiment / arm / f'f{index:02d}_w{window:02d}_r{region}.direct.frame').read_bytes()
                        symbols, shape, _, _ = stream.decode(frame)
                        old.append((0, begin, shape[0], frame))
                        source_positions += len(symbols)
                    source_bundle = pack_bundle(raw.shape, old)
                    fragments, candidate_positions, exact_bytes = [], 0, 0
                    candidate = np.empty_like(raw)
                    for begin, end, category in parts:
                        target_raw = raw[begin:end].copy()
                        if not np.isfinite(target_raw).any():
                            payload = exact_nonfinite(target_raw)
                            fragments.append((1, begin, end - begin, payload))
                            candidate.view(np.uint32)[begin:end] = target_raw.view(np.uint32)
                            exact_bytes += len(payload)
                        else:
                            target_packet = encode_numeric(target_raw)
                            tensor = torch.from_numpy(target_packet.values)[None, None].cuda()
                            h, w = target_raw.shape
                            z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
                            symbols = nearest(z, model.quantizer.embedding.weight)
                            latent = tuple(z.shape[-2:])
                            decoded = model.decoder(vectors(symbols, model.quantizer.embedding.weight, latent))[..., :h, :w][0, 0].cpu().numpy()
                            physical = restore(decoded, target_packet.side_stream)
                            payload = stream.encode(symbols, target_raw.shape, latent, target_packet.side_stream)
                            fragments.append((0, begin, end - begin, payload))
                            candidate.view(np.uint32)[begin:end] = physical.view(np.uint32)
                            candidate_positions += int(symbols.numel())
                    assert candidate_positions == source_positions, 'Different latent count is outside this contrast'
                    candidate_bundle = pack_bundle(raw.shape, fragments)
                    replay = decode_bundle(candidate_bundle, stream, model, 'cuda')
                    np.testing.assert_array_equal(replay.view(np.uint32), candidate.view(np.uint32))
                    before = score_restored(row, raw, packet, original_full[:, window * 1000:(window + 1) * 1000])
                    after = score_restored(row, raw, packet, candidate)
                    quality = physical_admit(before, after)
                    budget = len(candidate_bundle) <= len(source_bundle)
                    admitted = quality and budget
                    chosen, chosen_bundle = (candidate, candidate_bundle) if admitted else (original_full[:, window * 1000:(window + 1) * 1000], source_bundle)
                    if not admitted:
                        replay = decode_bundle(source_bundle, stream, model, 'cuda')
                        np.testing.assert_array_equal(replay.view(np.uint32), chosen.copy().view(np.uint32))
                    for label, bundle, metrics in [('source', source_bundle, before), ('candidate', candidate_bundle, after),
                                                  ('guarded', chosen_bundle, after if admitted else before)]:
                        (folder / f'f{index:02d}_w{window:02d}.{label}.fpb').write_bytes(bundle)
                        totals[label] += len(bundle)
                        scores[label].append(metrics)
                    outputs['candidate'][:, window * 1000:(window + 1) * 1000] = candidate
                    outputs['guarded'][:, window * 1000:(window + 1) * 1000] = chosen
                    records.append({'file_index': index, 'window': window, 'source': before, 'candidate': after,
                        'guarded': after if admitted else before, 'admitted': bool(admitted), 'quality_eligible': bool(quality), 'budget_eligible': bool(budget),
                        'source_bytes': len(source_bundle), 'candidate_bytes': len(candidate_bundle),
                        'neural_indices_source': source_positions, 'neural_indices_candidate': candidate_positions,
                        'exact_fragment_bytes': exact_bytes, 'source_fragments': len(old), 'candidate_fragments': len(fragments)})
                    if arm == 'compact':
                        transport.append({'file_index': index, 'window': window, 'drive_id': sample['drive_id'],
                            'dt_s': sample['dt_s'], 'dx_m': sample['dx_m'], 'start_position_m': sample['start_position_m'],
                            'clock_start_s': sample['clock_interval_s'][0] + window * .05})
                for label, output in outputs.items():
                    result = pooled_score(raw_full, output, valid_full)
                    all_files.append({'arm': arm, 'policy': label, 'drive_id': sample['drive_id'], 'orientation': sample['orientation'],
                        'stimulus_label_hz': sample['stimulus_label_hz'], 'pooled_1s': result, 'partitions': parts})
                    if label != 'source':
                        np.savez_compressed(folder / f'f{index:02d}.{label}.reconstruction.npz', data=output)
                print(json.dumps({'arm': arm, 'completed_files': index + 1}), flush=True)
            write_json(folder / 'records.json', records)
            summaries[arm] = {}
            for label, values in scores.items():
                macro, groups = aggregate_metrics(values)
                summaries[arm][label] = {'macro_full_window': macro, 'groups': groups, 'bundle_bytes': totals[label]}
            summaries[arm]['admitted_windows'] = sum(r['admitted'] for r in records)
            summaries[arm]['quality_eligible_windows'] = sum(r['quality_eligible'] for r in records)
            summaries[arm]['budget_eligible_windows'] = sum(r['budget_eligible'] for r in records)
            del model
            torch.cuda.empty_cache()
    write_json(out / 'transport.json', transport)
    result = {'parameter_updates': 0, 'input_cells_per_model': 6 * 551 * 20000, 'arms': summaries,
              'pooled_1s_files': all_files, 'full_coverage_preserved': True,
              'scope': 'same 551 rows, equal latent symbol counts; external DEVELOPMENT reuse, not promotion'}
    for arm in ('compact', 'dense'):
        auxiliary = (out / 'transport.json').stat().st_size + (out / arm / 'counts.npy').stat().st_size + (out / arm / 'descriptor.json').stat().st_size
        for label in ('source', 'candidate', 'guarded'):
            summaries[arm][label]['stream_bytes'] = summaries[arm][label]['bundle_bytes'] + auxiliary
        summaries[arm]['guarded_rmse_gain_vs_source'] = 1 - summaries[arm]['guarded']['macro_full_window']['macro_relative_rmse'] / summaries[arm]['source']['macro_full_window']['macro_relative_rmse']
        assert digest(getattr(args, arm)) == hashes[arm]
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({a: {'gain': v['guarded_rmse_gain_vs_source'], 'accepted': v['admitted_windows'],
                         'source_bytes': v['source']['bundle_bytes'], 'selected_bytes': v['guarded']['bundle_bytes']} for a, v in summaries.items()}), flush=True)


if __name__ == '__main__':
    main()
