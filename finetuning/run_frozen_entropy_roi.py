#!/usr/bin/env python3
"""Lossless index coding then deployable encoder-side tile-mode allocation.

No parameter updates, test access, denoising, or discarded retained channels.
Every validation frame is decoded back to identical indices before metrics.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'smartgrids'))
from compression.symbol_stream import SymbolCodec, HEADER
from finetuning.model_factory import build_model
from finetuning.latent_geometry import latent_positions
from finetuning.robust_escape import encode_robust
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.evaluate_checkpoints import metrics
from finetuning.diagnose_reconstruction import spectral_metrics
from finetuning.run_sampling_experiment import check_splits
from finetuning.run_spatial_abc_study import status


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def encoder_indices(model, values):
    tensor = torch.from_numpy(values)[None, None].cuda()
    h, w = values.shape
    multiple = 2 ** model.downsample_levels
    z = model.encoder(F.pad(tensor, (0, (-w) % multiple, 0, (-h) % multiple)))
    _, _, indices, _ = model.quantizer(z)
    result = indices.cpu().numpy().astype(np.uint16)
    assert result.size == latent_positions(model, h, w)
    return result, tuple(z.shape[-2:])


def decoder_waveform(model, indices, shape, latent_shape):
    codes = torch.from_numpy(indices.astype(np.int64)).cuda()
    vectors = model.quantizer.embedding(codes).view(1, *latent_shape, model.embedding_dim)
    vectors = vectors.permute(0, 3, 1, 2).contiguous()
    prediction = model.decoder(vectors)[0, 0, :shape[0], :shape[1]]
    assert tuple(prediction.shape) == tuple(shape)
    return prediction.cpu().numpy()


def select_modes(records, factor, policy, seed=2034):
    """Greedy allocation within a per-source-file byte budget, not an oracle model.

    Distortion measurements are available to a real encoder that has the original.
    The decoder only needs the selected frame's model identifier and symbols.
    """
    if policy not in ('uniform', 'rmse', 'guarded') or not np.isfinite(factor) or factor < 1:
        raise ValueError('Invalid mode allocation policy or byte multiplier')
    baseline = sum(r['compact']['frame_bytes'] for r in records)
    budget = math.floor(baseline * factor)
    choices = {r['sample_id']: 'compact' for r in records}
    used = baseline
    candidates = []
    for row in records:
        compact, dense = row['compact'], row['dense']
        delta = dense['frame_bytes'] - compact['frame_bytes']
        a, b = compact['metrics'], dense['metrics']
        gain = (a['relative_rmse'] ** 2 - b['relative_rmse'] ** 2) * row['input_cells']
        if policy != 'uniform':
            if not np.isfinite(gain) or gain <= 0:
                continue
            if policy == 'guarded' and (b['correlation'] < a['correlation'] or
                    b['ordinary_channel_p90_relative_rmse'] > a['ordinary_channel_p90_relative_rmse']):
                continue
        if policy == 'uniform':
            order = hashlib.sha256(f'{seed}:{row["sample_id"]}'.encode()).hexdigest()
        else:
            order = (-gain / max(delta, 1), row['sample_id'])
        candidates.append((order, delta, row))
    for _, delta, row in sorted(candidates, key=lambda item: item[0]):
        if used + delta <= budget:
            choices[row['sample_id']] = 'dense'
            used += delta
    assert used <= budget
    return choices, {'budget_bytes': budget, 'frame_bytes': used, 'compact_frame_bytes': baseline}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--samples-dir', type=Path, default=ROOT/'data/finetune/spatial_band_v1_fold0_20261024')
    parser.add_argument('--compact-checkpoint', type=Path, default=ROOT/'results/experiments/spatial_abc_20261024_v2/compact_A/best_macro.pt')
    parser.add_argument('--dense-checkpoint', type=Path, default=ROOT/'results/experiments/spatial_abc_20261024_v2/high_A/best_macro.pt')
    parser.add_argument('--test-fold', type=int, choices=range(4), default=0)
    parser.add_argument('--seed', type=int, default=2034)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or '4060' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This frozen experiment must run on the single local RTX 4060')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent / f'.{output.name}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    samples = args.samples_dir.resolve()
    check_splits(samples, args.test_fold)
    with (samples / 'samples_manifest.csv').open() as stream:
        manifest = list(csv.DictReader(stream))
    train = [r for r in manifest if r['split'] == 'train']
    validation = [r for r in manifest if r['split'] == 'val']
    assert train and validation
    for split, values in (('train', train), ('val', validation)):
        assert len(list((samples/split).glob('*.npz'))) == len(values)
        assert {r['family'] for r in values} == {'doped','multicore','multimode','single_mode'}
    validation_folds={int(r['fold']) for r in validation}
    assert len(validation_folds)==1 and args.test_fold not in validation_folds
    assert {r['spatial_band_policy'] for r in train + validation} == {'persistent-amplitude-v1'}
    grouped = defaultdict(list)
    for row in train:
        grouped[row['drive_id'], row['spatial_region_index']].append(row)
    rng = random.Random(args.seed)
    calibration = []
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda r: r['sample_path'])
        calibration.extend(rng.sample(values, min(2, len(values))))
    calibration.sort(key=lambda r: r['sample_path'])
    checkpoints = {'compact':args.compact_checkpoint.resolve(),'dense':args.dense_checkpoint.resolve()}
    files = sorted((ROOT / 'finetuning').glob('*.py')) + sorted((ROOT / 'compression').glob('*.py'))
    files += sorted((ROOT.parent / 'smartgrids/vqvae').glob('*.py'))
    protocol = {'kind': 'frozen_huffman_and_tile_roi_v1', 'seed': args.seed, 'test_fold': args.test_fold,
                'validation_fold': next(iter(validation_folds)), 'calibration_windows': [r['sample_path'] for r in calibration],
                'calibration_policy': 'up to two train windows per file/retained region, fixed seed',
                'checkpoints': {k: {'path': str(v), 'sha256': digest(v)} for k, v in checkpoints.items()},
                'metadata_sha256': {n: digest(samples / n) for n in
                    ('samples_manifest.csv', 'preprocessing.json', 'channel_maps.jsonl')},
                'code_sha256': {str(p): digest(p) for p in files},
                'byte_budgets': [1.5, 2.0], 'policies': ['uniform', 'rmse', 'guarded'],
                'roi_unit': 'existing retained spatial tile and 1000-sample time window',
                'roi_score': 'positive decrease of squared normalized RMSE times cells per extra framed byte',
                'guarded': 'also require per-window correlation and ordinary-channel-p90 no worse than compact',
                'uniform': 'fixed sample-ID hash order, original-independent, same byte ceiling',
                'coding': 'minimum payload of fixed11, train-fitted marginal Huffman, 17-context previous-symbol-bucket Huffman',
                'waveform': 'decode indices and side from real CRC-framed stream, then shared neural decoder',
                'updates': 0, 'no_test_inference': True, 'roi_is_not_residual_refinement': True,
                'software': {'torch': str(torch.__version__), 'numpy': str(np.__version__)},
                'gpu': torch.cuda.get_device_name(0), 'precision': 'FP32; deterministic cuDNN for exact replay'}
    if args.resume:
        assert json.loads((output / 'protocol.json').read_text()) == protocol, 'Frozen protocol changed'
        if (output / 'COMPLETE').exists():
            return
    else:
        output.mkdir(exist_ok=False)
        atomic_json(output / 'protocol.json', protocol)
        for source in files:
            relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('external_vqvae') / source.name
            destination = output / 'code_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    def verify():
        if shutil.disk_usage(output).free < 8 * 2**30:
            raise RuntimeError('Less than 8 GiB free; no automatic file deletion')
        for name, expected in protocol['code_sha256'].items():
            assert digest(name) == expected, name
        for name, expected in protocol['metadata_sha256'].items():
            assert digest(samples / name) == expected, name
    try:
        verify()
        started = time.monotonic()
        models, codecs = {}, {}
        for name, path in checkpoints.items():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert Path(saved['finetune']['samples_dir']).resolve() == samples
            config = {**saved['config'], 'exact_eval_quantization': True}
            models[name] = build_model(config, saved, 'cuda').eval()
            del saved
            table_path = output / f'{name}.tables.json'
            if table_path.exists():
                codecs[name] = SymbolCodec.from_bytes(table_path.read_bytes())
                continue
            status(output, 'fit_train_symbol_tables', mode=name, calibration_windows=len(calibration))
            counts = np.zeros(2048, dtype=np.int64)
            conditional = np.zeros((17, 2048), dtype=np.int64)
            with torch.no_grad():
                for row in calibration:
                    with np.load(ROOT / row['sample_path'], allow_pickle=False) as packet:
                        normalized = encode_robust(np.asarray(packet['data'], dtype=np.float32))
                    indices, _ = encoder_indices(models[name], normalized.values)
                    counts += np.bincount(indices, minlength=2048)
                    contexts = np.concatenate(([16], indices[:-1] // 128))
                    conditional += np.bincount(contexts * 2048 + indices, minlength=17 * 2048).reshape(17, 2048)
            codecs[name] = SymbolCodec.from_counts(counts, conditional, protocol['checkpoints'][name]['sha256'])
            table_path.write_bytes(codecs[name].serialized)
            np.savez_compressed(output / f'{name}.train_counts.npz', counts=counts, conditional=conditional)
        # Explicit physical coordinates and acquisition metadata are charged once.
        with (samples / 'channel_maps.jsonl').open() as stream:
            maps = [json.loads(line) for line in stream]
        selected_maps = {r['channel_map_id'] for r in validation}
        (output / 'channel_maps.jsonl').write_text(''.join(json.dumps(m, separators=(',', ':')) + '\n'
            for m in maps if m['channel_map_id'] in selected_maps))
        with (output / 'validation_metadata.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(validation[0]))
            writer.writeheader(); writer.writerows(validation)
        prep = json.loads((samples / 'preprocessing.json').read_text())
        atomic_json(output / 'preprocessing_policy.json', {k: prep[k] for k in ('spatial_band_policy', 'parameters')})
        metadata_bytes = sum((output / n).stat().st_size for n in
                             ('channel_maps.jsonl', 'validation_metadata.csv', 'preprocessing_policy.json'))
        records = []
        for index, row in enumerate(validation):
            sample_id = Path(row['sample_path']).stem
            cache = output / 'window_cache' / f'{sample_id}.json'
            if cache.exists():
                records.append(json.loads(cache.read_text()))
                continue
            if index % 100 == 0:
                verify()
                status(output, 'encode_decode_full_validation', completed_windows=index,
                       total_windows=len(validation), elapsed_sec=time.monotonic() - started)
            path = ROOT / row['sample_path']
            with np.load(path, allow_pickle=False) as packet:
                raw = np.asarray(packet['data'], dtype=np.float32)
            normalized = encode_robust(raw)
            record = {'sample_id': sample_id, 'drive_id': row['drive_id'], 'family': row['family'],
                      'configuration': row['configuration'], 'input_cells': int(raw.size),
                      'source_npz_sha256': digest(path)}
            with torch.no_grad():
                for name, model in models.items():
                    indices, latent_shape = encoder_indices(model, normalized.values)
                    frame, coder = codecs[name].encode(indices, raw.shape, latent_shape, normalized.side_stream)
                    recovered, shape, latent, side = codecs[name].decode(frame)
                    np.testing.assert_array_equal(indices, recovered)
                    assert side == normalized.side_stream
                    prediction = decoder_waveform(model, recovered, shape, latent)
                    target, restored, extras = restore_for_validation(path, prediction, 'robust-escape-v1', model, True)
                    measured = {**metrics(target[normalized.valid], restored[normalized.valid]),
                                **spectral_metrics(target, restored), **extras}
                    assert np.isfinite(measured['relative_rmse']) and np.isfinite(measured['correlation'])
                    destination = output / 'frames' / name / f'{sample_id}.fsc'
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(frame)
                    record[name] = {'frame_bytes': len(frame), 'payload_bytes': len(frame)-HEADER.size-len(side),
                                    'fixed11_payload_bytes': (len(indices)*11+7)//8,
                                    'side_bytes': len(side), 'coder': coder, 'metrics': measured}
            cache.parent.mkdir(exist_ok=True)
            atomic_json(cache, record)
            records.append(record)
        status(output, 'entropy_and_roi_analysis', windows=len(records))
        cells = sum(r['input_cells'] for r in records)
        by_file = defaultdict(list)
        for record in records:
            by_file[record['drive_id']].append(record)
        results = {}
        weights = {k: sum(p.numel() for p in m.parameters()) * 4 for k, m in models.items()}
        def summarize(label, choices, shared_modes, budget_info=None):
            chosen = [(r, choices[r['sample_id']]) for r in records]
            total_frames = sum(r[mode]['frame_bytes'] for r, mode in chosen)
            tables = sum(len(codecs[k].serialized) for k in shared_modes)
            deployment_weights = sum(weights[k] for k in shared_modes)
            aggregate_rows = [{'family': r['family'], 'configuration': r['configuration'],
                               '_aggregation_weight': float(r['input_cells']), **r[mode]['metrics']} for r, mode in chosen]
            macro, groups = aggregate_validation(aggregate_rows)
            actual = total_frames + tables + metadata_bytes
            result = {**macro, 'windows': len(records), 'input_cells': cells,
                      'frames_bytes': total_frames, 'shared_table_bytes': tables,
                      'geometry_metadata_bytes': metadata_bytes, 'total_stream_bytes': actual,
                      'ratio_with_metadata_shared_models': 4*cells/actual,
                      'shared_float32_weight_bytes': deployment_weights,
                      'ratio_including_float32_weights_once': 4*cells/(actual+deployment_weights),
                      'dense_windows': sum(mode == 'dense' for _, mode in chosen),
                      'dense_cell_fraction': sum(r['input_cells'] for r, mode in chosen if mode == 'dense')/cells}
            if budget_info is not None:
                result['file_budgets'] = budget_info
            atomic_json(output / f'{label}.json', result)
            with (output / f'{label}.groups.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
            # A replayable mode manifest references immutable CRC-framed files.
            (output / f'{label}.modes.json').write_text(json.dumps(choices, sort_keys=True) + '\n')
            result['mode_manifest_bytes'] = (output / f'{label}.modes.json').stat().st_size
            result['note'] = 'Mode-manifest is a redundant catalog; model/table ID already in each frame. Physical coordinates charged in metadata.'
            results[label] = result
            return groups
        base_groups = summarize('compact', {r['sample_id']: 'compact' for r in records}, ['compact'])
        summarize('dense', {r['sample_id']: 'dense' for r in records}, ['dense'])
        guards = {}
        for factor in protocol['byte_budgets']:
            for policy in protocol['policies']:
                label = f'{policy}_{str(factor).replace(".", "p")}'
                choices, budgets = {}, {}
                for file, values in by_file.items():
                    selected, accounting = select_modes(values, factor, policy, seed=args.seed)
                    choices.update(selected);budgets[file] = accounting
                groups = summarize(label, choices, ['compact', 'dense'], budgets)
                reference = {(r['family'], r['configuration']): r for r in base_groups}
                regressions = []
                for row in groups:
                    base = reference[row['family'], row['configuration']]
                    for metric, direction in (('relative_rmse', 1), ('correlation', -1), ('ordinary_channel_p90_relative_rmse', 1)):
                        delta = row[metric] - base[metric]
                        if direction * delta > 1e-8:
                            regressions.append({'family': row['family'], 'configuration': row['configuration'], 'metric': metric, 'delta': delta})
                guards[label] = {'regressions_against_compact': regressions,
                                 'all_descriptive_guards_pass': not regressions,
                                 'final_model_promoted': False}
        entropy = {}
        for name in models:
            fixed = sum(r[name]['fixed11_payload_bytes'] for r in records)
            coded = sum(r[name]['payload_bytes'] for r in records)
            entropy[name] = {'fixed11_payload_bytes': fixed, 'coded_payload_bytes': coded,
                             'payload_reduction_fraction': 1-coded/fixed,
                             'all_indices_and_side_streams_roundtripped': True,
                             'coder_window_counts': {str(k): sum(r[name]['coder']==k for r in records) for k in range(3)}}
        atomic_json(output / 'entropy.json', entropy)
        atomic_json(output / 'results.json', results)
        atomic_json(output / 'guards.json', guards)
        report = ['# Frozen neural index coding and tile ROI', '',
                  'Development-only; no new neural training, test access, or final promotion.', '',
                  '| Mode | RMSE | Correlation | Channel p90 | Ratio including geometry metadata | Dense cells |',
                  '|---|---:|---:|---:|---:|---:|']
        for label, result in results.items():
            report.append(f'| {label} | {result["macro_relative_rmse"]:.6f} | {result["macro_correlation"]:.6f} | {result["macro_ordinary_channel_p90_relative_rmse"]:.6f} | {result["ratio_with_metadata_shared_models"]:.3f} | {result["dense_cell_fraction"]:.3f} |')
        report += ['', 'All transmitted validation frames were decoded to identical indices and normalization side.',
                   'Guarded routing is a predeclared encoder decision using the original, not external validation.',
                   'Models shared across windows; cold-start weight costs reported separately.',
                   'ROI routes whole existing tiles between two neural decoders; it is not residual refinement.',
                   'Per-window/channel and configuration trade-offs remain in cache/groups/guards.',
                   'Inspect metadata cost and model-registry cost before claiming production bitrate savings.']
        (output / 'REPORT.md').write_text('\n'.join(report)+'\n')
        (output / 'COMPLETE').write_text('Frozen coding and predeclared ROI controls complete; no model promotion.\n')
        status(output, 'complete', final_model_promoted=False, windows=len(records), elapsed_sec=time.monotonic()-started)
    except BaseException as error:
        status(output, 'failed', error=repr(error))
        (output / 'FAILED').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
