#!/usr/bin/env python3
"""Validate paired LR candidates through transmitted, independently replayed VQ frames."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from compression.study_bundle import write_bundle, read_bundle
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import finite_json
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import digest, load_packet, read_manifest, score, write_json
from finetuning.run_index_search import nearest, vectors
from finetuning.run_lr_schedule_study import ARMS


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_regressions(reference, candidate):
    lookup = {(r['family'], r['configuration']): r for r in reference}
    if set(lookup) != {(r['family'], r['configuration']) for r in candidate}:
        raise ValueError('Different configuration sets')
    failures = []
    for row in candidate:
        base = lookup[row['family'], row['configuration']]
        for key, higher_is_better in [('relative_rmse', False), ('correlation', True),
                ('ordinary_channel_p90_relative_rmse', False), ('fft_magnitude_relative_error', False),
                ('fft_phase_alignment', True), ('reconstruction_rms_ratio', False)]:
            before, after = base[key], row[key]
            if key == 'reconstruction_rms_ratio':
                before, after = abs(before - 1), abs(after - 1)
            if not np.isfinite(before) or not np.isfinite(after) or (after < before if higher_is_better else after > before):
                failures.append({'family': row['family'], 'configuration': row['configuration'],
                                 'metric': key, 'before': before, 'after': after})
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one remote RTX4090 required')
    if not (args.study_dir / 'TRAINING_COMPLETE').exists():
        raise ValueError('Complete all training arms before full validation')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    training_protocol = json.loads((args.study_dir / 'protocol.json').read_text())
    if digest(args.source) != training_protocol['source_sha256']:
        raise ValueError('Source checkpoint differs')
    splits = read_manifest(args.samples_dir)
    if digest(args.samples_dir / 'samples_manifest.csv') != training_protocol['manifest_sha256']:
        raise ValueError('Evaluation manifest differs')
    rng = random.Random(2095)
    calibration = []
    for family in ('doped', 'multicore', 'multimode', 'single_mode'):
        candidates = sorted((r for r in splits['train'] if r['family'] == family), key=lambda r: r['sample_path'])
        calibration.extend(rng.sample(candidates, 8))
    validation = sorted(splits['val'], key=lambda r: r['sample_path'])
    candidates = {'source': args.source}
    for arm in ARMS:
        for kind, name in [('selected', 'selected.pt'), ('endpoint', 'last.pt')]:
            candidates[f'{arm}_{kind}'] = args.study_dir / arm / name
    if training_protocol['smoke_only']:
        validation = [max(validation, key=lambda r: int(r['channels']))]
        calibration = [max(calibration, key=lambda r: int(r['channels']))]
    out = args.output_dir or args.study_dir / 'wire_validation'
    protocol = {'kind': 'lr_full_wire_validation_v1', 'training_protocol_sha256': digest(args.study_dir / 'protocol.json'),
                'candidates': {label: digest(path) for label, path in candidates.items()},
                'calibration_train_only': [r['sample_path'] for r in calibration],
                'validation_windows': len(validation),
                'input_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in [*calibration, *validation]},
                'metadata_sha256': {name: digest(args.samples_dir / name) for name in
                                    ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')},
                'source_sha256': digest(Path(__file__)), 'wire': 'FAK1 + FSB1, counts and maps charged, model weights separate',
                'guards': 'Strict configuration RMSE/correlation/p90/magnitude/phase/RMS-distance, no new tolerances',
                'final_holdout_accessed': False, 'smoke_only': training_protocol['smoke_only']}
    if out.exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Evaluation resume protocol differs')
    else:
        out.mkdir()
        write_json(out / 'protocol.json', protocol)
    results = {}
    all_groups = {}
    for label, checkpoint_path in candidates.items():
        result_path = out / f'{label}.json'
        if result_path.exists():
            results[label] = json.loads(result_path.read_text())
            with (out / label / 'groups.csv').open() as stream:
                all_groups[label] = [{k: v if k in ('family', 'configuration') else float(v) for k, v in row.items()}
                                     for row in csv.DictReader(stream)]
            continue
        folder = out / label
        if folder.exists():
            raise ValueError('Partial evaluation preserved; choose a new evaluation directory rather than overwrite frames')
        folder.mkdir()
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = {**saved['config'], 'exact_eval_quantization': True, 'activation_checkpointing': True}
        model = build_model(config, saved, 'cuda').eval()
        weights = folder / 'decoder_model_weights.pt'
        torch.save({'config': config, 'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}, weights)
        weights_hash = digest(weights)
        del saved

        def encode_indices(packet):
            x = torch.from_numpy(packet.values)[None, None].cuda()
            h, w = packet.values.shape
            z = model.encoder(F.pad(x, (0, (-w) % 16, 0, (-h) % 16)))
            indices = nearest(z, model.quantizer.embedding.weight)
            return indices.cpu().numpy().reshape(-1), tuple(z.shape[-2:])

        started = time.perf_counter()
        counts = np.zeros(model.num_embeddings, dtype=np.int64)
        with torch.no_grad():
            for row in calibration:
                _, packet = load_packet(row)
                indices, _ = encode_indices(packet)
                counts += np.bincount(indices.astype(np.int64), minlength=len(counts))
        stream = AlphabetStream(counts, weights_hash)
        frame_paths, records, scores = [], [], []
        cells, frame_bytes = 0, 0
        frames = folder / 'frames'
        frames.mkdir()
        with torch.no_grad():
            for index, row in enumerate(validation):
                if digest(ROOT / row['sample_path']) != protocol['input_sha256'][row['sample_path']]:
                    raise ValueError('Validation array changed')
                raw, packet = load_packet(row)
                indices, latent = encode_indices(packet)
                frame = stream.encode(indices, raw.shape, latent, packet.side_stream)
                path = frames / f'{index:05d}.fak'
                path.write_bytes(frame); frame_paths.append(path)
                decoded, shape, latent_read, side = stream.decode(frame)
                if not np.array_equal(indices, decoded) or shape != raw.shape or latent_read != latent or side != packet.side_stream:
                    raise AssertionError('Frame roundtrip mismatch')
                # The decoder receives only transmitted discrete indices, shape, and side.
                recovered = model.decoder(vectors(torch.as_tensor(decoded.astype(np.int64), device='cuda'),
                    model.quantizer.embedding.weight, latent_read))[..., :shape[0], :shape[1]]
                metrics = score(row, raw, packet, recovered[0, 0].cpu().numpy())
                scores.append(metrics)
                records.append({'sample_path': row['sample_path'], 'drive_id': row['drive_id'],
                                'frame_bytes': len(frame), **metrics})
                frame_bytes += len(frame); cells += raw.size
        macro, groups = aggregate_validation(scores)
        assets = {name: (args.samples_dir / name).read_bytes() for name in
                  ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')}
        assets['counts.int64le'] = counts.astype('<i8').tobytes()
        assets['registry.json'] = json.dumps({'model_sha256': weights_hash, 'stream_id': stream.identifier.hex(),
            'frames': [r['sample_path'] for r in validation], 'weights_shared_separately': True}, sort_keys=True).encode()
        bundle = folder / 'validation.fsb'
        bundle_info = write_bundle(bundle, assets, frame_paths)
        reader = read_bundle(bundle)
        restored_assets = next(reader)
        counts_read = np.frombuffer(restored_assets['counts.int64le'], dtype='<i8')
        registry = json.loads(restored_assets['registry.json'])
        independent_stream = AlphabetStream(counts_read, registry['model_sha256'])
        count = 0
        for frame in reader:
            independent_stream.decode(frame)
            count += 1
        if count != len(frame_paths):
            raise AssertionError('Bundle frame count mismatch')
        result = {'label': label, 'macro': macro, 'windows': len(validation), 'input_cells': cells,
                  'frame_bytes': frame_bytes, 'complete_stream_bytes': bundle.stat().st_size,
                  'complete_ratio': 4 * cells / bundle.stat().st_size,
                  'shared_model_bytes': weights.stat().st_size, 'model_sha256': weights_hash,
                  'including_shared_model_once_ratio': 4 * cells / (bundle.stat().st_size + weights.stat().st_size),
                  'seconds': time.perf_counter() - started, 'independent_bundle_roundtrip': True, **bundle_info}
        write_csv(folder / 'windows.csv', records)
        write_csv(folder / 'groups.csv', groups)
        write_json(result_path, finite_json(result))
        results[label], all_groups[label] = result, groups
        print(f"wire {label}: RMSE={macro['macro_relative_rmse']:.6f} bytes={result['complete_stream_bytes']} ratio={result['complete_ratio']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    comparisons = {}
    for phase in ('selected', 'endpoint'):
        candidate = results[f'cosine_{phase}']
        comparisons[phase] = {}
        for ref in ('source', f'constant_high_{phase}', f'constant_mean_{phase}'):
            reference = results[ref]
            gain = 100 * (1 - candidate['macro']['macro_relative_rmse'] / reference['macro']['macro_relative_rmse'])
            failures = group_regressions(all_groups[ref], all_groups[f'cosine_{phase}'])
            comparisons[phase][ref] = {'rmse_gain_pct': gain,
                'stream_byte_change': candidate['complete_stream_bytes'] - reference['complete_stream_bytes'],
                'configuration_regressions': failures,
                'passes_gain_bytes_guards': gain >= 1 and candidate['complete_stream_bytes'] <= reference['complete_stream_bytes'] and not failures}
    decision = all(comparisons['selected'][ref]['passes_gain_bytes_guards'] for ref in
                   ('constant_high_selected', 'constant_mean_selected'))
    write_json(out / 'results.json', finite_json({'candidates': results, 'comparisons': comparisons,
        'confirmation_eligible': decision, 'model_promoted': False, 'smoke_only': training_protocol['smoke_only']}))
    (out / 'COMPLETE').touch()
    write_json(args.study_dir / 'STATUS.json', {'stage': 'COMPLETE', 'confirmation_eligible': decision,
        'model_promoted': False, 'smoke_only': training_protocol['smoke_only']})
    (args.study_dir / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
