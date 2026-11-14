#!/usr/bin/env python3
"""Full FEBUS validation of pretraining bridges through existing FAK1/FSB1 streams."""
import argparse
import csv
import json
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
from compression.alphabet_stream import AlphabetStream
from compression.study_bundle import write_bundle, read_bundle
from finetuning.evaluate_lr_schedule_study import group_regressions, write_csv
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import finite_json
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import digest, load_packet, read_manifest, score, write_json
from finetuning.run_index_search import nearest, vectors


def evaluate_candidate(path, folder, calibration, validation, samples, input_hashes):
    """No original waveform is available to the discrete neural replay itself."""
    folder.mkdir(exist_ok=False)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = {**saved['config'], 'exact_eval_quantization': True, 'activation_checkpointing': True}
    model = build_model(config, saved, 'cuda').eval()
    weights = folder / 'decoder_model_weights.pt'
    torch.save({'config': config, 'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}, weights)
    weights_hash = digest(weights)
    del saved

    def encode(packet):
        tensor = torch.from_numpy(packet.values)[None, None].cuda()
        h, w = packet.values.shape
        z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
        return nearest(z, model.quantizer.embedding.weight).cpu().numpy(), tuple(z.shape[-2:])

    started = time.perf_counter()
    counts = np.zeros(model.num_embeddings, dtype=np.int64)
    with torch.no_grad():
        for row in calibration:
            if digest(ROOT / row['sample_path']) != input_hashes[row['sample_path']]:
                raise ValueError('Train calibration array changed')
            _, packet = load_packet(row)
            indices, _ = encode(packet)
            counts += np.bincount(indices.astype(np.int64), minlength=len(counts))
    stream = AlphabetStream(counts, weights_hash)
    frames = folder / 'frames'; frames.mkdir()
    paths, records, scores, frame_bytes, cells = [], [], [], 0, 0
    with torch.no_grad():
        for index, row in enumerate(validation):
            if digest(ROOT / row['sample_path']) != input_hashes[row['sample_path']]:
                raise ValueError('Validation array changed')
            raw, packet = load_packet(row)
            indices, latent = encode(packet)
            frame = stream.encode(indices, raw.shape, latent, packet.side_stream)
            decoded, shape, decoded_latent, side = stream.decode(frame)
            if not np.array_equal(decoded, indices) or shape != raw.shape or latent != decoded_latent or side != packet.side_stream:
                raise AssertionError('FAK1 index/shape/side roundtrip mismatch')
            reconstruction = model.decoder(vectors(torch.as_tensor(decoded.astype(np.int64), device='cuda'),
                model.quantizer.embedding.weight, decoded_latent))[..., :shape[0], :shape[1]]
            metrics = score(row, raw, packet, reconstruction[0, 0].cpu().numpy())
            path = frames / f'{index:05d}.fak'; path.write_bytes(frame); paths.append(path)
            scores.append(metrics)
            records.append({'sample_path': row['sample_path'], 'drive_id': row['drive_id'],
                            'frame_bytes': len(frame), **metrics})
            frame_bytes += len(frame); cells += raw.size
    macro, groups = aggregate_validation(scores)
    assets = {name: (samples / name).read_bytes() for name in
              ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')}
    assets['counts.int64le'] = counts.astype('<i8').tobytes()
    assets['registry.json'] = json.dumps({'model_sha256': weights_hash,
        'stream_id': stream.identifier.hex(), 'frames': [r['sample_path'] for r in validation],
        'weights_shared_separately': True}, sort_keys=True).encode()
    bundle = folder / 'validation.fsb'
    framing = write_bundle(bundle, assets, paths)
    reader = read_bundle(bundle)
    restored = next(reader)
    registry = json.loads(restored['registry.json'])
    independent = AlphabetStream(np.frombuffer(restored['counts.int64le'], dtype='<i8'), registry['model_sha256'])
    count = 0
    for frame in reader:
        independent.decode(frame); count += 1
    if count != len(validation):
        raise AssertionError('FSB1 frame count mismatch')
    write_csv(folder / 'groups.csv', groups)
    write_csv(folder / 'windows.csv', records)
    result = {'macro': macro, 'windows': len(validation), 'input_cells': cells,
              'frame_bytes': frame_bytes, 'complete_stream_bytes': bundle.stat().st_size,
              'complete_ratio': 4 * cells / bundle.stat().st_size, 'shared_model_bytes': weights.stat().st_size,
              'including_shared_model_once_ratio': 4 * cells / (bundle.stat().st_size + weights.stat().st_size),
              'model_sha256': weights_hash, 'independent_bundle_roundtrip': True,
              'seconds': time.perf_counter() - started, **framing}
    del model
    torch.cuda.empty_cache()
    return result, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090')
    if not (args.study_dir / 'TRAINING_COMPLETE').exists():
        raise ValueError('Both complete training arms required before full validation')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    training = json.loads((args.study_dir / 'protocol.json').read_text())
    splits = read_manifest(args.samples_dir)
    if digest(args.samples_dir / 'samples_manifest.csv') != training['manifest_sha256']:
        raise ValueError('Target manifest changed')
    candidates = {'initial': args.study_dir / 'initial.pt'}
    for arm in ('febus', 'quantx'):
        candidates[f'{arm}_selected'] = args.study_dir / arm / 'selected.pt'
        candidates[f'{arm}_endpoint'] = args.study_dir / arm / 'last.pt'
    if digest(candidates['initial']) != training['source_sha256']:
        raise ValueError('Offshore compact initialization changed')
    rng, calibration = random.Random(2095), []
    for family in ('doped', 'multicore', 'multimode', 'single_mode'):
        available = sorted((r for r in splits['train'] if r['family'] == family), key=lambda r: r['sample_path'])
        calibration += rng.sample(available, 8)
    validation = sorted(splits['val'], key=lambda r: r['sample_path'])
    if training['smoke_only']:
        validation = [max(validation, key=lambda r: int(r['channels']))]
        calibration = [max(calibration, key=lambda r: int(r['channels']))]
    out = args.output_dir or args.study_dir / 'wire_validation'
    protocol = {'kind': 'pretraining_bridge_wire_validation_v1',
                'training_protocol_sha256': digest(args.study_dir / 'protocol.json'),
                'candidates': {label: digest(path) for label, path in candidates.items()},
                'calibration_train_only': [r['sample_path'] for r in calibration], 'validation_windows': len(validation),
                'input_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in calibration + validation},
                'metadata_sha256': {name: digest(args.samples_dir / name) for name in
                                    ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')},
                'code_sha256': {name: digest(ROOT / name) for name in
                    ('finetuning/evaluate_pretraining_bridge.py', 'compression/alphabet_stream.py',
                     'compression/study_bundle.py', 'finetuning/evaluate_lr_schedule_study.py')},
                'wire': 'Same FAK1/FSB1 and train-only calibration as LR evaluation; shared weights separate',
                'final_holdout_accessed': False, 'smoke_only': training['smoke_only']}
    if out.exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Evaluation resume protocol differs')
    else:
        out.mkdir()
        write_json(out / 'protocol.json', protocol)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
    results, groups = {}, {}
    for label, path in candidates.items():
        result_path = out / f'{label}.json'
        if result_path.exists():
            results[label] = json.loads(result_path.read_text())
            with (out / label / 'groups.csv').open() as stream:
                groups[label] = [{k: v if k in ('family', 'configuration') else float(v)
                                  for k, v in row.items()} for row in csv.DictReader(stream)]
            continue
        results[label], groups[label] = evaluate_candidate(path, out / label, calibration, validation,
                                                            args.samples_dir, protocol['input_sha256'])
        write_json(result_path, finite_json(results[label]))
        print(f"bridge-wire {label}: RMSE={results[label]['macro']['macro_relative_rmse']:.6f} bytes={results[label]['complete_stream_bytes']}", flush=True)
    comparisons = {}
    for phase in ('selected', 'endpoint'):
        candidate, reference = results[f'quantx_{phase}'], results[f'febus_{phase}']
        gain = 100 * (1 - candidate['macro']['macro_relative_rmse'] / reference['macro']['macro_relative_rmse'])
        failures = group_regressions(groups[f'febus_{phase}'], groups[f'quantx_{phase}'])
        comparisons[phase] = {'rmse_gain_pct': gain,
            'stream_byte_change': candidate['complete_stream_bytes'] - reference['complete_stream_bytes'],
            'configuration_regressions': failures,
            'passes_investment_gate': gain >= 5 and candidate['complete_stream_bytes'] <= reference['complete_stream_bytes'] and not failures}
    write_json(out / 'results.json', finite_json({'candidates': results, 'comparisons': comparisons,
        'confirmation_eligible': comparisons['selected']['passes_investment_gate'], 'model_promoted': False,
        'smoke_only': training['smoke_only']}))
    (out / 'COMPLETE').touch()
    write_json(args.study_dir / 'STATUS.json', {'stage': 'COMPLETE', 'model_promoted': False,
        'confirmation_eligible': comparisons['selected']['passes_investment_gate'], 'smoke_only': training['smoke_only']})
    (args.study_dir / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
