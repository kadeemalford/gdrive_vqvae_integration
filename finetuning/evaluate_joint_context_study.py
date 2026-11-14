#!/usr/bin/env python3
"""Full discrete replay and wire accounting for jointly trained context codecs."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from compression.study_bundle import write_bundle, read_bundle
from finetuning.evaluate_lr_schedule_study import group_regressions, write_csv
from finetuning.joint_context_vq import ARMS, build_joint_context
from finetuning.paired_search_diagnostics import finite_json
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import digest, load_packet, read_manifest, score, write_json


def groups_from(path):
    with path.open() as stream:
        return [{k: v if k in ('family', 'configuration') else float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def evaluate_candidate(path, folder, calibration, validation, samples, inputs):
    folder.mkdir(exist_ok=False)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = saved['config']
    model = build_joint_context(config, saved, 'cuda').requires_grad_(False)
    weights = folder / 'decoder_model_weights.pt'
    torch.save({'config': config, 'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}, weights)
    weights_hash = digest(weights); del saved
    receiver_state = torch.load(weights, map_location='cpu', weights_only=False)
    receiver = build_joint_context(config, receiver_state, 'cuda').requires_grad_(False); del receiver_state
    counts = np.zeros(model.num_embeddings, dtype=np.int64)
    started = time.perf_counter()
    with torch.no_grad():
        for row in calibration:
            if digest(ROOT / row['sample_path']) != inputs[row['sample_path']]:
                raise ValueError('Train-only calibration sample changed')
            _, packet = load_packet(row)
            indices, _ = model.encode_indices(torch.from_numpy(packet.values)[None, None].cuda())
            counts += np.bincount(indices.cpu().numpy(), minlength=len(counts))
    stream = AlphabetStream(counts, weights_hash)
    frames = folder / 'frames'; frames.mkdir()
    paths, records, scores, cells, frame_bytes = [], [], [], 0, 0
    with torch.no_grad():
        for index, row in enumerate(validation):
            if digest(ROOT / row['sample_path']) != inputs[row['sample_path']]:
                raise ValueError('Validation sample changed')
            raw, packet = load_packet(row)
            indices, latent = model.encode_indices(torch.from_numpy(packet.values)[None, None].cuda())
            source_ids = indices.cpu().numpy()
            frame = stream.encode(source_ids, raw.shape, latent, packet.side_stream)
            recovered_ids, shape, recovered_latent, side = stream.decode(frame)
            if not np.array_equal(source_ids, recovered_ids) or shape != raw.shape or recovered_latent != latent or side != packet.side_stream:
                raise AssertionError('Joint FAK1 roundtrip mismatch')
            predicted = receiver.decode_indices(torch.as_tensor(recovered_ids.astype(np.int64), device='cuda'), recovered_latent, shape)
            metrics = score(row, raw, packet, predicted[0, 0].cpu().numpy())
            scores.append(metrics); cells += raw.size; frame_bytes += len(frame)
            records.append({'sample_path': row['sample_path'], 'drive_id': row['drive_id'], 'frame_bytes': len(frame), **metrics})
            frame_path = frames / f'{index:05d}.fak'; frame_path.write_bytes(frame); paths.append(frame_path)
    macro, groups = aggregate_validation(scores)
    assets = {name: (samples / name).read_bytes() for name in ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')}
    assets['counts.int64le'] = counts.astype('<i8').tobytes()
    assets['registry.json'] = json.dumps({'model_sha256': weights_hash, 'stream_id': stream.identifier.hex(),
        'frames': [row['sample_path'] for row in validation], 'weights_shared_separately': True}, sort_keys=True).encode()
    bundle = folder / 'validation.fsb'; info = write_bundle(bundle, assets, paths)
    reader = read_bundle(bundle); restored = next(reader); registry = json.loads(restored['registry.json'])
    independent = AlphabetStream(np.frombuffer(restored['counts.int64le'], dtype='<i8'), registry['model_sha256'])
    count = 0
    for frame in reader:
        independent.decode(frame); count += 1
    if count != len(validation):
        raise AssertionError('Independent joint bundle frame count differs')
    write_csv(folder / 'groups.csv', groups); write_csv(folder / 'windows.csv', records)
    result = {'macro': macro, 'windows': len(validation), 'input_cells': cells, 'frame_bytes': frame_bytes,
        'complete_stream_bytes': bundle.stat().st_size, 'complete_ratio': 4 * cells / bundle.stat().st_size,
        'shared_model_bytes': weights.stat().st_size, 'model_sha256': weights_hash,
        'including_shared_model_once_ratio': 4 * cells / (bundle.stat().st_size + weights.stat().st_size),
        'shared_model_scope': 'Full codec weights charged conservatively, including encoder, as in historical reference; not minimal receiver export',
        'encoder_context_tensor_bytes': sum(p.numel() * p.element_size() for p in model.analysis_context.parameters()),
        'decoder_context_tensor_bytes': sum(p.numel() * p.element_size() for p in model.synthesis_context.parameters()),
        'receiver_required_tensor_bytes': sum(p.numel() * p.element_size() for component in
            (model.base.decoder, model.base.quantizer, model.synthesis_context) for p in component.parameters()),
        'receiver_inputs': 'Only transmitted discrete IDs, latent/original shapes and original reversible side; separately loaded shared weights',
        'independent_bundle_roundtrip': True, 'seconds': time.perf_counter() - started, **info}
    del model, receiver
    torch.cuda.empty_cache()
    return result, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090')
    if not (args.study_dir / 'TRAINING_COMPLETE').exists():
        raise ValueError('Complete both joint-context arms first')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.backends.mha.set_fastpath_enabled(False)
    sys.path.insert(0, str(args.repo_root.resolve()))
    lock = (args.study_dir.parent / '.lr_schedule_gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    training = json.loads((args.study_dir / 'protocol.json').read_text())
    for name, expected in training['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen joint/replay code changed: {name}')
    for name, expected in training['vendor_sha256'].items():
        if digest(args.repo_root / 'vqvae' / name) != expected:
            raise ValueError('Frozen vendor changed')
    audit = training['control_audit']; old_wire = Path(audit['study']) / 'wire_validation'
    if digest(old_wire / 'protocol.json') != audit['wire_protocol_sha256']:
        raise ValueError('Native control wire protocol changed')
    old_protocol = json.loads((old_wire / 'protocol.json').read_text())
    for artifacts in audit['wire_artifacts'].values():
        for name, expected in artifacts.items():
            if digest(old_wire / name) != expected:
                raise ValueError('Native control wire artifact changed')
    samples = Path(training['samples_dir']); rows = read_manifest(samples)
    if digest(samples / 'samples_manifest.csv') != training['manifest_sha256']:
        raise ValueError('Target corpus differs')
    rng, calibration = random.Random(2095), []
    for family in ('doped', 'multicore', 'multimode', 'single_mode'):
        candidates = sorted((r for r in rows['train'] if r['family'] == family), key=lambda row: row['sample_path'])
        calibration += rng.sample(candidates, 8)
    validation = sorted(rows['val'], key=lambda row: row['sample_path'])
    inputs = {r['sample_path']: digest(ROOT / r['sample_path']) for r in calibration + validation}
    if inputs != old_protocol['input_sha256'] or [r['sample_path'] for r in calibration] != old_protocol['calibration_train_only']:
        raise ValueError('Calibration/validation inputs differ from native reference')
    for name, expected in old_protocol['metadata_sha256'].items():
        if digest(samples / name) != expected:
            raise ValueError('Target metadata differs')
    if training['smoke_only']:
        calibration = [max(calibration, key=lambda row: int(row['channels']))]
        validation = [max(validation, key=lambda row: int(row['channels']))]
    candidates = {f'{kind}_{phase}': args.study_dir / kind / name for kind in ARMS
                  for phase, name in [('selected', 'selected.pt'), ('endpoint', 'last.pt')]}
    protocol = {'kind': 'joint_context_full_wire_v1', 'training_protocol_sha256': digest(args.study_dir / 'protocol.json'),
        'control_audit': audit, 'candidates': {k: digest(v) for k, v in candidates.items()},
        'input_sha256': inputs, 'metadata_sha256': old_protocol['metadata_sha256'],
        'validation_windows': len(validation), 'calibration_train_only': [r['sample_path'] for r in calibration],
        'code_sha256': training['code_sha256'], 'smoke_only': training['smoke_only'], 'final_holdout_accessed': False}
    out = args.output_dir or args.study_dir / 'wire_validation'
    if out.exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Joint wire resume differs')
    else:
        out.mkdir(); write_json(out / 'protocol.json', protocol)
    results, groups = {}, {}
    if not training['smoke_only']:
        for phase in ('selected', 'endpoint'):
            label = f'constant_high_{phase}'
            results[f'native_{phase}'] = json.loads((old_wire / f'{label}.json').read_text())
            groups[f'native_{phase}'] = groups_from(old_wire / label / 'groups.csv')
    for label, path in candidates.items():
        result_path = out / f'{label}.json'
        if result_path.exists():
            results[label] = json.loads(result_path.read_text()); groups[label] = groups_from(out / label / 'groups.csv')
        else:
            results[label], groups[label] = evaluate_candidate(path, out / label, calibration, validation, samples, inputs)
            write_json(result_path, finite_json(results[label]))
        print(f"joint-wire {label} RMSE={results[label]['macro']['macro_relative_rmse']:.6f} bytes={results[label]['complete_stream_bytes']}", flush=True)
    comparisons = {}
    for phase in ('selected', 'endpoint'):
        comparisons[phase] = {}
        for reference in (('pointwise',) if training['smoke_only'] else ('pointwise', 'native')):
            before, after = results[f'{reference}_{phase}'], results[f'axial_{phase}']
            gain = 100 * (1 - after['macro']['macro_relative_rmse'] / before['macro']['macro_relative_rmse'])
            delta = after['complete_stream_bytes'] - before['complete_stream_bytes']
            failures = group_regressions(groups[f'{reference}_{phase}'], groups[f'axial_{phase}'])
            comparisons[phase][reference] = {'rmse_gain_pct': gain, 'stream_byte_change': delta,
                'configuration_regressions': failures,
                'passes_investment_gate': not training['smoke_only'] and gain >= 5 and delta <= 0 and not failures}
    eligible = not training['smoke_only'] and all(v['passes_investment_gate'] for v in comparisons['selected'].values())
    write_json(out / 'results.json', finite_json({'candidates': results, 'comparisons': comparisons,
        'confirmation_eligible': eligible, 'model_promoted': False, 'smoke_only': training['smoke_only']}))
    (out / 'COMPLETE').touch(); (args.study_dir / 'COMPLETE').touch()
    write_json(args.study_dir / 'STATUS.json', {'stage': 'COMPLETE', 'confirmation_eligible': eligible,
                                              'model_promoted': False, 'smoke_only': training['smoke_only']})
    lock.close()


if __name__ == '__main__':
    main()
