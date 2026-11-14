"""Explicit-loader wire evaluation for versioned codecs with discrete encode/decode APIs."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from compression.alphabet_stream import AlphabetStream
from compression.study_bundle import write_bundle, read_bundle
from finetuning.evaluate_lr_schedule_study import write_csv
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.representation_study import digest, load_packet, score

ROOT = Path(__file__).resolve().parents[1]


def evaluate_codec(checkpoint, folder, calibration, checks, samples, input_hashes, loader, descriptor):
    folder = Path(folder); folder.mkdir(exist_ok=False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    encoder = loader(config, saved, 'cuda').requires_grad_(False); del saved
    weights = folder / 'codec_weights.pt'
    torch.save({'config': config, 'model_state_dict': {key: value.cpu() for key, value in encoder.state_dict().items()}}, weights)
    weights_hash = digest(weights)
    saved = torch.load(weights, map_location='cpu', weights_only=False)
    receiver = loader(config, saved, 'cuda').requires_grad_(False); del saved
    counts = np.zeros(encoder.num_embeddings, dtype=np.int64)
    started = time.perf_counter()
    with torch.no_grad():
        for row in calibration:
            if row['split'] != 'train' or digest(ROOT / row['sample_path']) != input_hashes[row['sample_path']]:
                raise ValueError('Calibration must contain unchanged training samples')
            _, packet = load_packet(row)
            ids, _ = encoder.encode_indices(torch.from_numpy(packet.values)[None, None].cuda())
            counts += np.bincount(ids.cpu().numpy(), minlength=len(counts))
    stream = AlphabetStream(counts, weights_hash)
    frames = folder / 'frames'; frames.mkdir()
    paths, records, scores, cells, frame_bytes = [], [], [], 0, 0
    with torch.no_grad():
        for index, row in enumerate(checks):
            if digest(ROOT / row['sample_path']) != input_hashes[row['sample_path']]:
                raise ValueError('Evaluation sample changed')
            raw, packet = load_packet(row)
            ids, latent = encoder.encode_indices(torch.from_numpy(packet.values)[None, None].cuda())
            original_ids = ids.cpu().numpy()
            frame = stream.encode(original_ids, raw.shape, latent, packet.side_stream)
            decoded, shape, read_latent, side = stream.decode(frame)
            if not np.array_equal(original_ids, decoded) or shape != raw.shape or read_latent != latent or side != packet.side_stream:
                raise AssertionError('Codec discrete/side roundtrip mismatch')
            predicted = receiver.decode_indices(torch.as_tensor(decoded.astype(np.int64), device='cuda'), read_latent, shape)
            metrics = score(row, raw, packet, predicted[0, 0].cpu().numpy())
            scores.append(metrics); cells += raw.size; frame_bytes += len(frame)
            records.append({'sample_path': row['sample_path'], 'drive_id': row['drive_id'], 'frame_bytes': len(frame), **metrics})
            path = frames / f'{index:05d}.fak'; path.write_bytes(frame); paths.append(path)
    macro, groups = aggregate_validation(scores)
    assets = {name: (samples / name).read_bytes() for name in ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')}
    assets['counts.int64le'] = counts.astype('<i8').tobytes()
    assets['codec_descriptor.json'] = json.dumps(descriptor, sort_keys=True).encode()
    assets['registry.json'] = json.dumps({'model_sha256': weights_hash, 'stream_id': stream.identifier.hex(),
        'frames': [r['sample_path'] for r in checks], 'weights_shared_separately': True}, sort_keys=True).encode()
    bundle = folder / 'checks.fsb'; framing = write_bundle(bundle, assets, paths)
    reader = read_bundle(bundle); restored = next(reader); registry = json.loads(restored['registry.json'])
    independent = AlphabetStream(np.frombuffer(restored['counts.int64le'], dtype='<i8'), registry['model_sha256'])
    number = 0
    for frame in reader:
        independent.decode(frame); number += 1
    if number != len(checks):
        raise AssertionError('Independent bundle count mismatch')
    write_csv(folder / 'groups.csv', groups); write_csv(folder / 'windows.csv', records)
    result = {'macro': macro, 'windows': len(checks), 'input_cells': cells, 'frame_bytes': frame_bytes,
        'complete_stream_bytes': bundle.stat().st_size, 'complete_ratio': 4 * cells / bundle.stat().st_size,
        'shared_model_bytes': weights.stat().st_size, 'model_sha256': weights_hash,
        'shared_tensor_bytes': sum(t.numel() * t.element_size() for t in encoder.state_dict().values()),
        'shared_model_scope': 'Full encoder/quantizer/decoder state, including all basis/center buffers; conservative nonminimal receiver export',
        'including_shared_model_once_ratio': 4 * cells / (bundle.stat().st_size + weights.stat().st_size),
        'independent_bundle_roundtrip': True, 'receiver_without_encoder_features': True,
        'seconds': time.perf_counter() - started, **framing}
    del encoder, receiver
    torch.cuda.empty_cache()
    return result, groups
