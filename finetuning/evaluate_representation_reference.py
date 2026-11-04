#!/usr/bin/env python3
"""Frozen compact VQ reference in the same explicit map/transport accounting as E2."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.run_index_search import FrozenIndexStream, nearest, vectors
from finetuning.model_factory import build_model
from finetuning.representation_study import (digest, load_packet, pilot_rows, read_manifest,
                                             score, tensor_packet, write_json)
from finetuning.reconstruction_objective import aggregate_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    splits = read_manifest(args.samples_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model({**checkpoint['config'], 'exact_eval_quantization': True}, checkpoint, 'cuda').eval()
    counts = np.zeros(2048, dtype=np.int64)
    def analyze(packet):
        tensor, _ = tensor_packet(packet)
        h, w = packet.values.shape
        z = model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
        indices = nearest(z, model.quantizer.embedding.weight)
        return indices, z.shape[-2:]
    with torch.no_grad():
        for row in pilot_rows(splits['train']):
            _, packet = load_packet(row)
            indices, _ = analyze(packet)
            counts += np.bincount(indices.cpu().numpy(), minlength=2048)
        stream = FrozenIndexStream(counts, digest(args.checkpoint))
        (output / 'huffman_table.json').write_bytes(stream.huffman.serialized)
        (output / 'range_descriptor.json').write_bytes(stream.codec.serialized)
        (output / 'range_probabilities.f32').write_bytes(stream.probabilities.tobytes())
        records, rows, transport = [], [], []
        cells = frame_bytes = 0
        for i, row in enumerate(splits['val']):
            raw, packet = load_packet(row)
            indices, latent = analyze(packet)
            frame = stream.encode(indices, raw.shape, latent, packet.side_stream)
            recovered, shape, recovered_latent, side = stream.decode(frame)
            np.testing.assert_array_equal(recovered, indices.cpu().numpy())
            assert shape == raw.shape and tuple(recovered_latent) == tuple(latent) and side == packet.side_stream
            prediction = model.decoder(vectors(torch.from_numpy(recovered.astype(np.int64)).cuda(),
                model.quantizer.embedding.weight, latent))[0, 0, :raw.shape[0], :raw.shape[1]].cpu().numpy()
            measured = score(row, raw, packet, prediction)
            rows.append(measured)
            filename = f'{i:05d}.fnp'
            (output / filename).write_bytes(frame)
            transport.append({'frame': filename, 'sample_path': row['sample_path'], 'drive_id': row['drive_id'],
                'block_index': int(row['block_index']), 'time_start': int(row['time_start']),
                'channel_map_id': row['channel_map_id']})
            records.append({'sample_path': row['sample_path'], 'metrics': measured, 'frame_bytes': len(frame)})
            cells += raw.size; frame_bytes += len(frame)
            if (i + 1) % 512 == 0:
                print(json.dumps({'windows': i + 1, 'total': len(splits['val'])}), flush=True)
    maps = [json.loads(line) for line in (args.samples_dir / 'channel_maps.jsonl').read_text().splitlines()]
    identifiers = {r['channel_map_id'] for r in splits['val']}
    write_json(output / 'maps.json', [m for m in maps if m['channel_map_id'] in identifiers])
    write_json(output / 'transport.json', transport)
    auxiliary = sum((output / name).stat().st_size for name in
        ('maps.json', 'transport.json', 'huffman_table.json', 'range_descriptor.json', 'range_probabilities.f32'))
    macro, groups = aggregate_validation(rows)
    write_json(output / 'records.json', records)
    write_json(output / 'results.json', {'windows': len(rows), 'input_cells': cells,
        'frame_bytes': frame_bytes, 'shared_tables_maps_transport_bytes': auxiliary,
        'total_stream_bytes': frame_bytes + auxiliary, 'total_bpc': 8 * (frame_bytes + auxiliary) / cells,
        'ratio_float32': 4 * cells / (frame_bytes + auxiliary), 'macro': macro, 'groups': groups,
        'checkpoint_sha256': digest(args.checkpoint), 'source_sha256': digest(Path(__file__)),
        'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'no_test_inference': True, 'parameter_updates': 0, 'shared_waveform_weights_excluded': True,
        'coding': 'same frozen marginal/fixed/Huffman as E1; not claimed to be the historical best stream'})
    (output / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
