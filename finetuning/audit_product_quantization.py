#!/usr/bin/env python3
"""Read-only frame, subspace replay, train-only lineage and decision audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from compression.product_quantization import replay_product, validate_codebooks
from finetuning.representation_study import digest, read_manifest, write_json
from finetuning.run_physical_guard_search import physical_admit
from finetuning.run_spatial_output_study import macro_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    assert (root / 'COMPLETE').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    result = json.loads((root / 'results.json').read_text())
    assert result['neural_parameter_updates'] == 0 and result['validation_signal_windows_read'] == 0
    source_hash = digest(args.source)
    assert source_hash == protocol['source_checkpoint_sha256']
    hashes = 0
    for f, expected in protocol['code_sha256'].items():
        assert digest(root / 'code_snapshot' / f) == expected
        hashes += 1
    for f, expected in protocol['input_sha256'].items():
        assert digest(ROOT / f) == expected
        hashes += 1
    assert digest(args.samples_dir / 'samples_manifest.csv') == protocol['manifest_sha256']
    rows = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    fit, checks = protocol['fit_train'], protocol['check_train']
    assert all(f in rows for f in [*fit, *checks])
    assert not {rows[f]['drive_id'] for f in fit} & {rows[f]['drive_id'] for f in checks}
    assert set(protocol['input_sha256']) == set(fit) | set(checks)
    source_folder = root / 'source'
    stream = AlphabetStream(np.load(source_folder / 'counts.npy', allow_pickle=False), source_hash)
    source_frames = []
    for i, _ in enumerate(checks):
        frame = (source_folder / f'{i:05d}.fak').read_bytes()
        source_frames.append(frame)
        stream.decode(frame)
        assert (source_folder / f'{i:05d}.route').read_bytes() == b'\x00' + frame
    verified = len(source_frames)
    summary = {}
    for arm, expected_shape in protocol['arms'].items():
        folder = root / arm
        descriptor = json.loads((folder / 'descriptor.json').read_text())
        books_path, counts_path = folder / 'codebooks.npy', folder / 'counts.npy'
        book_hash = digest(books_path)
        assert book_hash == descriptor['codebooks_sha256']
        assert digest(counts_path) == descriptor['counts_sha256']
        basis = center = None
        fingerprint = bytes.fromhex(source_hash) + bytes.fromhex(book_hash) + descriptor['representation'].encode()
        if descriptor['representation'] == 'pca-product-vq-2-v1':
            assert protocol['rotation']['rotation_probe_allowed']
            assert digest(folder / 'basis.npy') == descriptor['basis_sha256'] == protocol['rotation']['basis_sha256']
            assert digest(folder / 'center.npy') == descriptor['center_sha256'] == protocol['rotation']['center_sha256']
            basis = torch.from_numpy(np.load(folder / 'basis.npy', allow_pickle=False))
            center = torch.from_numpy(np.load(folder / 'center.npy', allow_pickle=False))
            fingerprint += bytes.fromhex(descriptor['basis_sha256']) + bytes.fromhex(descriptor['center_sha256'])
        else:
            assert descriptor['representation'] == 'product-vq-2-v1'
        binding = hashlib.sha256(fingerprint).hexdigest()
        assert binding == descriptor['binding_sha256']
        codec = AlphabetStream(np.load(counts_path, allow_pickle=False), binding)
        assert codec.identifier.hex() == descriptor['identifier']
        books = torch.from_numpy(np.load(books_path, allow_pickle=False))
        assert list(books.shape) == expected_shape
        groups, _, _ = validate_codebooks(books)
        records = json.loads((folder / 'records.json').read_text())
        assert [r['sample_path'] for r in records] == checks
        accepted = frame_bytes = guarded_bytes = 0
        losses, chosen_losses = [], []
        for i, record in enumerate(records):
            frame = (folder / f'{i:05d}.candidate.fak').read_bytes()
            symbols, shape, symbol_grid, side = codec.decode(frame)
            original, old_shape, old_grid, old_side = stream.decode(source_frames[i])
            assert shape == old_shape and side == old_side
            assert symbol_grid == (old_grid[0], old_grid[1] * groups)
            assert len(symbols) == record['symbols'] == len(original) * groups
            assert record['nominal_symbol_bits'] == len(symbols) * codec.bits
            q = replay_product(torch.from_numpy(symbols.astype(np.int64)).view(-1, groups), tuple(old_grid), books, basis, center)
            assert torch.isfinite(q).all()
            assert record['source_frame_bytes'] == len(source_frames[i])
            assert record['candidate_frame_bytes'] == len(frame)
            allowed = (len(frame) <= len(source_frames[i]) and physical_admit(record['source'], record['candidate'])
                       and record['candidate_normalized_distortion'] < record['source_normalized_distortion'])
            assert record['admitted'] == allowed
            route = bytes([int(allowed)]) + (frame if allowed else source_frames[i])
            assert (folder / f'{i:05d}.guarded.route').read_bytes() == route
            loss = record['candidate_normalized_distortion'] if allowed else record['source_normalized_distortion']
            assert record['guarded_normalized_distortion'] == loss
            row = rows[record['sample_path']]
            losses.append((row['family'], row['configuration'], record['candidate_normalized_distortion']))
            chosen_losses.append((row['family'], row['configuration'], loss))
            accepted += allowed
            frame_bytes += len(frame)
            guarded_bytes += len(route)
            verified += 1
        arm_result = result['arms'][arm]
        assert arm_result['macro_normalized_distortion'] == macro_loss(losses)
        assert arm_result['guarded_macro_normalized_distortion'] == macro_loss(chosen_losses)
        assert arm_result['candidate_frame_bytes'] == frame_bytes
        assert arm_result['guarded_routed_frame_bytes'] == guarded_bytes
        assert arm_result['admitted_windows'] == accepted
        aux = sum((folder / f).stat().st_size for f in ('counts.npy', 'descriptor.json'))
        source_aux = sum((source_folder / f).stat().st_size for f in ('counts.npy', 'descriptor.json'))
        transport_bytes = (root / 'transport.json').stat().st_size
        assert arm_result['guarded_stream_bytes'] == guarded_bytes + aux + source_aux + transport_bytes
        assert arm_result['shared_codebook_tensor_bytes'] == books.numel() * books.element_size()
        if basis is not None:
            assert arm_result['shared_transform_tensor_bytes'] == basis.numel() * basis.element_size() + center.numel() * center.element_size()
        gate = arm_result['guarded_distortion_gain'] >= .01 and arm_result['guarded_stream_bytes'] <= result['source']['stream_bytes']
        assert gate == arm_result['advance_gate_met']
        summary[arm] = {'accepted': accepted, 'frames': len(records), 'shared_codebook_bytes': arm_result['shared_codebook_tensor_bytes'],
                        'advance_gate_met': gate}
    output = {'neural_updates': 0, 'validation_windows_accessed': 0, 'frames_verified': verified,
              'external_centroid_parameter_updates': result.get('external_centroid_parameter_updates', 0),
              'snapshot_and_input_hashes_verified': hashes, 'check_windows': len(checks), 'fit_windows': len(fit),
              'source_weights_unchanged': True, 'arms': summary, 'auditor_sha256': digest(Path(__file__))}
    write_json(root / 'audit.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
