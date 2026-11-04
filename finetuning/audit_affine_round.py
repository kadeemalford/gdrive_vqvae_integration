#!/usr/bin/env python3
"""Read-only frame/guard/hash/coverage audit of the adaptive-restoration round."""
import argparse
import json
from pathlib import Path
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.affine_restoration import AFFINE_MAGIC
from compression.numeric_restoration import NUMERIC_MAGICS, parse_numeric_side
from compression.partition_bundle import unpack_bundle
from finetuning.representation_study import digest, write_json
from finetuning.reversible_normalization import MAGIC, parse_side_stream
from finetuning.run_index_search import FrozenIndexStream
from finetuning.run_physical_guard_search import physical_admit


def validate_frame(frame, stream, expected_shape=None):
    indices, shape, latent, side = stream.decode(frame)
    if side[:4] in NUMERIC_MAGICS:
        side_shape, _, _, _, _ = parse_numeric_side(side)
    else:
        side_shape, _, _, _, _ = parse_side_stream(MAGIC + side[4:] if side[:4] == AFFINE_MAGIC else side)
    assert side_shape == shape and len(indices) == int(np.prod(latent))
    if expected_shape is not None:
        assert shape == expected_shape
    return side[:4].decode('ascii')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT / 'results/experiments')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--calibration-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, source_hash = args.root, digest(args.source)
    names = ['affine_side_train_probe_20261102', 'bounded_affine_train_probe_20261102',
             'affine_e1_fresh_local_20261102', 'bounded_affine_e1_fresh_local_20261103',
             'external_smf_frozen_stress_20261102', 'external_smf_bounded_transfer_20261103']
    hashes, frames, guard_checks = 0, 0, 0
    kinds = {}
    for name in names:
        directory = root / name
        assert (directory / 'COMPLETE').exists(), name
        protocol = json.loads((directory / 'protocol.json').read_text())
        for path, expected in protocol['code_sha256'].items():
            assert digest(directory / 'code_snapshot' / path) == expected
            hashes += 1
        result = json.loads((directory / 'results.json').read_text())
        assert result['parameter_updates'] == 0
        if name == 'external_smf_frozen_stress_20261102':
            assert protocol['source_checkpoint_sha256'] == {'compact': source_hash, 'dense': digest(args.dense)}
            for arm, model_hash in protocol['source_checkpoint_sha256'].items():
                stream = FrozenIndexStream(np.load(directory / arm / 'counts.npy', allow_pickle=False), model_hash)
                for frame in (directory / arm).glob('*.frame'):
                    kind = validate_frame(frame.read_bytes(), stream)
                    kinds[kind] = kinds.get(kind, 0) + 1
                    frames += 1
                for row in json.loads((directory / arm / 'records.json').read_text()):
                    assert row['affine_frame_bytes'] <= row['direct_frame_bytes']
                    assert row['restoration'] == 'original' or physical_admit(row['direct'], row['affine'])
                    guard_checks += 1
                for sample_index in range(6):
                    for policy in ('direct', 'affine'):
                        with np.load(directory / arm / f'f{sample_index:02d}.{policy}.reconstruction.npz', allow_pickle=False) as data:
                            decoded = data['data']
                        sample = json.loads((root / 'external_smf_stress_samples_v2_20261102/samples.json').read_text())[sample_index]
                        with np.load(root / 'external_smf_stress_samples_v2_20261102' / sample['path'], allow_pickle=False) as data:
                            raw = data['data']
                        invalid = ~np.isfinite(raw)
                        assert invalid.sum() == 120000
                        np.testing.assert_array_equal(raw.view(np.uint32)[invalid], decoded.view(np.uint32)[invalid])
            continue
        assert protocol['source_sha256'] == source_hash
        counts = directory / 'counts.npy'
        counts = counts if counts.exists() else args.calibration_dir / 'counts.npy'
        stream = FrozenIndexStream(np.load(counts, allow_pickle=False), source_hash)
        for frame in directory.glob('*.frame'):
            kind = validate_frame(frame.read_bytes(), stream)
            kinds[kind] = kinds.get(kind, 0) + 1
            frames += 1
        rows = json.loads((directory / 'records.json').read_text())
        if 'train_probe' in name:
            for row in rows:
                assert row['frame_bytes'] <= row['source_frame_bytes']
                assert row['chosen'] == 'source' or physical_admit(row['source'], row['strict_affine'])
                guard_checks += 1
        else:
            if 'excluded_validation_windows' in protocol:
                assert not set(protocol['validation']) & set(protocol['excluded_validation_windows'])
            for row in rows:
                for arm in ('strict', 'affine', 'bounded'):
                    if arm not in row:
                        continue
                    candidate = row[arm]
                    assert candidate['frame_bytes'] <= row['source_frame_bytes']
                    assert (not candidate['changed_indices'] and candidate['restoration'] == 'original') or physical_admit(row['source'], candidate['metrics'])
                    guard_checks += 1
    source_samples = json.loads((root / 'external_smf_stress_samples_v2_20261102/samples.json').read_text())
    preparation = json.loads((root / 'external_smf_stress_samples_v2_20261102/protocol.json').read_text())
    assert all(s['orientation'] in ('Horizontal', 'Vertical') for s in source_samples)
    assert {r['id'] for r in preparation['selected_files']} <= set(preparation['external_development_group_ids'])
    assert all(r['candidate_folder']['fiber'] == 'SMF' for r in preparation['selected_files'])
    assert len(preparation['signal_unseen_reserved_groups']) == 2
    bundles = embedded = exact_fragments = 0
    coverage = root / 'external_smf_fixed_coverage_20261103'
    assert (coverage / 'COMPLETE').exists()
    protocol = json.loads((coverage / 'protocol.json').read_text())
    for path, expected in protocol['code_sha256'].items():
        assert digest(coverage / 'code_snapshot' / path) == expected
        hashes += 1
    for arm, checkpoint_hash in protocol['source_checkpoint_sha256'].items():
        stream = FrozenIndexStream(np.load(root / 'external_smf_frozen_stress_20261102' / arm / 'counts.npy', allow_pickle=False), checkpoint_hash)
        for frame in (coverage / arm).glob('*.frame'):
            kind_name = validate_frame(frame.read_bytes(), stream)
            kinds[kind_name] = kinds.get(kind_name, 0) + 1
            frames += 1
    for name in ('full_coverage_partition_smf_20261103', 'dense_e1_bundle_control_smf_20261103'):
        directory = root / name
        assert (directory / 'COMPLETE').exists(), name
        protocol = json.loads((directory / 'protocol.json').read_text())
        assert json.loads((directory / 'results.json').read_text())['parameter_updates'] == 0
        for path, expected in protocol['code_sha256'].items():
            assert digest(directory / 'code_snapshot' / path) == expected
            hashes += 1
        arm_paths = [(directory / arm, checkpoint_hash) for arm, checkpoint_hash in protocol['source_checkpoint_sha256'].items()] if 'source_checkpoint_sha256' in protocol else [(directory, protocol['source_sha256'])]
        for folder, checkpoint_hash in arm_paths:
            stream = FrozenIndexStream(np.load(folder / 'counts.npy', allow_pickle=False), checkpoint_hash)
            for path in folder.glob('*.fpb'):
                shape, fragments = unpack_bundle(path.read_bytes())
                assert shape == (551, 1000)
                bundles += 1
                for kind, start, height, payload in fragments:
                    if kind == 0:
                        kind_name = validate_frame(payload, stream, (height, shape[1]))
                        kinds[kind_name] = kinds.get(kind_name, 0) + 1
                        embedded += 1
                    else:
                        codec = zlib.decompressobj()
                        raw = codec.decompress(payload, height * shape[1] * 4 + 1)
                        assert len(raw) == height * shape[1] * 4 and codec.eof
                        assert not codec.unused_data and not codec.unconsumed_tail
                        assert not np.isfinite(np.frombuffer(raw, dtype='<f4')).any()
                        exact_fragments += 1
            rows = json.loads((folder / 'records.json').read_text())
            if name.startswith('full_coverage'):
                for row in rows:
                    assert row['neural_indices_source'] == row['neural_indices_candidate']
                    assert row['quality_eligible'] == physical_admit(row['source'], row['candidate'])
                    assert row['budget_eligible'] == (row['candidate_bytes'] <= row['source_bytes'])
                    assert row['admitted'] == (row['quality_eligible'] and row['budget_eligible'])
                    key = f'f{row["file_index"]:02d}_w{row["window"]:02d}'
                    choice = 'candidate' if row['admitted'] else 'source'
                    assert (folder / f'{key}.guarded.fpb').read_bytes() == (folder / f'{key}.{choice}.fpb').read_bytes()
                    guard_checks += 1
            else:
                for row in rows:
                    for arm in ('strict', 'affine', 'bounded'):
                        item = row[arm]
                        assert item['frame_bytes'] <= row['source_bytes']
                        assert item['metrics'] == row['source'] or physical_admit(row['source'], item['metrics'])
                        guard_checks += 1
    result = {'neural_parameter_updates_this_round': 0, 'frame_streams_verified': frames,
        'embedded_neural_frames_verified': embedded, 'full_bundle_streams_verified': bundles,
        'exact_only_nonfinite_fragments_verified': exact_fragments,
        'snapshot_hashes_verified': hashes, 'guard_decisions_verified': guard_checks, 'side_formats_observed': kinds,
        'nonfinite_values_per_external_file_preserved': 120000, 'external_files': 6,
        'EDAS_groups_without_signal_evaluation': 2, 'source_sha256': source_hash, 'dense_sha256': digest(args.dense),
        'auditor_sha256': digest(Path(__file__))}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
