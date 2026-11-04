"""Shared split-safe I/O and physical scoring for representation pilots."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from finetuning.evaluate_checkpoints import metrics
from finetuning.diagnose_reconstruction import spectral_metrics
from finetuning.physical_validation import ordinary_fidelity
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_manifest(samples):
    with (samples / 'samples_manifest.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    selected = {split: [r for r in rows if r['split'] == split] for split in ('train', 'val')}
    assert all(selected.values())
    assert not ({r['group_id'] for r in selected['train']} & {r['group_id'] for r in selected['val']})
    assert not ({r['drive_id'] for r in selected['train']} & {r['drive_id'] for r in selected['val']})
    for split, values in selected.items():
        assert {r['family'] for r in values} == {'doped', 'multicore', 'multimode', 'single_mode'}
        assert all((ROOT / r['sample_path']).is_file() for r in values)
        assert all(r['spatial_band_policy'] == 'persistent-amplitude-v1' for r in values)
    return selected


def pilot_rows(rows, per_region=1, seed=2078):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['drive_id'], row['spatial_region_index']].append(row)
    rng = random.Random(seed)
    return [r for key in sorted(grouped) for r in rng.sample(
        sorted(grouped[key], key=lambda r: r['sample_path']), min(per_region, len(grouped[key])))]


def load_packet(row):
    assert row['split'] in ('train', 'val'), 'Test waveform access is forbidden'
    with np.load(ROOT / row['sample_path'], allow_pickle=False) as payload:
        raw = np.asarray(payload['data'], dtype=np.float32)
    return raw, encode_robust(raw)


def distortion(prediction, target, valid):
    return .5 * F.mse_loss(prediction[valid], target[valid]) + .5 * F.l1_loss(prediction[valid], target[valid])


def score(row, raw, packet, prediction):
    restored = decode(prediction, packet.side_stream)
    assert raw[~packet.valid].tobytes() == restored[~packet.valid].tobytes()
    physical = ordinary_fidelity(raw, restored, packet.valid)
    # Score on the same ordinary normalized reference as the VQ studies.
    from finetuning.reversible_normalization import parse_side_stream
    _, mean, scale, _, _ = parse_side_stream(packet.side_stream)
    normalized = ((restored.astype(np.float64) - mean) / scale).astype(np.float32)
    normalized[~packet.valid] = 0
    return {'family': row['family'], 'configuration': row['configuration'],
            '_aggregation_weight': float(raw.size),
            **metrics(packet.values[packet.valid], normalized[packet.valid]),
            **spectral_metrics(packet.values, normalized), **physical}


def tensor_packet(packet, device='cuda'):
    return (torch.from_numpy(packet.values)[None, None].to(device),
            torch.from_numpy(packet.valid)[None, None].to(device))
