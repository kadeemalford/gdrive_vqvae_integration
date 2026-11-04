#!/usr/bin/env python3
"""Post-hoc coverage/energy/seam diagnosis; never change codec targets or frames."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.numeric_restoration import parse_input_side
from finetuning.paired_search_diagnostics import finite_json
from finetuning.physical_validation import ordinary_fidelity
from finetuning.representation_study import digest, write_json
from finetuning.run_index_search import FrozenIndexStream
from finetuning.spatial_band_filter import detect_spatial_bands


def seam_control(raw, decoded, valid):
    mask = valid[:, 1:] & valid[:, :-1]
    seam = (np.arange(1, raw.shape[1]) % 1000 == 0)[None, :]
    values = {}
    for label, include in [('seam', mask & seam), ('interior', mask & ~seam)]:
        target = raw[:, 1:][include].astype(np.float64) - raw[:, :-1][include].astype(np.float64)
        predicted = decoded[:, 1:][include].astype(np.float64) - decoded[:, :-1][include].astype(np.float64)
        error = float(np.sum((predicted - target) ** 2))
        power = float(np.sum(target ** 2))
        values[label] = {'pairs': int(include.sum()), 'error_power': error, 'target_power': power,
                         'relative_difference_rmse': float(np.sqrt(error / power)) if power > 1e-20 else None}
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir
    assert (root / 'COMPLETE').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    samples = json.loads((args.samples_dir / 'samples.json').read_text())
    records = []
    for index, sample in enumerate(samples):
        with np.load(args.samples_dir / sample['path'], allow_pickle=False) as p:
            raw = p['data']
        decision = detect_spatial_bands(raw[None, ...])
        keep = np.isfinite(raw).all(axis=1)
        keep[decision['removed_source_channel_indices']] = False
        item = {'drive_id': sample['drive_id'], 'orientation': sample['orientation'],
                'stimulus_label_hz': sample['stimulus_label_hz'], 'fixed_policy_decision': decision,
                'diagnostic_retained_rows': int(keep.sum()), 'source_rows': raw.shape[0], 'arms': {}}
        for arm in ('compact', 'dense'):
            stream = FrozenIndexStream(np.load(root / arm / 'counts.npy', allow_pickle=False), protocol['source_checkpoint_sha256'][arm])
            valid = np.zeros(raw.shape, dtype=bool)
            for window in range(20):
                for region, start in enumerate((0, 512)):
                    frame = (root / arm / f'f{index:02d}_w{window:02d}_r{region}.direct.frame').read_bytes()
                    _, shape, _, side = stream.decode(frame)
                    side_shape, _, _, escaped, _ = parse_input_side(side)
                    assert side_shape == shape
                    local = np.ones(shape, dtype=bool)
                    local.reshape(-1)[escaped] = False
                    valid[start:start + shape[0], window * 1000:(window + 1) * 1000] = local
            with np.load(root / arm / f'f{index:02d}.direct.reconstruction.npz', allow_pickle=False) as p:
                decoded = p['data']
            assert raw[~valid].tobytes() == decoded[~valid].tobytes()
            ordinary = raw[valid].astype(np.float64)
            center = float(ordinary.mean())
            total_power = float(np.sum((ordinary - center) ** 2))
            tile_stats = []
            for begin in (0, 512):
                end = min(begin + 512, raw.shape[0])
                local = valid[begin:end]
                x, y = raw[begin:end][local].astype(np.float64), decoded[begin:end][local].astype(np.float64)
                tile_stats.append({'start': begin, 'stop': end, 'ordinary_cells': int(local.sum()),
                    'target_global_centered_power_fraction': float(np.sum((x - center) ** 2) / total_power),
                    'ordinary_raw_rmse': float(np.sqrt(np.mean((y - x) ** 2)))})
            restricted = valid & keep[:, None]
            item['arms'][arm] = {'tile_energy': tile_stats, 'seam_control': seam_control(raw, decoded, valid),
                'fixed_coverage_stratification': ordinary_fidelity(raw, decoded, restricted) if restricted.any() else None}
        records.append(item)
    result = {'source_results_sha256': digest(root / 'results.json'), 'code_sha256': digest(Path(__file__)),
        'spatial_policy_sha256': digest(ROOT / 'finetuning/spatial_band_filter.py'), 'files': records,
        'limitations': ['post-hoc diagnostic only, no new model or frame selection',
            'fixed historical spatial heuristic applied only for stratification; input was NOT re-encoded after filtering',
            'no change to coverage, bytes or primary full-zone results',
            'seam first-difference errors controlled for target difference energy; not a causal padding test']}
    write_json(root / 'coverage_seam_diagnostic.json', finite_json(result))
    print(json.dumps([{'orientation': r['orientation'], 'hz': r['stimulus_label_hz'],
        'retained_rows': r['diagnostic_retained_rows'], 'amplitude_removed_rows': len(r['fixed_policy_decision']['removed_source_channel_indices']),
        'tile_target_power_fractions': [t['target_global_centered_power_fraction'] for t in r['arms']['compact']['tile_energy']],
        'seam_relative': {a: {k: v['relative_difference_rmse'] for k, v in r['arms'][a]['seam_control'].items()} for a in r['arms']}}
        for r in records], indent=2))


if __name__ == '__main__':
    main()
