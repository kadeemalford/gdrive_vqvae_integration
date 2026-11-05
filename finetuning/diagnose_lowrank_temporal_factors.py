#!/usr/bin/env python3
"""Train-only opportunity gate for a fixed sparse spectral temporal-factor control."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.fft import dct

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.lowrank_stream import factorize
from finetuning.representation_study import digest, read_manifest, load_packet, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.probe
    assert (root / 'COMPLETE').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    assert digest(args.samples_dir / 'samples_manifest.csv') == protocol['manifest_sha256']
    target = root / 'temporal_factor_diagnostic.json'
    if target.exists():
        raise FileExistsError('Preserve previous factor diagnosis')
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    rows, grouped = [], defaultdict(list)
    for path in protocol['selection']:
        assert digest(ROOT / path) == protocol['input_sha256'][path]
        row = train[path]
        _, packet = load_packet(row)
        _, singular, vt = factorize(packet)
        rank = min(8, len(singular))
        # A geometry/energy diagnostic, not the rank selected by the actual byte controller.
        factors = singular[:rank, None] * vt[:rank]
        coefficients = dct(factors, type=2, norm='ortho', axis=1, workers=1)
        keep = min(16, coefficients.shape[1])
        power = np.square(coefficients)
        full = float(power.sum())
        retained = float(np.sort(power, axis=1)[:, -keep:].sum())
        variable = float(np.std(packet.values[packet.valid].astype(np.float64))) > 1e-6
        fraction = retained / full if full > 1e-20 else None
        rows.append({'sample_path': path, 'family': row['family'], 'configuration': row['configuration'],
            'rank_diagnostic': rank, 'dct_coefficients_per_factor': keep,
            'retained_temporal_factor_energy_fraction': fraction, 'variable_target': variable})
        if variable and fraction is not None:
            grouped[row['family'], row['configuration']].append(fraction)
    groups = [{'family': f, 'configuration': c, 'mean_retained_energy': float(np.mean(values)),
               'windows': len(values)} for (f, c), values in sorted(grouped.items())]
    qualifying = [g for g in groups if g['mean_retained_energy'] >= .8]
    result = {'training_updates': 0, 'validation_signal_windows_read': 0, 'rows': rows, 'groups': groups,
        'code_sha256': digest(Path(__file__)), 'reference_results_sha256': digest(root / 'results.json'),
        'rule': 'one fixed top16-DCT-per-temporal-factor coded probe only if >=80% temporal energy in >=2 distinct train configurations; no coefficient-count sweep',
        'sparse_factor_probe_allowed': len(qualifying) >= 2, 'qualifying_configurations': len(qualifying),
        'limitations': ['continuous energy opportunity only, not an encoded waveform result',
            'diagnostic rank<=8 is not a fixed-rate comparison; actual indices/quantizers/scales/rank must be charged',
            'no proof of noise independence or information-theoretic rank limit']}
    write_json(target, result)
    print(json.dumps({'groups': groups, 'sparse_factor_probe_allowed': result['sparse_factor_probe_allowed']}))


if __name__ == '__main__':
    main()
