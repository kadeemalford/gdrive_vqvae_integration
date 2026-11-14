#!/usr/bin/env python3
"""Distinguish phase drift from narrowband content; analysis only, no target preprocessing."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def trend_probe(values, sample_rate, excitation):
    x = np.asarray(values, dtype=np.float64)
    x = x - x.mean(axis=0)
    t = np.arange(len(x), dtype=np.float64)
    t -= t.mean()
    trend = t[:, None] * ((t @ x) / (t @ t))[None, :]
    residual = x - trend

    def excitation_fraction(signal):
        p = np.abs(np.fft.rfft(signal * np.hanning(len(signal))[:, None], axis=0)) ** 2
        p[1:-1 if len(signal) % 2 == 0 else None] *= 2
        p[0] = 0
        mask = np.abs(np.fft.rfftfreq(len(signal), 1 / sample_rate) - excitation) <= 20
        energy = p.sum(axis=0)
        fractions = np.divide(p[mask].sum(axis=0), energy, out=np.zeros_like(energy), where=energy > 0)
        return {'pooled': float(p[mask].sum() / max(energy.sum(), 1e-30)),
                'max_channel': float(fractions.max()), 'median_channel': float(np.median(fractions))}

    return {'linear_trend_energy_fraction': float(np.square(trend).sum() / max(np.square(x).sum(), 1e-30)),
            'raw_excitation_band': excitation_fraction(x),
            'detrended_excitation_band': excitation_fraction(residual),
            'difference_excitation_band': excitation_fraction(np.diff(values, axis=0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    args = parser.parse_args()
    downloads = json.loads((args.audit_dir / 'downloads.json').read_text())
    rows = []
    for entry in downloads:
        if not entry['name'].endswith('.h5'):
            continue
        excitation = float(entry['scenario'].split('_')[-1].removesuffix('hz'))
        with h5py.File(entry['local_path'], 'r') as h:
            raw = h['Acquisition/Raw[0]']
            ds = raw['RawData']
            fs = float(raw.attrs['OutputDataRate'])
            flags = raw['Custom/GpBits'][...]
            probes = [trend_probe(ds[int(start):int(start) + 2048, :].astype(np.float64), fs, excitation)
                      for start in np.linspace(0, ds.shape[0] - 2048, 5, dtype=int)]
            rows.append({'id': entry['id'], 'family': entry['family'], 'scenario': entry['scenario'],
                         'modified_flag_samples': int(np.count_nonzero(flags & 1)),
                         'compromised_flag_samples': int(np.count_nonzero(flags & 2)),
                         'median_linear_trend_energy_fraction': float(np.median([r['linear_trend_energy_fraction'] for r in probes])),
                         'median_raw_excitation_band_pooled': float(np.median([r['raw_excitation_band']['pooled'] for r in probes])),
                         'median_detrended_excitation_band_pooled': float(np.median([r['detrended_excitation_band']['pooled'] for r in probes])),
                         'median_detrended_excitation_band_max_channel': float(np.median([r['detrended_excitation_band']['max_channel'] for r in probes])),
                         'median_difference_excitation_band_pooled': float(np.median([r['difference_excitation_band']['pooled'] for r in probes])),
                         'probes': probes})
    result = {'source': 'downloaded sample selected before signal inspection',
              'source_window_samples': 2048, 'excitation_band_half_width_hz': 20,
              'preprocessing_for_training': False, 'target_modified': False,
              'limitation': 'Linear trend subtraction and differencing are diagnostic contrasts only; no calibration or channel validity inferred',
              'rows': rows}
    with (args.audit_dir / 'drift_diagnostics.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in rows:
        print({k: v for k, v in row.items() if k != 'probes'})


if __name__ == '__main__':
    main()
