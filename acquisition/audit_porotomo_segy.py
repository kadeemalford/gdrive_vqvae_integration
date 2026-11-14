#!/usr/bin/env python3
"""Check the converted PoroTomo HDF5 against its published native SEG-Y source."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import h5py
import numpy as np
import requests

URL = 'https://nrel-pds-porotomo.s3.amazonaws.com/DAS/SEG-Y/DASV/20160317/PoroTomo_iDAS025_160318160317.sgy'
SIZE = 46175760


def native_traces(payload):
    if len(payload) < 3600:
        raise ValueError('Truncated SEG-Y header')
    interval_us, samples, format_code = (struct.unpack_from('>H', payload, offset)[0]
                                         for offset in (3216, 3220, 3224))
    if format_code != 5 or not samples or not interval_us:
        raise ValueError('Expected unambiguous IEEE float32 SEG-Y')
    stride = 240 + 4 * samples
    count, remainder = divmod(len(payload) - 3600, stride)
    if not count or remainder:
        raise ValueError('Unexpected SEG-Y trace layout')
    return np.ndarray((count, samples), dtype='>f4', buffer=payload, offset=3840,
                      strides=(stride, 4)).T, interval_us


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hdf5', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    path = args.raw_dir / URL.rsplit('/', 1)[1]
    received = 0
    with requests.get(URL, stream=True, timeout=90) as response, path.open('xb') as stream:
        response.raise_for_status()
        for block in response.iter_content(1024 * 1024):
            received += len(block)
            if received > SIZE:
                raise ValueError('Native source exceeds bounded download')
            stream.write(block)
    if received != SIZE:
        raise ValueError('Native source size mismatch')
    payload = path.read_bytes()
    values, interval = native_traces(payload)
    native_hash = hashlib.sha3_256(payload).hexdigest()
    with h5py.File(args.hdf5, 'r') as handle:
        expected = handle.attrs['source_checksum'][0]
        expected = expected.decode() if isinstance(expected, bytes) else str(expected)
        if expected != 'SHA3-256:' + native_hash:
            raise ValueError('Native source checksum differs from HDF5 provenance')
        converted = handle['das'][...]
        raw = values.astype('<f4')
        identical = (converted.dtype == np.dtype('float32') and converted.shape == raw.shape
                     and np.array_equal(converted.view('<u4'), raw.view('<u4')))
    result = {'native_url': URL, 'native_bytes': received, 'native_sha3_256': native_hash,
              'native_source_checksum_matches_hdf5_provenance': True,
              'native_sha256': hashlib.sha256(payload).hexdigest(),
              'hdf5_sha256': hashlib.sha256(args.hdf5.read_bytes()).hexdigest(),
              'shape_time_channels': list(values.shape), 'native_sampling_interval_us': interval,
              'all_waveform_values_bit_exact_after_endian_conversion': identical,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_evaluations': 0,
              'limitation': 'Identity excludes waveform filtering during this conversion; it does not independently calibrate differential phase to strain-rate'}
    (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for encoding in ('ascii', 'cp500'):
        (args.output_dir / f'native_text_header_{encoding}.txt').write_text(payload[:3200].decode(encoding, errors='replace'))
    if not identical:
        raise ValueError('Converted waveforms are not bit-exact native traces')
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
