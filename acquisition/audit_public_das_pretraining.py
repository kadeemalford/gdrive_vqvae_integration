#!/usr/bin/env python3
"""Extract one predefined public DAS HDF5 by bounded ZIP ranges, then audit on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile
import zlib

import h5py
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acquisition.audit_external_hdf5_headers import BoundedRangeReader, describe_metadata
from acquisition.analyze_quantx_pretraining import diagnostics

URL = 'https://ndownloader.figshare.com/files/52736258'
ZIP_SIZE = 47120418495
MEMBER = 'DAS-dataset/data/fence/fence_2023-04-17T123117+0100.h5'
NOVEMBER_MEMBER = 'DAS-dataset/data/fence/kopani_plot_2023-11-07T150207Z.h5'


class ZipRangeReader(BoundedRangeReader):
    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
            if size > self.byte_budget:
                raise ValueError('ZIP request exceeds bounded metadata budget')
        return super().read(size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--member', choices=(MEMBER, NOVEMBER_MEMBER), default=MEMBER)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    reader = ZipRangeReader(session, URL, ZIP_SIZE, byte_budget=2 * 1024**2)
    with zipfile.ZipFile(reader) as archive:
        info = archive.getinfo(args.member)
    if info.compress_type != zipfile.ZIP_DEFLATED or max(info.file_size, info.compress_size) > 1200 * 1024**2:
        raise ValueError('Unexpected ZIP member type/size')
    reader.seek(info.header_offset)
    header = reader.read(30)
    values = struct.unpack('<4s5H3I2H', header)
    if values[0] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP member header')
    offset = info.header_offset + 30 + values[-2] + values[-1]
    end = offset + info.compress_size - 1
    protocol = {'kind': 'public_das_waveform_preflight_v1', 'url': URL, 'zip_size': ZIP_SIZE,
                'member': args.member, 'selection': 'Smallest April HDF5 or smallest November HDF5; product checked after selection',
                'compressed_bytes': info.compress_size, 'uncompressed_bytes': info.file_size,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'license': 'CC-BY-4.0 dataset; no third-party loader copied',
                'training_updates': 0, 'model_evaluations': 0, 'representative_sample': False}
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    output = args.raw_dir / Path(args.member).name
    decompressor, crc, sha, written = zlib.decompressobj(-15), 0, hashlib.sha256(), 0
    with session.get(URL, headers={'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity'},
                     stream=True, timeout=120) as response:
        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{ZIP_SIZE}':
            raise ValueError('Server did not honor member range')
        compressed_read = 0
        with output.open('xb') as stream:
            for block in response.iter_content(1024 * 1024):
                compressed_read += len(block)
                if compressed_read > info.compress_size:
                    raise ValueError('Response exceeds requested member')
                pending = block
                while pending:
                    data = decompressor.decompress(pending, 8 * 1024**2)
                    pending = decompressor.unconsumed_tail
                    written += len(data)
                    if written > info.file_size:
                        raise ValueError('Decompressed member exceeds declared size')
                    stream.write(data); crc = zlib.crc32(data, crc); sha.update(data)
            if compressed_read != info.compress_size or written != info.file_size or crc != info.CRC or not decompressor.eof:
                raise ValueError('ZIP member integrity mismatch')
    with h5py.File(output, 'r') as handle:
        metadata = describe_metadata(handle)
        raw = handle['Acquisition/Raw[0]']
        fs = float(raw.attrs['OutputDataRate'])
        ds = raw['RawData']
        shape = list(ds.shape)
        windows = []
        # Bound CPU memory and HDF5 reads; contiguous block includes all recorded loci.
        for start in np.linspace(0, ds.shape[0] - 2048, 5, dtype=int):
            values = ds[int(start):int(start)+2048, :].astype(np.float64)
            windows.append({'start_sample': int(start), 'nonfinite_cells': int((~np.isfinite(values)).sum()),
                            'raw_product': diagnostics(values, fs)})
    result = {'sha256': sha.hexdigest(), 'zip_crc_verified': True, 'downloaded_compressed_bytes': compressed_read,
              'uncompressed_bytes': written, 'metadata_range_bytes': reader.fetched_bytes,
              'output_sample_rate_hz': fs, 'signal_shape': shape, 'windows': windows, 'metadata': metadata,
              'limitations': ['One fence-event acquisition, not generalization or full-corpus characterization',
                              'No transfer benefit established without controlled FEBUS fine-tuning']}
    (args.output_dir / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps({k:v for k,v in result.items() if k not in ('windows','metadata')}, indent=2))


if __name__ == '__main__':
    main()
