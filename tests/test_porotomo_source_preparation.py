from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from acquisition import prepare_porotomo_source as preparation


def hdf_bytes(identifier=None):
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as h:
        if identifier is not None:
            data = h.create_dataset('das', data=np.arange(2000, dtype=np.float32).reshape(1000, 2))
            data.attrs['sampling_interval_seconds'] = .001
            times = h.create_dataset('t', data=np.arange(1000) * .001)
            times.attrs['units'] = 'seconds since 1970-01-01'
            h.create_dataset('channel', data=[1, 2])
            h.attrs['source_checksum'] = ['SHA3-256:' + hashlib.sha3_256(identifier.encode()).hexdigest()]
    return buffer.getvalue()


class Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield self.content


class SourcePreparationTests(unittest.TestCase):
    def test_metadata_only_objects_are_excluded_but_small_waveforms_are_kept(self):
        files, listings = {}, {}
        for day in preparation.DAYS:
            nodes = ET.Element('ListBucketResult', xmlns=preparation.NS['s'])
            for index in range(16):
                key = f'DAS/H5/DASV/{day}/{day}_{index:02d}.h5'
                payload = hdf_bytes(key if index < 15 else None)
                files[key] = payload
                item = ET.SubElement(nodes, 'Contents')
                for name, text in [('Key', key), ('Size', str(len(payload))), ('ETag', 'fixture')]:
                    ET.SubElement(item, name).text = text
            listings[f'DAS/H5/DASV/{day}/'] = ET.tostring(nodes)

        class Session:
            def get(self, url, params=None, **kwargs):
                return Response(listings[params['prefix']] if params else files[url.removeprefix(preparation.BASE)])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['prepare', '--raw-dir', str(root / 'raw'), '--samples-dir', str(root / 'samples'),
                    '--output-dir', str(root / 'results')]
            with patch.object(sys, 'argv', argv), patch.object(preparation.requests, 'Session', return_value=Session()), redirect_stdout(io.StringIO()):
                preparation.main()
            protocol = json.loads((root / 'results/protocol.json').read_text())
            summary = json.loads((root / 'results/summary.json').read_text())
            self.assertEqual(len(protocol['structural_exclusions']), 4)
            self.assertEqual(summary['source_files'], 60)
            self.assertEqual(summary['windows'], 60)
            self.assertTrue((root / 'results/COMPLETE').is_file())
            self.assertTrue(all(r['size'] < 65536 for r in protocol['selected_files']))
