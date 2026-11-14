import json
from pathlib import Path
import tempfile
import unittest
import struct

import h5py
import numpy as np

from acquisition.audit_porotomo_pretraining import resolve_metadata_references, sampling_from_file
from acquisition.audit_porotomo_segy import native_traces
from acquisition.prepare_porotomo_source import spread_selection, normalize_rate


class PoroTomoMetadataTests(unittest.TestCase):
    def test_hdf5_dimension_references_are_serialized_as_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / 'fixture.h5', 'w') as handle:
                target = handle.create_dataset('time', data=[0, 1])
                metadata = {'dimension_list': [[target.ref]], 'null': h5py.Reference()}
                resolved = resolve_metadata_references(metadata, handle)
                self.assertEqual(resolved['dimension_list'], [[{'hdf5_reference': '/time'}]])
                self.assertEqual(resolved['null'], {'hdf5_reference': None})
                self.assertEqual(json.loads(json.dumps(resolved)), resolved)

    def test_file_timestamp_units_override_readme_assumptions(self):
        timestamps = 1_458_295_397.99 + np.arange(30000) * .001
        rate, steps = sampling_from_file(timestamps, 'seconds since 1970-01-01', .001)
        self.assertEqual(rate, 1000.)
        self.assertAlmostEqual(float(np.mean(steps)), .001, places=8)
        rate, _ = sampling_from_file(np.arange(30000) * 1000, 'microseconds since start', .001)
        self.assertEqual(rate, 1000.)
        with self.assertRaises(ValueError):
            sampling_from_file(timestamps, 'microseconds since start', .001)

    def test_native_ieee_traces_keep_time_channel_order(self):
        header = bytearray(3600)
        for offset, value in [(3216, 1000), (3220, 3), (3224, 5)]:
            struct.pack_into('>H', header, offset, value)
        payload = bytes(header) + bytes(240) + np.array([1, 2, 3], dtype='>f4').tobytes()
        payload += bytes(240) + np.array([4, 5, 6], dtype='>f4').tobytes()
        traces, interval = native_traces(payload)
        np.testing.assert_array_equal(traces, [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(interval, 1000)
        with self.assertRaises(ValueError):
            native_traces(payload[:-1])

    def test_source_selection_and_normalization_are_model_independent(self):
        selected = spread_selection([{'key':f'{i:03d}'} for i in range(100)], 15)
        self.assertEqual(selected[0]['key'], '000')
        self.assertEqual(selected[-1]['key'], '099')
        self.assertEqual(len({r['key'] for r in selected}), 15)
        values = np.arange(100, dtype=np.float64).reshape(20, 5) * 1e-10
        a, _, _ = normalize_rate(values)
        b, _, _ = normalize_rate(values * 1000)
        np.testing.assert_allclose(a,b,atol=1e-7)
        self.assertEqual(a.shape,(5,20))
        with self.assertRaises(ValueError):
            normalize_rate(np.array([[np.nan,1.]]))
