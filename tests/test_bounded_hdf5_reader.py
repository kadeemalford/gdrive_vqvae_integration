import io
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from acquisition.audit_external_hdf5_headers import BoundedRangeReader, describe_metadata, acquisition_screen
from acquisition.summarize_external_acquisitions import merge_intervals, overlap_seconds, metadata_intervals


class BoundedReaderTests(unittest.TestCase):
    def test_clock_interval_union_does_not_count_overlapped_blocks_twice(self):
        self.assertEqual(merge_intervals([(0., 1.05), (1., 2.05)]), [[0., 2.05]])
        self.assertAlmostEqual(overlap_seconds([(0., 1.05), (1., 2.05)], [(1.5, 3.)]), .55)
        self.assertEqual(overlap_seconds([(0., 1.)], [(1., 2.)]), 0.)

    def test_clock_intervals_use_spacing_origin_and_block_shape(self):
        record = {'objects': [
            {'path': 'i/Source1/Zone1', 'attributes': {'Spacing': [.2, .05, 1.], 'Origin': [0., -525., 0.]}},
            {'path': 'i/Source1/time', 'attributes': {}, 'clock_metadata_values': [100., 101.]},
            {'path': 'i/Source1/Zone1/Strain Rate [nStrain|s]', 'attributes': {}, 'shape': [2, 21000, 551]}]}
        intervals, schema = metadata_intervals(record)
        self.assertAlmostEqual(intervals[0][0], 99.475)
        self.assertAlmostEqual(intervals[0][1], 101.525)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(schema[0]['dt_s'], .00005)

    def test_metadata_visitor_reads_only_whitelisted_clock_arrays(self):
        memory = io.BytesIO()
        with h5py.File(memory, 'w') as handle:
            source = handle.create_group('instrument/Source1')
            source.create_dataset('time', data=np.array([100., 101.]))
            source.create_dataset('Zone1/Strain Rate [nStrain|s]', data=np.ones((2, 4, 3)))
            original = h5py.Dataset.__getitem__
            accesses = []
            def checked(dataset, key, **kwargs):
                accesses.append(dataset.name)
                if not dataset.name.endswith('/Source1/time'):
                    raise AssertionError('Signal dataset array access forbidden')
                return original(dataset, key, **kwargs)
            with patch.object(h5py.Dataset, '__getitem__', checked):
                result = describe_metadata(handle, True)
            self.assertEqual(accesses, ['/instrument/Source1/time'])
            self.assertEqual(next(o['clock_metadata_values'] for o in result['objects'] if 'clock_metadata_values' in o), [100., 101.])

    def test_acquisition_matches_are_conservative_not_independence_claims(self):
        def record(role, id, clock):
            return {'status': 'metadata_read', 'role': role, 'id': id,
                    'objects': [{'attributes': {'Hostname': 'instrument', 'AcqStartTime': [10]}, 'clock_metadata_values': [clock]}]}
        r = acquisition_screen([record('development', 'old', 100.), record('candidate', 'new', 101.)])
        self.assertEqual(r['candidate_shared_development_acq_attribute_count'], 1)
        self.assertEqual(r['candidate_shared_development_recorded_clock_count'], 0)
        self.assertFalse(r['independence_proven'])

    def test_seeks_cache_and_budget(self):
        data = bytes(range(128))
        class Session:
            def get(self, url, headers, **kwargs):
                start, end = map(int, headers['Range'][6:].split('-'))
                class Response:
                    status_code = 206
                    headers = {'Content-Range': f'bytes {start}-{end}/128'}
                    raw = io.BytesIO(data[start:end + 1])
                    def close(self):
                        pass
                return Response()
        reader = BoundedRangeReader(Session(), 'unused', 128, byte_budget=32, block_size=16)
        self.assertEqual(reader.read(8), data[:8])
        reader.seek(0)
        self.assertEqual(reader.read(20), data[:20])
        self.assertEqual(reader.fetched_bytes, 32)
        self.assertEqual(reader.requests, 2)
        with self.assertRaises(ValueError):
            reader.read(20)
        with self.assertRaises(ValueError):
            reader.read()

    def test_full_file_response_is_rejected_before_body_read(self):
        class Session:
            def get(self, *args, **kwargs):
                class Response:
                    status_code = 200
                    def close(self):
                        pass
                return Response()
        reader = BoundedRangeReader(Session(), 'unused', 128)
        with self.assertRaises(ValueError):
            reader.read(8)
        self.assertEqual(reader.fetched_bytes, 0)
