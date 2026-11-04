import unittest
import zlib

import numpy as np

from compression.partition_bundle import pack_bundle, unpack_bundle, exact_nonfinite, decode_bundle


class PartitionBundleTests(unittest.TestCase):
    def test_decoder_rejects_finite_signal_hidden_in_exact_fragment(self):
        raw = np.zeros((1, 20), dtype=np.float32)
        bundle = pack_bundle(raw.shape, [(1, 0, 1, zlib.compress(raw.tobytes()))])
        with self.assertRaises(ValueError):
            decode_bundle(bundle, None, None, 'cpu')

    def test_exact_fragments_preserve_nan_payloads_and_coverage(self):
        raw = np.zeros((3, 20), dtype=np.float32)
        raw.view(np.uint32)[:] = 0x7fffffff
        bundle = pack_bundle(raw.shape, [(1, 0, 1, exact_nonfinite(raw[:1])), (1, 1, 2, exact_nonfinite(raw[1:]))])
        np.testing.assert_array_equal(decode_bundle(bundle, None, None, 'cpu').view(np.uint32), raw.view(np.uint32))
        self.assertEqual(unpack_bundle(bundle)[0], raw.shape)

    def test_no_gaps_overlaps_or_silent_finite_bypass(self):
        with self.assertRaises(ValueError):
            pack_bundle((3, 20), [(0, 1, 2, b'payload')])
        with self.assertRaises(ValueError):
            pack_bundle((3, 20), [(0, 0, 2, b'payload')])
        with self.assertRaises(ValueError):
            exact_nonfinite(np.zeros((1, 20), dtype=np.float32))

    def test_corruption_is_detected(self):
        raw = np.full((1, 20), np.nan, dtype=np.float32)
        bundle = pack_bundle(raw.shape, [(1, 0, 1, exact_nonfinite(raw))])
        for bad in (bundle[:-1], bundle + b'x', bundle[:-5] + bytes([bundle[-5] ^ 1]) + bundle[-4:]):
            with self.assertRaises(ValueError):
                unpack_bundle(bad)
