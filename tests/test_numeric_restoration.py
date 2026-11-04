import unittest

import numpy as np

from compression.numeric_restoration import encode_numeric, parse_numeric_side, COMPRESSED_MAGIC, parse_input_side
from compression.affine_restoration import restore, affine_side
from finetuning.robust_escape import encode_robust
from finetuning.run_external_smf_stress import pooled_score


class NumericRestorationTests(unittest.TestCase):
    def test_pooled_metrics_score_only_ordinary_values_but_keep_escape_bits(self):
        raw = np.random.default_rng(4).normal(size=(5, 2000)).astype(np.float32)
        raw[0].view(np.uint32)[:] = 0x7fffffff
        valid = np.isfinite(raw)
        result = pooled_score(raw, raw.copy(), valid)
        self.assertEqual(result['escaped_cells'], 2000)
        self.assertEqual(result['ordinary_cells'], 8000)
        self.assertEqual(result['relative_rmse'], 0.)
        self.assertEqual(result['seam_pairs'], 4)

    def test_nonfinite_payload_bits_survive_identity_and_affine(self):
        raw = np.arange(10000, dtype=np.float32).reshape(10, 1000)
        patterns = np.array([0x7fffffff, 0x7fc01234, 0xffc05678, 0x7f800000, 0xff800000], dtype=np.uint32)
        raw.reshape(-1).view(np.uint32)[:5] = patterns
        packet = encode_numeric(raw)
        for side in (packet.side_stream, affine_side(packet.side_stream, 1.1, .2)):
            result = restore(packet.values, side)
            np.testing.assert_array_equal(result.reshape(-1).view(np.uint32)[:5], patterns)
            self.assertTrue(np.isfinite(result[packet.valid]).all())

    def test_finite_input_is_exactly_legacy_compatible(self):
        raw = np.arange(20, dtype=np.float32).reshape(4, 5)
        a, b = encode_numeric(raw), encode_robust(raw)
        self.assertEqual(a.side_stream, b.side_stream)
        np.testing.assert_array_equal(a.values, b.values)

    def test_contiguous_nonfinite_rows_compress_without_imputation(self):
        raw = np.ones((10, 1000), dtype=np.float32)
        raw[:2].view(np.uint32)[:] = 0x7fffffff
        packet = encode_numeric(raw)
        self.assertEqual(packet.side_stream[:4], COMPRESSED_MAGIC)
        self.assertLess(len(packet.side_stream), 1000)
        result = restore(packet.values, packet.side_stream)
        np.testing.assert_array_equal(result.view(np.uint32), raw.view(np.uint32))

    def test_truncation_and_trailing_payload_are_rejected(self):
        raw = np.ones((4, 100), dtype=np.float32)
        raw[0].view(np.uint32)[:] = 0x7fffffff
        side = encode_numeric(raw).side_stream
        for bad in (side[:-1], side + b'x', side[:5]):
            with self.assertRaises(ValueError):
                parse_numeric_side(bad)

    def test_no_ordinary_target_fails_closed(self):
        with self.assertRaises(ValueError):
            encode_numeric(np.full((2, 20), np.nan, dtype=np.float32))
