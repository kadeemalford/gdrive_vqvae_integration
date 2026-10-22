import unittest
import numpy as np
from finetuning.robust_escape import robust_threshold, encode_robust
from finetuning.reversible_normalization import encode, decode


class RobustEscapeTests(unittest.TestCase):
    def test_residual_tail_is_escaped_and_regular_channels_preserved(self):
        raw = np.random.default_rng(1).normal(size=(512, 1000)).astype(np.float32) * 1e5
        raw[-2:] = 1e16
        original = raw.copy()
        baseline, robust = encode(raw), encode_robust(raw)
        self.assertTrue(baseline.valid.all())
        self.assertFalse(robust.valid[-2:].any())
        recovered = decode(robust.values, robust.side_stream)
        self.assertEqual(recovered[~robust.valid].tobytes(), raw[~robust.valid].tobytes())
        error = np.linalg.norm(recovered[robust.valid].astype(np.float64) - raw[robust.valid])
        self.assertLess(error / np.linalg.norm(raw[robust.valid]), 1e-6)
        self.assertGreater(len(robust.side_stream), len(baseline.side_stream))
        np.testing.assert_array_equal(raw, original)

    def test_sparse_real_signal_is_not_automatically_all_escaped(self):
        raw = np.zeros((4, 1000), dtype=np.float32)
        raw[0, 0] = 1000.
        self.assertGreater(robust_threshold(raw), 1000.)
        self.assertTrue(encode_robust(raw).valid.all())

    def test_zero_and_constant_roundtrip(self):
        for value in (0., 17.):
            raw = np.full((4, 1000), value, dtype=np.float32)
            packet = encode_robust(raw)
            np.testing.assert_array_equal(decode(packet.values, packet.side_stream), raw)

    def test_invalid_payload_rejected(self):
        for raw in (np.ones((4, 1000), dtype=np.float64), np.full((4, 1000), np.nan, dtype=np.float32)):
            with self.assertRaises(ValueError):
                encode_robust(raw)
