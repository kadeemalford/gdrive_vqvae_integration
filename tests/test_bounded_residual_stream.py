import unittest

import numpy as np

from compression.bounded_residual_stream import audit, decode, encode


class BoundedResidualTests(unittest.TestCase):
    def test_roundtrip_at_multiple_scales_and_shapes(self):
        rng = np.random.default_rng(122)
        for shape in ((1, 1), (3, 17), (16, 64)):
            for scale in (1e-20, 1., 1e20):
                raw = (rng.normal(size=shape) * scale).astype(np.float32)
                base = (rng.normal(size=shape) * scale).astype(np.float32)
                for fraction in (.1, .03, .01):
                    epsilon = fraction * scale
                    frame = encode(raw, base, epsilon)
                    audit(raw, decode(frame, base.copy()), epsilon)

    def test_extremes_special_bits_and_tiny_tolerance(self):
        raw = np.array([0x00000000, 0x80000000, 0x7fc01234, 0xffc04321,
                        0x7f800000, 0xff800000, 0x7f7fffff, 1], np.uint32).view(np.float32)[None]
        base = np.full_like(raw, np.inf)
        result = decode(encode(raw, base, 1e-50), base)
        audit(raw, result, 1e-50)
        np.testing.assert_array_equal(raw.view(np.uint32), result.view(np.uint32))

    def test_invalid_base_and_corruption_rejected(self):
        raw = np.ones((2, 3), np.float32)
        frame = encode(raw, raw, .1)
        with self.assertRaises(ValueError):
            decode(frame, raw + 1)
        with self.assertRaises(ValueError):
            decode(frame[:-1], raw)
        for epsilon in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                encode(raw, raw, epsilon)

    def test_float_rounding_is_checked_after_restoration(self):
        raw = np.array([[1e12, -1e12, 1.2345, -7.1]], np.float32)
        base = np.zeros_like(raw)
        result = decode(encode(raw, base, 1e-8), base)
        audit(raw, result, 1e-8)

    def test_audit_does_not_accept_nan_for_finite_target(self):
        with self.assertRaises(AssertionError):
            audit(np.ones((1, 1), np.float32), np.full((1, 1), np.nan, np.float32), .1)
