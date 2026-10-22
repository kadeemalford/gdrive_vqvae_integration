import unittest
import numpy as np
from finetuning.reversible_normalization import encode, decode, parse_side_stream, HEADER


class ReversibleNormalizationTests(unittest.TestCase):
    def test_extreme_bits_preserved_and_ordinary_variation_retained(self):
        raw = np.random.default_rng(1).normal(size=(8, 1000)).astype(np.float32)
        raw[7, 10:12] = [1e35, -1e25]
        original = raw.copy()
        packet = encode(raw)
        restored = decode(packet.values, packet.side_stream)
        self.assertEqual(restored[~packet.valid].tobytes(), raw[~packet.valid].tobytes())
        np.testing.assert_allclose(restored[packet.valid], raw[packet.valid], atol=3e-7, rtol=3e-7)
        self.assertTrue(np.all(np.ptp(packet.values[:7], axis=1) > 0))
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(len(packet.side_stream), HEADER.size + 16)

    def test_escape_overwrites_arbitrary_model_predictions(self):
        raw = np.ones((2, 1000), dtype=np.float32)
        raw[0, 0] = -1e30
        packet = encode(raw)
        restored = decode(np.full(raw.shape, 5, dtype=np.float32), packet.side_stream)
        self.assertEqual(restored[0, 0].tobytes(), raw[0, 0].tobytes())

    def test_no_escape_matches_legacy_normalization(self):
        raw = np.random.default_rng(2).normal(size=(4, 1000)).astype(np.float32)
        packet = encode(raw)
        expected = ((raw.astype(np.float64) - raw.mean(dtype=np.float64)) / (raw.std(dtype=np.float64) + 1e-6)).astype(np.float32)
        np.testing.assert_array_equal(packet.values, expected)
        self.assertEqual(len(packet.side_stream), HEADER.size)

    def test_rejects_invalid_inputs(self):
        for raw in [np.full((2, 4), np.nan, np.float32), np.full((2, 4), 1e30, np.float32), np.ones((2, 4), np.float64)]:
            with self.assertRaises(ValueError):
                encode(raw)

    def test_rejects_corrupt_stream_and_wrong_shape(self):
        packet = encode(np.ones((2, 4), np.float32))
        for stream in [b"", packet.side_stream[:-1], b"XXXX" + packet.side_stream[4:], packet.side_stream + b"x"]:
            with self.assertRaises(ValueError):
                parse_side_stream(stream)
        with self.assertRaises(ValueError):
            decode(np.ones((4, 2)), packet.side_stream)
