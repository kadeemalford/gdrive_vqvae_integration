import unittest
import numpy as np
from finetuning.channel_normalization import encode_channel, decode_channel, HEADER, parse_channel_stream


class ChannelNormalizationTests(unittest.TestCase):
    def test_roundtrip_scales_offsets_and_exact_escape(self):
        raw = np.random.default_rng(1).normal(size=(8, 1000)).astype(np.float32)
        raw *= np.logspace(-3, 6, 8).astype(np.float32)[:, None]
        raw += np.arange(8, dtype=np.float32)[:, None] * 100
        raw[-1, 0] = 1e35
        packet = encode_channel(raw)
        restored = decode_channel(packet.values, packet.side_stream)
        self.assertEqual(restored[~packet.valid].tobytes(), raw[~packet.valid].tobytes())
        # Ordinary cells are lossy. Near-zero values can have a large relative
        # point error after offset subtraction; bound error against channel scale.
        for channel in range(raw.shape[0]):
            valid = packet.valid[channel]
            scale = raw[channel, valid].astype(np.float64).std()
            error = np.abs(restored[channel, valid].astype(np.float64) - raw[channel, valid])
            self.assertLess(float(error.max()), 5e-7 * scale + 1e-6)
        self.assertEqual(len(packet.side_stream), HEADER.size + 8 * 8 + 8)

    def test_constant_and_fully_escaped_channel(self):
        raw = np.ones((3, 1000), np.float32)
        raw[0] = 2
        raw[1] = -1e30
        packet = encode_channel(raw)
        np.testing.assert_array_equal(decode_channel(packet.values, packet.side_stream), raw)
        self.assertTrue((packet.values[1] == 0).all())

    def test_distinct_channel_scale_normalized(self):
        raw = np.random.default_rng(2).normal(size=(3, 1000)).astype(np.float32)
        raw *= np.array([1, 10, 1000], np.float32)[:, None]
        packet = encode_channel(raw)
        np.testing.assert_allclose(packet.values.std(axis=1), np.ones(3), atol=3e-6)

    def test_bad_stream_rejected(self):
        packet = encode_channel(np.ones((3, 1000), np.float32))
        for stream in (b"", packet.side_stream[:-1], packet.side_stream + b"x"):
            with self.assertRaises(ValueError):
                parse_channel_stream(stream)
