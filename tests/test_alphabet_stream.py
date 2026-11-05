import importlib.util
import unittest

import numpy as np

from compression.alphabet_stream import AlphabetStream


@unittest.skipIf(importlib.util.find_spec('constriction') is None, 'Optional entropy dependency')
class AlphabetStreamTests(unittest.TestCase):
    def test_roundtrip_alphabets_and_budgets(self):
        for k in (32, 64, 2048):
            stream = AlphabetStream(np.ones(k, dtype=np.int64), '12' * 32)
            indices = np.random.default_rng(4).integers(k, size=111, dtype=np.int32)
            frame = stream.encode(indices, (17, 1000), (3, 37), b'side')
            actual, shape, latent, side = stream.decode(frame)
            np.testing.assert_array_equal(indices, actual)
            self.assertEqual((shape, latent, side), ((17, 1000), (3, 37), b'side'))
            with self.assertRaises(ValueError):
                stream.decode(frame + b'x')
            damaged = bytearray(frame)
            damaged[-1] ^= 1
            with self.assertRaises(ValueError):
                stream.decode(bytes(damaged))
            with self.assertRaises(ValueError):
                AlphabetStream(np.ones(k, dtype=np.int64), '13' * 32).decode(frame)

    def test_reject_out_of_alphabet_and_invalid_counts(self):
        with self.assertRaises(ValueError):
            AlphabetStream(np.ones(33), '12' * 32)
        with self.assertRaises(ValueError):
            AlphabetStream(np.ones(32), '12' * 32).encode([32], (1, 1), (1, 1), b'')

    def test_range_fallback_handles_highly_skewed_symbols(self):
        from compression.alphabet_stream import HEADER
        counts = np.ones(64, dtype=np.int64)
        counts[0] = 100000
        stream = AlphabetStream(counts, '12' * 32)
        indices = np.zeros(2048, dtype=np.int32)
        frame = stream.encode(indices, (32, 1000), (32, 64), b'side')
        self.assertEqual(HEADER.unpack_from(frame)[2], 1)
        np.testing.assert_array_equal(stream.decode(frame)[0], indices)
