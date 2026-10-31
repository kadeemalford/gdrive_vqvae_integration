import unittest
import numpy as np

from compression.symbol_stream import SymbolCodec, CanonicalTable, code_lengths


class SymbolStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        counts = np.ones(2048, dtype=np.int64)
        counts[7] = 100000
        cls.codec = SymbolCodec.from_counts(counts, np.tile(counts, (17, 1)), 'a' * 64)

    def test_roundtrip_random_constant_rare_and_all_symbols(self):
        rng = np.random.default_rng(301)
        for symbols in (np.zeros(9, dtype=int), np.full(15, 7),
                        rng.integers(0, 2048, 103), np.arange(2048)):
            frame, _ = self.codec.encode(symbols, (1, 1000), (1, len(symbols)), b'normalization')
            recovered, shape, latent, side = self.codec.decode(frame)
            np.testing.assert_array_equal(symbols, recovered)
            self.assertEqual(shape, (1, 1000))
            self.assertEqual(latent, (1, len(symbols)))
            self.assertEqual(side, b'normalization')

    def test_no_payload_expansion_beyond_fixed_width(self):
        from compression.symbol_stream import HEADER
        symbols = np.arange(2048)
        frame, _ = self.codec.encode(symbols, (32, 64), (32, 64), b'')
        self.assertLessEqual(len(frame) - HEADER.size, (len(symbols) * 11 + 7) // 8)

    def test_tables_portable_and_corruption_rejected(self):
        codec = SymbolCodec.from_bytes(self.codec.serialized)
        frame, _ = codec.encode(np.full(11, 7), (1, 1000), (1, 11), b'')
        damaged = bytearray(frame); damaged[-1] ^= 16
        with self.assertRaises(ValueError):
            codec.decode(bytes(damaged))
        with self.assertRaises(ValueError):
            codec.decode(frame[:10])

    def test_invalid_codes_and_shapes(self):
        with self.assertRaises(ValueError):
            CanonicalTable([1] * 2048)
        with self.assertRaises(ValueError):
            self.codec.encode([2048], (1, 1), (1, 1), b'')
        with self.assertRaises(ValueError):
            code_lengths(np.full(2048, -1))

    def test_conditional_coder_is_selected_and_roundtrips(self):
        counts = np.zeros(2048, dtype=np.int64)
        counts[[0, 128]] = 100000
        context = np.zeros((17, 2048), dtype=np.int64)
        context[16, 0] = context[0, 128] = context[1, 0] = 100000
        codec = SymbolCodec.from_counts(counts, context, 'b'*64)
        symbols = np.tile([0, 128], 1000)
        frame, choice = codec.encode(symbols, (2, 1000), (2, 1000), b'')
        self.assertEqual(choice, 2)
        np.testing.assert_array_equal(codec.decode(frame)[0], symbols)


if __name__ == '__main__':
    unittest.main()
