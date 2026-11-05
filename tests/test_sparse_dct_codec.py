import unittest
import numpy as np
from finetuning.sparse_dct_codec import compress, compress_packet, decompress, pack, unpack, HEADER


class SparseDctCodecTests(unittest.TestCase):
    def test_preprotected_packet_preserves_legacy_and_robust_contract(self):
        from finetuning.reversible_normalization import encode
        from finetuning.robust_escape import encode_robust
        raw = np.random.default_rng(13).normal(size=(4, 1000)).astype(np.float32)
        self.assertEqual(compress(raw, 500)[0], compress_packet(encode(raw), 500)[0])
        raw[3, 10] = 1e35
        packet = encode_robust(raw)
        frame, _ = compress_packet(packet, 500)
        restored = decompress(frame)
        np.testing.assert_array_equal(raw.view(np.uint32)[~packet.valid], restored.view(np.uint32)[~packet.valid])
        self.assertLessEqual(len(frame), 500)

    def test_exact_bit_packing(self):
        for width in (3, 8, 13, 27):
            values = np.array([0, 1, 2 ** width - 1], dtype=np.uint32)
            np.testing.assert_array_equal(unpack(pack(values, width), width, len(values)), values)

    def test_budget_and_constant_reconstruction(self):
        raw = np.full((4, 1000), 3, dtype=np.float32)
        stream, stats = compress(raw, 300)
        self.assertLessEqual(len(stream), 300)
        self.assertEqual(stats["encoded_bytes"], len(stream))
        np.testing.assert_array_equal(decompress(stream), raw)

    def test_escape_bit_exact_and_positive_rate(self):
        raw = np.random.default_rng(1).normal(size=(4, 1000)).astype(np.float32)
        raw[1, 10] = -1e35
        stream, _ = compress(raw, 500)
        restored = decompress(stream)
        self.assertEqual(restored[1, 10].tobytes(), raw[1, 10].tobytes())
        self.assertTrue(np.isfinite(restored).all())

    def test_more_coefficients_increase_prequantization_energy(self):
        raw = np.random.default_rng(2).normal(size=(4, 1000)).astype(np.float32)
        _, low = compress(raw, 300)
        _, high = compress(raw, 1500)
        self.assertGreater(high["retained_coefficients"], low["retained_coefficients"])
        self.assertGreaterEqual(high["prequantization_energy_fraction_retained"], low["prequantization_energy_fraction_retained"])

    def test_corrupt_stream_refused(self):
        stream, _ = compress(np.ones((4, 1000), np.float32), 300)
        for corrupted in (b"", stream[:-1], b"BAD!" + stream[4:], stream + b"x"):
            with self.assertRaises(ValueError):
                decompress(corrupted)

    def test_no_coefficient_budget_valid(self):
        raw = np.random.default_rng(3).normal(size=(4, 1000)).astype(np.float32)
        stream, stats = compress(raw, HEADER.size + 40)
        self.assertEqual(stats["retained_coefficients"], 0)
        restored = decompress(stream)
        self.assertLess(float(restored.std()), 1e-5)
