import importlib.util
import unittest

import numpy as np
import torch

from compression.product_quantization import quantize_product, replay_product


class ProductQuantizationTests(unittest.TestCase):
    def test_orthogonal_rotation_replays_same_coded_vectors(self):
        generator = torch.Generator().manual_seed(18)
        books = torch.randn(2, 32, 4, generator=generator)
        basis, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
        center = torch.randn(8, generator=generator)
        z = torch.randn(1, 8, 3, 7, generator=generator)
        symbols, q = quantize_product(z, books, basis, center)
        torch.testing.assert_close(q, replay_product(symbols, (3, 7), books, basis, center), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            quantize_product(z, books, basis * 2, center)
        with self.assertRaises(ValueError):
            quantize_product(z, books, basis, None)

    def test_exact_combinatorial_replay(self):
        books = torch.tensor([[[0., 0.], [1., 1.]], [[-1., -1.], [2., 2.]]])
        symbols = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        z = replay_product(symbols, (2, 2), books)
        recovered, hard = quantize_product(z, books)
        torch.testing.assert_close(symbols, recovered, rtol=0, atol=0)
        torch.testing.assert_close(z, hard, rtol=0, atol=0)

    def test_fail_closed_on_bad_shape_alphabet_or_symbols(self):
        books = torch.zeros(2, 32, 4)
        for bad in (torch.zeros(3, 2, dtype=torch.long), torch.full((4, 2), 32, dtype=torch.long)):
            with self.assertRaises(ValueError):
                replay_product(bad, (2, 2), books)
        with self.assertRaises(ValueError):
            quantize_product(torch.zeros(1, 7, 2, 2), books)
        with self.assertRaises(ValueError):
            quantize_product(torch.zeros(1, 8, 2, 2), torch.full_like(books, float('nan')))

    @unittest.skipIf(importlib.util.find_spec('constriction') is None, 'Optional entropy dependency')
    def test_interleaved_framed_symbols_replay_without_original_latents(self):
        from compression.alphabet_stream import AlphabetStream
        books = torch.randn(2, 32, 4, generator=torch.Generator().manual_seed(4))
        z = torch.randn(1, 8, 3, 7, generator=torch.Generator().manual_seed(7))
        symbols, hard = quantize_product(z, books)
        stream = AlphabetStream(np.ones(32), '23' * 32)
        frame = stream.encode(symbols.numpy().reshape(-1), (17, 1000), (3, 14), b'side')
        indices, shape, symbol_grid, side = stream.decode(frame)
        restored = replay_product(torch.from_numpy(indices.astype(np.int64)).view(-1, 2),
                                  (symbol_grid[0], symbol_grid[1] // 2), books)
        torch.testing.assert_close(hard, restored, rtol=0, atol=0)
        self.assertEqual((shape, side), ((17, 1000), b'side'))
