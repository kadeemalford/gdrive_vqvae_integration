import unittest

import torch

from finetuning.run_alphabet_factorization import lloyd, ARMS


class AlphabetFactorizationTests(unittest.TestCase):
    def test_train_initialization_is_finite_and_reproducible(self):
        data = torch.randn(300, 8, generator=torch.Generator().manual_seed(5))
        first = lloyd(data, 32)
        torch.testing.assert_close(first, lloyd(data, 32), rtol=0, atol=0)
        self.assertEqual(tuple(first.shape), (32, 8))
        self.assertTrue(torch.isfinite(lloyd(torch.zeros_like(data), 32)).all())

    def test_grid_and_alphabet_have_separate_controls(self):
        self.assertEqual(ARMS['compact64'][1], ARMS['dense64'][1])
        self.assertEqual(ARMS['dense64'][0], ARMS['dense32'][0])
        self.assertEqual(ARMS['compact2048'][0], ARMS['compact64'][0])
