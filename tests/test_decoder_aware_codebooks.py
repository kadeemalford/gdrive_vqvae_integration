import unittest

import torch

from finetuning.run_decoder_aware_codebooks import centroid_objective


class DecoderAwareCodebookTests(unittest.TestCase):
    def test_gradients_update_only_centroids_and_follow_the_requested_objective(self):
        decoder = torch.nn.Conv2d(4, 1, 1, bias=False).requires_grad_(False)
        decoder.weight.data[:] = torch.tensor([1., 0., 0., 0.]).view(1, 4, 1, 1)
        books = torch.nn.Parameter(torch.tensor([[[1., 2.], [10., 10.]], [[3., 4.], [10., 10.]]]))
        z = torch.zeros(1, 4, 2, 2, requires_grad=True)
        target = torch.zeros(1, 1, 2, 2)
        valid = torch.ones_like(target, dtype=torch.bool)
        wave = centroid_objective(z, books, None, None, decoder, target, valid, 'waveform')
        wave.backward()
        gradient = books.grad.clone()
        self.assertGreater(float(gradient[0, 0, 0]), 0)
        self.assertEqual(float(gradient[0, 0, 1]), 0)
        self.assertEqual(float(gradient[1].abs().sum()), 0)
        self.assertIsNone(z.grad)
        self.assertIsNone(decoder.weight.grad)
        books.grad = None
        latent = centroid_objective(z, books, None, None, decoder, target, valid, 'latent')
        latent.backward()
        self.assertGreater(float(books.grad[1, 0].abs().sum()), 0)
        with self.assertRaises(ValueError):
            centroid_objective(z, books, None, None, decoder, target, valid, 'unknown')
