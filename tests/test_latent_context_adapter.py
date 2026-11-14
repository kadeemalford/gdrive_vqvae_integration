import unittest

import torch

from finetuning.latent_context_adapter import KINDS, LatentContextAdapter, known_shape_features


class LatentContextAdapterTests(unittest.TestCase):
    def test_all_controls_start_as_exact_identity_and_have_similar_capacity(self):
        q = torch.randn(1, 128, 4, 5)
        counts = []
        for kind in KINDS:
            adapter = LatentContextAdapter(kind)
            self.assertTrue(torch.equal(adapter(q, (9, 73)), q))
            counts.append(sum(p.numel() for p in adapter.parameters()))
        self.assertLess(max(counts) / min(counts), 1.01)

    def test_shape_fractions_come_only_from_transmitted_dimensions(self):
        q = torch.zeros(1, 128, 16, 63)
        position, fractions = known_shape_features(q, (55, 1000))
        self.assertEqual(tuple(position.shape), (1, 32, 16, 63))
        self.assertAlmostEqual(float(fractions[0, 0, 0, 0]), 55 / 72, places=6)
        self.assertAlmostEqual(float(fractions[0, 1, 0, 0]), 1000 / 1008, places=6)
        with self.assertRaises(ValueError):
            known_shape_features(q, (100, 1000))

    def test_frozen_token_inputs_receive_no_gradients(self):
        q = torch.randn(1, 128, 4, 5)
        adapter = LatentContextAdapter('moments')
        adapter(q, (9, 73)).square().mean().backward()
        self.assertIsNone(q.grad)
        self.assertIsNotNone(adapter.output.weight.grad)
