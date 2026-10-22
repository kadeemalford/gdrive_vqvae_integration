import unittest

import torch

from finetuning.norm_matched_probe import NormMatchedProbe


class NormMatchedProbeTests(unittest.TestCase):
    def test_matches_quantized_norm_and_preserves_encoder_direction(self):
        z = torch.arange(1., 25.).reshape(1, 2, 3, 4)
        q = torch.ones_like(z) * 2
        output = (q, torch.tensor(.1), torch.zeros(12, dtype=torch.long), torch.tensor(1.))
        probe = NormMatchedProbe()
        result = probe(None, (z,), output)
        torch.testing.assert_close(result[0].norm(), q.norm())
        torch.testing.assert_close(result[0] / z, torch.ones_like(z) * probe.latest["norm_matching_scale"])
        self.assertIs(result[2], output[2])
        self.assertEqual(result[1].item(), 0.)

    def test_zero_norm_behavior(self):
        z = torch.zeros(1, 2, 3, 4)
        output = (z, torch.tensor(0.), torch.zeros(12, dtype=torch.long), torch.tensor(1.))
        self.assertEqual(NormMatchedProbe()(None, (z,), output)[0].norm().item(), 0.)
        with self.assertRaises(ValueError):
            NormMatchedProbe()(None, (z,), (z + 1, *output[1:]))
