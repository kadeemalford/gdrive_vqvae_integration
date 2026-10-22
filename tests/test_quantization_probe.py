import unittest

import torch

from finetuning.quantization_probe import QuantizationProbe, payload_bits


class QuantizationProbeTests(unittest.TestCase):
    def test_continuous_replaces_latents_but_keeps_reference_indices(self):
        latent = torch.ones(1, 2, 3, 4)
        quantized = latent * .5
        output = (quantized, torch.tensor(.1), torch.zeros(12, dtype=torch.long), torch.tensor(1.))
        vq = QuantizationProbe()
        self.assertIs(vq(None, (latent,), output), output)
        self.assertAlmostEqual(vq.latest["relative_quantization_error"], .5)
        continuous = QuantizationProbe(True)
        result = continuous(None, (latent,), output)
        self.assertIs(result[0], latent)
        self.assertIs(result[2], output[2])
        self.assertEqual(result[1].item(), 0.)

    def test_rate_charges_all_continuous_components(self):
        self.assertEqual(payload_bits(32000, 64, 2048, False), 352000)
        self.assertEqual(payload_bits(32000, 64, 2048, True), 65536000)
        with self.assertRaises(ValueError):
            payload_bits(0, 64, 2048, True)

    def test_invalid_reference_refused(self):
        probe = QuantizationProbe(True)
        latent = torch.ones(1, 2, 3, 4)
        with self.assertRaises(ValueError):
            probe(None, (latent,), (latent, torch.tensor(0.), torch.zeros(3), torch.tensor(1.)))
