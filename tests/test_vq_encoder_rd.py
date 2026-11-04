import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetuning.vq_encoder_rd import (PairedTileSampler, configure_encoder_only,
                                      frozen_forward, hard_rate_with_soft_gradient)


class EncoderRDTests(unittest.TestCase):
    def test_temperature_changes_only_the_gradient_not_the_hard_rate(self):
        a = torch.tensor([[.2, .21, 1.]], requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        costs = torch.tensor([1., 4., 8.])
        broad, _ = hard_rate_with_soft_gradient(a, costs, 1., 10)
        sharp, _ = hard_rate_with_soft_gradient(b, costs, .1, 10)
        torch.testing.assert_close(broad, sharp, rtol=0, atol=0)
        x = torch.autograd.grad(broad, a)[0]
        y = torch.autograd.grad(sharp, b)[0]
        self.assertFalse(torch.allclose(x, y))
        self.assertTrue(torch.isfinite(y).all())

    def test_rate_forward_is_hard_but_gradient_reaches_latent(self):
        z = torch.tensor([[.1, .4, 1.], [.8, .3, .2]], requires_grad=True)
        costs = torch.tensor([1., 2., 4.])
        rate, indices = hard_rate_with_soft_gradient(z, costs, .2, 10)
        torch.testing.assert_close(rate, costs[indices].sum() / 10, rtol=0, atol=0)
        rate.backward()
        self.assertGreater(float(z.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_sampler_preserves_exact_configuration_dose_and_regions(self):
        rows = [{'sample_path': f'{group}_{height}', 'split': 'train', 'family': 'f',
                 'configuration': group, 'channels': height, 'time_samples': 10}
                for group in ('a', 'b') for height in (2, 512)]
        a = PairedTileSampler(rows, 'window', 2081)
        b = PairedTileSampler(rows, 'configuration-cell', 2081)
        tiny_a = tiny_b = 0
        for _ in range(10000):
            x, y = a.draw(), b.draw()
            self.assertEqual(a.key(x), b.key(y))
            tiny_a += x['channels'] == 2
            tiny_b += y['channels'] == 2
        self.assertGreater(tiny_a, 4000)
        self.assertGreater(tiny_b, 0)
        self.assertLess(tiny_b, 100)
        with self.assertRaises(ValueError):
            PairedTileSampler([{**rows[0], 'split': 'test'}], 'window', 1)

    def test_only_encoder_changes_and_eval_uses_exact_vectors(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = torch.nn.Module()
        model.encoder = torch.nn.Conv2d(1, 4, 1)
        model.decoder = torch.nn.Conv2d(4, 1, 1)
        model.quantizer = torch.nn.Module()
        model.quantizer.embedding = torch.nn.Embedding(8, 4)
        model.downsample_levels = 1
        parameters = configure_encoder_only(model)
        frozen = {k: p.clone() for k, p in model.named_parameters() if not p.requires_grad}
        model.encoder.train()
        x = torch.randn(1, 1, 4, 7)
        costs = torch.arange(1., 9.)
        output, vq, rate, indices, latent, _ = frozen_forward(model, x, costs, .5)
        optimizer = torch.optim.SGD(parameters, lr=.01)
        (output.square().mean() + vq + .1 * rate).backward()
        optimizer.step()
        for name, p in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
                self.assertIsNone(p.grad)
        model.eval()
        with torch.no_grad():
            output, _, _, indices, latent, _ = frozen_forward(model, x, costs, .5)
            q = model.quantizer.embedding(indices).view(1, *latent, 4).permute(0, 3, 1, 2).contiguous()
            replay = model.decoder(q)[..., :4, :7]
            torch.testing.assert_close(output, replay, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
