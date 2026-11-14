import unittest

import torch
import torch.nn.functional as F

from finetuning.analysis_context import analysis_context_forward
from finetuning.latent_context_adapter import LatentContextAdapter
from finetuning.run_index_search import vectors


class TinyQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(2, 128)
        with torch.no_grad():
            self.embedding.weight[0].zero_(); self.embedding.weight[1].fill_(1.)

    def forward(self, features):
        flat = features.permute(0, 2, 3, 1).reshape(-1, 128)
        indices = (flat.mean(dim=1) > .5).long()
        exact = vectors(indices, self.embedding.weight, features.shape[-2:])
        loss = .25 * F.mse_loss(features, exact.detach())
        quantized = features + (exact - features).detach() if self.training else exact + .125
        return quantized, loss, indices, features.new_tensor(2.)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quantizer = TinyQuantizer()
        self.decoder = torch.nn.Conv2d(128, 1, 1, bias=False)
        with torch.no_grad():
            self.decoder.weight.fill_(1 / 128)


class AnalysisContextTests(unittest.TestCase):
    def test_evaluation_uses_exact_code_vectors_not_continuous_or_st_remainders(self):
        model = TinyModel().eval().requires_grad_(False)
        adapter = LatentContextAdapter('local')
        features = torch.full((1, 128, 2, 3), .8)
        with torch.no_grad():
            reconstructed, _, indices, _ = analysis_context_forward(model, adapter, features, (2, 3))
            expected = model.decoder(vectors(indices, model.quantizer.embedding.weight, (2, 3)))
        self.assertTrue(torch.equal(reconstructed, expected))
        self.assertTrue(torch.equal(reconstructed, torch.ones_like(reconstructed)))
        self.assertFalse(model.quantizer.training)

    def test_training_gradient_reaches_only_analysis_adapter_through_st(self):
        model = TinyModel().eval().requires_grad_(False)
        adapter = LatentContextAdapter('axial')
        features = torch.full((1, 128, 2, 3), .8)
        output, vq, _, _ = analysis_context_forward(model, adapter, features, (2, 3), training=True)
        (output.square().mean() + vq).backward()
        self.assertTrue(model.quantizer.training)
        self.assertIsNone(features.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertGreater(float(adapter.output.weight.grad.abs().sum()), 0.)

    def test_analysis_adapter_can_change_ids_without_continuous_decoder_side(self):
        model = TinyModel().eval().requires_grad_(False)
        adapter = LatentContextAdapter('local')
        features = torch.full((1, 128, 2, 3), .8)
        with torch.no_grad():
            _, _, initial_ids, _ = analysis_context_forward(model, adapter, features, (2, 3))
            adapter.output.bias.fill_(-1.)
            decoded, _, new_ids, _ = analysis_context_forward(model, adapter, features, (2, 3))
        self.assertTrue(torch.equal(initial_ids, torch.ones_like(initial_ids)))
        self.assertTrue(torch.equal(new_ids, torch.zeros_like(new_ids)))
        self.assertTrue(torch.equal(decoded, torch.zeros_like(decoded)))
