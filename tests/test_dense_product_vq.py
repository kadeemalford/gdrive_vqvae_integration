import unittest
from unittest.mock import patch

import numpy as np
import torch

from compression.full_tile_vq_stream import FullTileVQWire
from finetuning.dense_product_vq import VERSION, build
from finetuning.factorized_vq import ProductEmbedding
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.run_dense_product_study import should_evaluate
from tools.run_dense_product_campaign import identical
from tests.test_joint_anchor_vq import ToyCodec


class ProductToy(ToyCodec):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(1, 128, 1)
        self.decoder = torch.nn.Conv2d(128, 1, 1)
        self.quantizer = ProductQuantizer()


class ProductQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = ProductEmbedding(torch.randn(32, 64), torch.randn(64, 64),
                                          torch.eye(128), torch.zeros(128))

    def forward(self, z):
        values = z.permute(0, 2, 3, 1).reshape(-1, 128)
        weight = self.embedding.weight
        ids = (values.square().sum(1, keepdim=True) - 2 * values @ weight.T + weight.square().sum(1)).argmin(1)
        q = self.embedding(ids).view(z.shape[0], z.shape[2], z.shape[3], 128).permute(0, 3, 1, 2)
        loss = (q - z.detach()).square().mean() + .25 * (q.detach() - z).square().mean()
        return q, loss, ids, torch.tensor(32.)


class DenseProductTests(unittest.TestCase):
    def test_both_books_train_and_hard_receiver_replays(self):
        torch.manual_seed(105)
        model = ProductToy().train()
        raw = np.random.default_rng(105).normal(size=(3, 17)).astype(np.float32)
        x = torch.from_numpy(raw)[None, None]
        predicted, loss, ids, latent, _ = exact_branch_forward(model, x)
        (predicted.square().mean() + loss).backward()
        for parameter in (model.quantizer.embedding.left, model.quantizer.embedding.right):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        wire = FullTileVQWire(model.eval(), np.ones(2048, np.int64), "0" * 64)
        self.assertEqual(wire.decode(wire.encode(raw)).shape, raw.shape)
        self.assertLess(int(ids.max()), 2048)

    def test_versioned_loader_preserves_product_books(self):
        model = ProductToy()
        config = {"architecture": VERSION, "variant": "product", "base_config": {}}
        with patch("finetuning.dense_product_vq.new_model", side_effect=lambda *args: ProductToy()):
            recovered = build(config, model.state_dict(), "cpu")
        self.assertTrue(identical(model.state_dict(), recovered.state_dict()))

    def test_nested_checkpoint_equality_detects_optimizer_changes(self):
        left = {"weights": torch.ones(3), "optimizer": {0: {"step": torch.tensor(1)}}}
        right = {"weights": torch.ones(3), "optimizer": {0: {"step": torch.tensor(2)}}}
        self.assertFalse(identical(left, right))

    def test_smoke_selection_events_match_continuous_and_resumed(self):
        direct = [i for i in (1, 2) if should_evaluate(i, 2, True)]
        resumed = [i for i in (1,) if should_evaluate(i, 1, True)] + [
            i for i in (2,) if should_evaluate(i, 2, True)]
        self.assertEqual(direct, resumed)
        self.assertEqual(direct, [1, 2])
        self.assertFalse(should_evaluate(1, 8192, False))
