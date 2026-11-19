import unittest
from unittest.mock import patch

import torch
from torch import nn

from finetuning.local_internal_norm import LocalResidualGroupNorm, initialize
from finetuning.dense_product_vq import build as generic_build
from tests.test_joint_anchor_vq import ToyCodec


class NormToy(ToyCodec):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 1), nn.GroupNorm(4, 16), nn.Conv2d(16, 1, 1))
        self.decoder = nn.Sequential(nn.Conv2d(1, 16, 1), nn.GroupNorm(4, 16), nn.Conv2d(16, 1, 1))


class LocalInternalNormTests(unittest.TestCase):
    def test_neutral_function_and_finite_nonzero_gate_gradient(self):
        torch.manual_seed(115)
        original = nn.GroupNorm(4, 16)
        local = LocalResidualGroupNorm(original, True)
        x = torch.randn(1, 16, 3, 17, requires_grad=True)
        self.assertTrue(torch.equal(original(x).view(torch.int32), local(x).view(torch.int32)))
        (local(x) * torch.randn_like(x)).sum().backward()
        self.assertTrue(torch.isfinite(local.gate.grad))
        self.assertNotEqual(float(local.gate.grad), 0.)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_control_is_original_and_gate_is_not_trainable(self):
        original = nn.GroupNorm(4, 16)
        control = LocalResidualGroupNorm(original, False)
        x = torch.randn(1, 16, 2, 17)
        self.assertTrue(torch.equal(original(x), control(x)))
        self.assertFalse(control.gate.requires_grad)
        self.assertEqual(float(control.gate), 0.)

    def test_local_path_statistics_do_not_pool_other_positions(self):
        original = nn.GroupNorm(4, 16)
        local = LocalResidualGroupNorm(original, True)
        x = torch.randn(1, 16, 2, 17)
        changed = x.clone()
        changed[:, :, -1, -1] += 100
        self.assertTrue(torch.equal(local.local_values(x)[:, :, 0, 0],
                                    local.local_values(changed)[:, :, 0, 0]))
        self.assertFalse(torch.equal(original(x)[:, :, 0, 0], original(changed)[:, :, 0, 0]))
        with torch.no_grad():
            local.gate.fill_(.5)
        self.assertLess(abs(float(.5 * torch.tanh(local.gate.detach()))), .5)
        self.assertFalse(torch.equal(local(x), original(x)))

    def test_versioned_loader_preserves_neutral_function_and_gate_state(self):
        original = NormToy().eval()
        source = {"config": {"latent_geometry": "spatial4-temporal4"}, "model_state_dict": original.state_dict()}
        with patch("finetuning.local_internal_norm.new_model", side_effect=lambda *args: NormToy()):
            model, config = initialize(source, True, "cpu")
            recovered = generic_build(config, model.state_dict(), "cpu")
        self.assertEqual(config["norm_layers"], 2)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, recovered.state_dict()[key]))
        x = torch.randn(1, 1, 3, 17)
        self.assertTrue(torch.equal(original.encoder(x), model.encoder(x)))
