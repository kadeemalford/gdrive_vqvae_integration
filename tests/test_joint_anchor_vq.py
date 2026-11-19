import unittest

import torch
from torch import nn
import torch.nn.functional as F

from finetuning.joint_anchor_vq import AnchorPredictor, JointAnchorVQ, exact_branch_forward


class ToyQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 1)
        with torch.no_grad():
            self.embedding.weight[:, 0] = torch.linspace(-4, 4, 16)

    def forward(self, z):
        flat = z.permute(0, 2, 3, 1).reshape(-1, 1)
        distances = (flat - self.embedding.weight.T).square()
        indices = distances.argmin(1)
        exact = self.embedding(indices).view(z.shape[0], z.shape[2], z.shape[3], 1).permute(0, 3, 1, 2)
        loss = F.mse_loss(exact, z.detach()) + .25 * F.mse_loss(exact.detach(), z)
        return z + (exact - z).detach(), loss, indices, torch.tensor(8.)


class ToyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 1, 1)
        self.decoder = nn.Conv2d(1, 1, 1)
        self.quantizer = ToyQuantizer()


class JointAnchorVQTests(unittest.TestCase):
    def test_exact_hard_forward_and_gradient(self):
        model = ToyCodec().train()
        x = torch.randn(1, 1, 3, 17, requires_grad=True)
        output, loss, ids, latent, _ = exact_branch_forward(model, x)
        exact = model.quantizer.embedding(ids).view(1, *latent, 1).permute(0, 3, 1, 2)
        expected = model.decoder(exact)[..., :3, :17]
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        (output.square().mean() + loss).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_joint_replay_parity_and_predictor_gradient(self):
        torch.manual_seed(97)
        for mode in ("independent", "closed-loop"):
            model = JointAnchorVQ(ToyCodec(), ToyCodec(), mode, predictor_width=4)
            for channels in (1, 2, 3, 7):
                model.train()
                x = torch.randn(1, 1, channels, 17)
                prediction, loss, info = model(x)
                ((prediction - x).square().mean() + loss).backward()
                model.eval()
                direct, _, info = model(x)
                replay = model.decode_indices(info["anchor_ids"], info["anchor_latent"],
                                              info["innovation_ids"], info["innovation_latent"], x.shape[-2:])
                torch.testing.assert_close(direct, replay, atol=0, rtol=0)
            if mode == "closed-loop":
                gradient = model.predictor.net[-1].weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
            else:
                self.assertTrue(all(p.grad is None for p in model.predictor.parameters()))

    def test_predictor_only_sees_anchors(self):
        model = AnchorPredictor(4)
        anchors = torch.randn(1, 1, 3, 17)
        expected = .5 * (anchors[:, :, :2] + anchors[:, :, 1:3])
        torch.testing.assert_close(model(anchors, 2), expected, atol=0, rtol=0)

    def test_branch_weights_and_optimizer_states_do_not_alias(self):
        model = JointAnchorVQ(ToyCodec(), ToyCodec(), "closed-loop", 4).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(1, 1, 3, 17)
        y, loss, _ = model(x)
        ((y - x).square().mean() + loss).backward()
        optimizer.step()
        a, b = model.anchor.quantizer.embedding.weight, model.innovation.quantizer.embedding.weight
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertNotEqual(optimizer.state[a]["step"].data_ptr(), optimizer.state[b]["step"].data_ptr())
        self.assertNotEqual(optimizer.state[a]["exp_avg"].data_ptr(), optimizer.state[b]["exp_avg"].data_ptr())
