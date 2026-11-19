import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from compression.joint_anchor_vq_stream import JointAnchorWire
from finetuning.encoder_observation_vq import ObservationJointVQ, VERSION, build
from finetuning.joint_anchor_vq import JointAnchorVQ
from tests.test_joint_anchor_vq import ToyCodec


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 1, 1))

    def forward(self, tensor):
        return self.net(tensor)


class NativeStyleToy(ToyCodec):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderNet()


class EncoderObservationTests(unittest.TestCase):
    def test_zero_adapter_same_function_and_no_persisted_observations(self):
        torch.manual_seed(99)
        original = JointAnchorVQ(NativeStyleToy(), NativeStyleToy(), "closed-loop", 4).eval()
        for enabled in (False, True):
            model = ObservationJointVQ(copy.deepcopy(original.anchor), copy.deepcopy(original.innovation), enabled, 4).eval()
            model.predictor.load_state_dict(original.predictor.state_dict())
            for channels in (1, 3, 8):
                x = torch.randn(1, 1, channels, 17)
                expected, _, before = original(x)
                actual, _, after = model(x)
                self.assertTrue(torch.equal(expected.view(torch.int32), actual.view(torch.int32)))
                self.assertTrue(torch.equal(before["anchor_ids"], after["anchor_ids"]))
                self.assertIsNone(model.anchor.encoder.net[0].current)
                self.assertIsNone(model.innovation.encoder.net[0].current)
                self.assertFalse(any("current" in key for key in model.state_dict()))

    def test_gradient_isolation_and_receiver_uses_only_indices(self):
        for enabled in (False, True):
            model = ObservationJointVQ(NativeStyleToy(), NativeStyleToy(), enabled, 4).train()
            x = torch.randn(1, 1, 7, 17)
            y, loss, _ = model(x)
            ((y - x).square().mean() + loss).backward()
            total = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if ".extra." in n)
            self.assertEqual(total > 0, enabled)
            sender = JointAnchorWire(model.eval(), np.ones((2, 2048), np.int64), "0" * 64)
            receiver_model = copy.deepcopy(model).eval()
            receiver = JointAnchorWire.from_setup(receiver_model, sender.setup_bytes())
            raw = np.random.default_rng(99).normal(size=(7, 17)).astype(np.float32)
            decoded = receiver.decode(sender.encode(raw))
            self.assertEqual(decoded.shape, raw.shape)
            self.assertIsNone(receiver_model.anchor.encoder.net[0].current)

    def test_versioned_loader(self):
        model = ObservationJointVQ(NativeStyleToy(), NativeStyleToy(), True, 4).eval()
        config = {"architecture": VERSION, "base_config": {}, "mode": "closed-loop",
                  "predictor_width": 4, "encoder_context": True}
        with patch("finetuning.encoder_observation_vq.new_model", side_effect=lambda *args: NativeStyleToy()):
            receiver = build(config, model.state_dict(), "cpu")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, receiver.state_dict()[key]))
