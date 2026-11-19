import unittest

import numpy as np
import torch
from torch import nn

from compression.anchor_innovation_stream import AnchorInnovationCodec, predict
from finetuning.learned_lifting import LiftingCodec
from finetuning.run_index_search import FrozenIndexStream


class FakeVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.quantizer = nn.Module()
        self.quantizer.embedding = nn.Embedding(2048, 1)
        with torch.no_grad():
            self.quantizer.embedding.weight[:, 0] = torch.linspace(-10, 10, 2048)


class AnchorInnovationTests(unittest.TestCase):
    def codec(self):
        model, receiver = FakeVQ(), FakeVQ()
        receiver.load_state_dict(model.state_dict())
        stream = FrozenIndexStream(np.ones(2048, np.int64), "0" * 64)
        return AnchorInnovationCodec(model, receiver, stream, LiftingCodec(learned=False), "0" * 64)

    def test_closed_loop_odd_single_and_escape(self):
        codec = self.codec()
        rng = np.random.default_rng(96)
        for channels in (1, 2, 3, 7):
            raw = rng.normal(size=(channels, 31)).astype(np.float32)
            if channels > 1:
                raw[1, 0] = 1e30
            frame = codec.encode(raw, [.5, .5], .01)
            restored = codec.decode(frame)
            self.assertEqual(restored.shape, raw.shape)
            self.assertTrue(np.isfinite(restored).all())
            mask = np.abs(raw) < 1e20
            self.assertLess(float(np.abs(raw[mask] - restored[mask]).max()), .1)
            if channels > 1:
                self.assertEqual(restored[1, 0].tobytes(), raw[1, 0].tobytes())
            with self.assertRaises(ValueError):
                codec.decode(frame[:-1])

    def test_literal_and_predictor_edges(self):
        codec = self.codec()
        raw = np.full((3, 17), -0., np.float32)
        self.assertEqual(codec.decode(codec.encode(raw, [.5, .5], 1.)).tobytes(), raw.tobytes())
        anchors = np.array([[1., 2.], [3., 4.]])
        np.testing.assert_array_equal(predict(anchors, 2, [.5, .5]), [[2, 3], [3, 4]])
        with self.assertRaises(ValueError):
            predict(anchors, 1, [float("nan"), .5])
