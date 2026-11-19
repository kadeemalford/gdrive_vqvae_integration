import unittest

import numpy as np
import torch

from compression.joint_anchor_vq_stream import JointAnchorWire
from finetuning.joint_anchor_vq import JointAnchorVQ
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust
from tests.test_joint_anchor_vq import ToyCodec


class JointAnchorWireTests(unittest.TestCase):
    def test_exact_stream_receiver_and_registry(self):
        for mode in ("independent", "closed-loop"):
            torch.manual_seed(98)
            model = JointAnchorVQ(ToyCodec(), ToyCodec(), mode, 4).eval()
            counts = np.ones((2, 2048), np.int64)
            sender = JointAnchorWire(model, counts, "0" * 64)
            receiver = JointAnchorWire.from_setup(model, sender.setup_bytes())
            for channels in (1, 3, 8):
                raw = np.random.default_rng(channels).normal(size=(channels, 17)).astype(np.float32)
                packet = encode_robust(raw)
                with torch.no_grad():
                    direct, _, _ = model(torch.from_numpy(packet.values)[None, None])
                expected = decode(direct[0, 0].numpy(), packet.side_stream)
                frame = sender.encode(raw)
                self.assertEqual(receiver.decode(frame).tobytes(), expected.tobytes())
                with self.assertRaises(ValueError):
                    receiver.decode(frame[:-1])

    def test_literals_and_escape(self):
        model = JointAnchorVQ(ToyCodec(), ToyCodec(), "closed-loop", 4).eval()
        wire = JointAnchorWire(model, np.ones((2, 2048), np.int64), "0" * 64)
        raw = np.full((3, 17), -0., np.float32)
        self.assertEqual(wire.decode(wire.encode(raw)).tobytes(), raw.tobytes())
        raw = np.random.default_rng(4).normal(size=(3, 17)).astype(np.float32)
        raw[1, 0] = 1e30
        self.assertEqual(wire.decode(wire.encode(raw))[1, 0].tobytes(), raw[1, 0].tobytes())

    def test_book_identity_differs_even_with_identical_counts(self):
        model = JointAnchorVQ(ToyCodec(), ToyCodec(), "closed-loop", 4).eval()
        wire = JointAnchorWire(model, np.ones((2, 2048), np.int64), "0" * 64)
        self.assertNotEqual(wire.streams[0].identifier, wire.streams[1].identifier)
