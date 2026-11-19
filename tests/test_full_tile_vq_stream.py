import unittest

import numpy as np
import torch

from compression.full_tile_vq_stream import FullTileVQWire
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust
from tests.test_joint_anchor_vq import ToyCodec


class FullTileWireTests(unittest.TestCase):
    def test_exact_symbols_and_standalone_registry(self):
        model = ToyCodec().eval()
        sender = FullTileVQWire(model, np.ones(2048, np.int64), "0" * 64)
        receiver = FullTileVQWire.from_setup(model, sender.setup_bytes())
        raw = np.random.default_rng(101).normal(size=(3, 17)).astype(np.float32)
        packet = encode_robust(raw)
        with torch.no_grad():
            predicted, _, _, _, _ = exact_branch_forward(model, torch.from_numpy(packet.values)[None, None])
        expected = decode(predicted[0, 0].numpy(), packet.side_stream)
        self.assertEqual(receiver.decode(sender.encode(raw)).tobytes(), expected.tobytes())
        constant = np.full((3, 17), -0., np.float32)
        self.assertEqual(receiver.decode(sender.encode(constant)).tobytes(), constant.tobytes())
