import unittest

import numpy as np
import torch

from compression.adaptive_lifting_stream import decode_frame, encode_frame
from compression.lifting_stream import decode_frame as decode_legacy, encode_frame as encode_legacy
from compression.lifting_stream import load_receiver, receiver_bytes
from finetuning.learned_lifting import LiftingCodec


class AdaptiveLiftingStreamTests(unittest.TestCase):
    def test_identical_default_quantization_and_independent_receiver(self):
        torch.manual_seed(96)
        for learned in (False, True):
            model = LiftingCodec(learned=learned).eval()
            receiver = load_receiver(receiver_bytes(model))
            for shape in ((1, 3), (3, 51), (16, 31)):
                raw = np.random.default_rng(3).normal(size=shape).astype(np.float32)
                steps = model.steps.detach().numpy().tolist()
                actual = decode_frame(encode_frame(raw, model, steps), receiver)
                expected = decode_legacy(encode_legacy(raw, model), receiver)
                self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_profile_odd_constant_and_escape(self):
        model = LiftingCodec().eval()
        for raw in (np.full((3, 17), -0., np.float32),
                    np.full((3, 17), 7., np.float32)):
            self.assertEqual(decode_frame(encode_frame(raw, model, [1.] * 9), model).tobytes(), raw.tobytes())
        raw = np.random.default_rng(1).normal(size=(3, 17)).astype(np.float32)
        raw[0, 0] = 1e30
        result = decode_frame(encode_frame(raw, model, [.25] * 9), model)
        self.assertEqual(result[0, 0].tobytes(), raw[0, 0].tobytes())

    def test_invalid_steps(self):
        model = LiftingCodec().eval()
        raw = np.ones((2, 17), np.float32)
        raw[0, 0] = 0
        for steps in ([1.], [float("nan")] * 9, [-1.] * 9):
            with self.assertRaises(ValueError):
                encode_frame(raw, model, steps)
