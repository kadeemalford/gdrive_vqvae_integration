import unittest

import numpy as np
import torch

from compression.lifting_stream import decode_frame, encode_frame, load_receiver, receiver_bytes
from finetuning.learned_lifting import LiftingCodec


class LearnedLiftingTests(unittest.TestCase):
    def test_identity_even_odd_and_small(self):
        torch.manual_seed(17)
        for learned in (False, True):
            model = LiftingCodec(learned=learned, spatial_levels=3, temporal_levels=2)
            if learned:
                for layer in list(model.predict) + list(model.update):
                    torch.nn.init.normal_(layer.last.weight, std=.01)
            for shape in ((1, 1), (1, 17), (2, 16), (3, 19), (17, 35)):
                x = torch.randn(1, 1, *shape)
                bands = model.analysis(x)
                self.assertEqual(sum(b.numel() for b in bands), x.numel())
                torch.testing.assert_close(model.synthesis(bands, shape), x, atol=2e-6, rtol=2e-6)

    def test_gradients_and_frozen_control(self):
        for learned in (False, True):
            model = LiftingCodec(learned=learned)
            x = torch.randn(1, 1, 19, 51)
            prediction, rate = model(x)
            ((prediction - x).square().mean() + .1 * rate).backward()
            self.assertTrue(torch.isfinite(model.log_step.grad).all())
            self.assertEqual(any(p.grad is not None for p in model.predict.parameters()), learned)

    def test_separate_receiver_and_constants(self):
        model = LiftingCodec().eval()
        receiver = load_receiver(receiver_bytes(model))
        rng = np.random.default_rng(3)
        for shape in ((1, 13), (3, 17), (16, 32)):
            raw = rng.normal(size=shape).astype(np.float32)
            frame = encode_frame(raw, model)
            restored = decode_frame(frame, receiver)
            self.assertEqual(restored.shape, shape)
            self.assertTrue(np.isfinite(restored).all())
        for value in (0., -0., 3.25):
            raw = np.full((3, 17), value, np.float32)
            self.assertEqual(decode_frame(encode_frame(raw, model), receiver).tobytes(), raw.tobytes())

    def test_receiver_mismatch(self):
        a, b = LiftingCodec(), LiftingCodec()
        raw = np.random.default_rng(4).normal(size=(4, 17)).astype(np.float32)
        with self.assertRaises(ValueError):
            decode_frame(encode_frame(raw, a), b)

    def test_fixed_minimal_receiver(self):
        model = LiftingCodec(learned=False).eval()
        receiver = load_receiver(receiver_bytes(model))
        raw = np.random.default_rng(5).normal(size=(3, 19)).astype(np.float32)
        self.assertEqual(decode_frame(encode_frame(raw, model), receiver).shape, raw.shape)

    def test_shared_haar_initial_function(self):
        x = torch.randn(1, 1, 17, 51)
        a, b = LiftingCodec(learned=False), LiftingCodec(learned=True)
        xa, ra = a(x)
        xb, rb = b(x)
        torch.testing.assert_close(xa, xb, atol=0, rtol=0)
        torch.testing.assert_close(ra, rb, atol=0, rtol=0)
