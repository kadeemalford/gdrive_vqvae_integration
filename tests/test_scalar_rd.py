import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/smartgrids'))
sys.path.insert(0, str(ROOT.parent / 'smartgrids'))
try:
    import constriction
except ImportError:
    constriction = None
if constriction is not None:
    from compression.scalar_stream import ScalarStream


@unittest.skipIf(constriction is None, 'Optional requirements-entropy.txt not installed')
class ScalarStreamTests(unittest.TestCase):
    def test_native_shapes_and_exact_stream_replay(self):
        from finetuning.scalar_rd import ScalarRDCodec
        torch.set_num_threads(1)
        model = ScalarRDCodec(base_channels=8, latent_channels=2).eval()
        codec = ScalarStream.from_scales(model.scales.detach().numpy())
        with torch.no_grad():
            for h in (1, 3, 111, 512):
                x = torch.randn(1, 1, h, 65)
                y, rate, symbols = model(x)
                frame = codec.encode(symbols[0].numpy(), (h, 65), b'side', 'map')
                recovered, shape, _, _ = codec.decode(frame)
                replay = model.synthesis(torch.from_numpy(recovered.copy())[None].float(), shape)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(rate))
                torch.testing.assert_close(y, replay, rtol=0, atol=0)

    def test_exact_roundtrip_and_escapes(self):
        codec = ScalarStream.from_scales([.2, 1.5])
        values = np.arange(120).reshape(2, 3, 20).astype(np.int32) - 40
        values[0, 0, :2] = [-9000, 10000]
        frame = codec.encode(values, (7, 101), b'side', 'map-id')
        restored, shape, side, mapping = codec.decode(frame)
        np.testing.assert_array_equal(restored, values)
        self.assertEqual((shape, side, mapping), ((7, 101), b'side', 'map-id'))
        with self.assertRaises(ValueError):
            codec.decode(frame[:-1] + bytes([frame[-1] ^ 1]))

    def test_noninteger_rejected(self):
        codec = ScalarStream.from_scales([1.])
        with self.assertRaises(ValueError):
            codec.encode(np.array([[[.2]]]), (1, 1), b'', 'id')

    def test_out_of_range_rejected_without_integer_wrap(self):
        codec = ScalarStream.from_scales([1.])
        for value in (2**40, -(2**63)):
            with self.assertRaises(ValueError):
                codec.encode(np.array([[[value]]], dtype=np.int64), (1, 1), b'', 'id')


class ScalarPriorTests(unittest.TestCase):
    def test_rate_has_encoder_gradient(self):
        # Exercise the prior without importing the external backbone.
        from finetuning.scalar_rd import ScalarRDCodec
        model = ScalarRDCodec.__new__(ScalarRDCodec)
        torch.nn.Module.__init__(model)
        model.log_scale = torch.nn.Parameter(torch.zeros(2))
        encoder = torch.nn.Conv2d(1, 2, 1)
        x = torch.ones(1, 1, 3, 7)
        bits = model.rate_bits(encoder(x)).mean()
        bits.backward()
        self.assertTrue(torch.isfinite(bits))
        self.assertGreater(float(encoder.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.log_scale.grad.abs().sum()), 0)

    def test_rate_matches_logistic_mass(self):
        from finetuning.scalar_rd import ScalarRDCodec
        model = ScalarRDCodec.__new__(ScalarRDCodec)
        torch.nn.Module.__init__(model)
        model.log_scale = torch.nn.Parameter(torch.zeros(1))
        values = torch.tensor([[[[-100., -2., 0., 2., 100.]]]])
        bits = model.rate_bits(values)
        self.assertTrue(torch.isfinite(bits).all())
        torch.testing.assert_close(bits[..., :2], bits[..., -2:].flip(-1))
        scale = model.scales.item()
        expected = -np.log2(1 / (1 + np.exp(-.5 / scale)) - 1 / (1 + np.exp(.5 / scale)))
        self.assertAlmostEqual(float(bits[..., 2].detach()), expected, places=5)


if __name__ == '__main__':
    unittest.main()
