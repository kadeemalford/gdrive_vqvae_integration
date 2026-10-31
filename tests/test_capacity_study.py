import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/smartgrids'))
sys.path.insert(0, '/home/murilo-barroso/source/smartgrids')
from finetuning.model_factory import new_model
from finetuning.run_capacity_study import capacity_config


class CapacityStudyTests(unittest.TestCase):
    def test_capacity_without_larger_bottleneck(self):
        for width, count in ((64, 27958081), (120, 96517033)):
            config = capacity_config(width)
            model = new_model(config, 'meta')
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual(config['num_embeddings'], 2048)
            self.assertEqual(config['embedding_dim'], 128)
            self.assertEqual(config['latent_geometry'], 'spatial-4x')

    def test_narrow_width_replay_and_gradient(self):
        from compression.encode_decode_samples import encode_and_decode
        config = {**capacity_config(8), 'embedding_dim': 8, 'num_embeddings': 16}
        model = new_model(config, 'cpu')
        x = torch.randn(1, 1, 31, 65)
        y, vq, _, _ = model(x)
        (y.square().mean() + vq).backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
        model.eval()
        with torch.no_grad():
            y, _, indices, _ = model(x)
            replay, codes = encode_and_decode(model, x)
        torch.testing.assert_close(y, replay, atol=0, rtol=0)
        torch.testing.assert_close(indices, codes)


if __name__ == '__main__':
    unittest.main()
