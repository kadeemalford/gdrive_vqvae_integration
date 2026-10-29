"""Architecture C: exact rate geometry, coded-only decoding and checkpoint replay."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, "/home/murilo-barroso/source/smartgrids")
from finetuning.model_factory import new_model, build_model
from finetuning.latent_geometry import POLICIES, apply_geometry, latent_positions
from compression.encode_decode_samples import encode_and_decode


class AnisotropicVQVAETests(unittest.TestCase):
    def config(self, geometry="standard"):
        return dict(base_channels=8, embedding_dim=8, num_embeddings=16,
                    downsample_levels=4, commitment_cost=.25,
                    architecture="anisotropic-multiscale-v1", latent_geometry=geometry, exact_eval_quantization=True)

    def test_shapes_and_exact_index_counts_across_geometries(self):
        model = new_model(self.config(), "cpu")
        for geometry in POLICIES:
            apply_geometry(model, geometry)
            for height, width in [(1, 33), (31, 65), (48, 80)]:
                with self.subTest(geometry=geometry, shape=(height, width)), torch.no_grad():
                    original = torch.randn(1,1,height,width)
                    result, loss, indices, _ = model(original)
                    self.assertEqual(result.shape, original.shape)
                    self.assertEqual(indices.numel(), latent_positions(model, height, width))
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertTrue(torch.isfinite(loss))

    def test_inference_replays_only_transmitted_codes(self):
        model = new_model(self.config("spatial4-temporal4"), "cpu").eval()
        x = torch.randn(1,1,31,65)
        with torch.no_grad():
            expected, _, indices, _ = model(x)
            actual, transmitted = encode_and_decode(model, x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(indices, transmitted)

    def test_checkpoint_replay_and_gradients(self):
        config = self.config("spatial-4x")
        model = new_model(config, "cpu")
        x = torch.randn(1,1,31,65)
        y, vq, _, _ = model(x)
        ((x-y).square().mean()+vq).backward()
        for module in (model.encoder, model.quantizer, model.decoder):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters()))
        saved = {"config": config, "model_state_dict": model.state_dict()}
        restored = build_model(config, saved, "cpu")
        with torch.no_grad():
            torch.testing.assert_close(model(x)[0], restored(x)[0])

    def test_native_exact_eval_replays_indices_with_activation_checkpointing(self):
        config = {**self.config("spatial4-temporal4"),"architecture":"native-v1","activation_checkpointing":True}
        model = new_model(config,"cpu")
        x = torch.randn(1,1,31,65)
        y,loss,_,_ = model(x)
        (y.square().mean()+loss).backward()
        model.eval()
        with torch.no_grad():
            y,_,indices,_ = model(x)
            replay,codes = encode_and_decode(model,x)
        torch.testing.assert_close(y,replay,atol=0,rtol=0)
        torch.testing.assert_close(indices,codes)


if __name__ == "__main__":
    unittest.main()
