import unittest
import torch

from finetuning.fsq_quantizer import FiniteScalarQuantizer
from finetuning.run_residual_refinement import new_refiner, load_refiner
from compression.encode_decode_samples import encode_and_decode


class FsqTests(unittest.TestCase):
    def test_all_2048_states_decodable(self):
        quantizer=FiniteScalarQuantizer(8)
        decoded=quantizer.embedding(torch.arange(2048))
        self.assertEqual(tuple(decoded.shape),(2048,8))
        self.assertEqual(torch.unique(decoded,dim=0).shape[0],2048)

    def test_eval_exact_replay_and_training_gradients(self):
        quantizer=FiniteScalarQuantizer(8)
        z=torch.randn(1,8,5,7,requires_grad=True)
        y,loss,indices,_=quantizer(z)
        (y.square().mean()+loss).backward()
        self.assertGreater(z.grad.abs().sum(),0)
        quantizer.eval()
        with torch.no_grad():
            y,_,indices,_=quantizer(z)
            decoded=quantizer.embedding(indices).view(1,5,7,8).permute(0,3,1,2).contiguous()
        torch.testing.assert_close(y,decoded,atol=0,rtol=0)
        self.assertTrue(((indices>=0)&(indices<2048)).all())

    def test_residual_factory_and_checkpoint_roundtrip(self):
        config=dict(architecture='native-v1',base_channels=8,embedding_dim=8,
                    num_embeddings=2048,downsample_levels=4,commitment_cost=.25,
                    latent_geometry='standard',activation_checkpointing=True,
                    exact_eval_quantization=True,quantizer='fsq-8x8x32')
        model=new_refiner(config,'cpu').eval()
        tensor=torch.randn(1,1,31,65)
        with torch.no_grad():
            predicted,_,indices,_=model(tensor)
            decoded,codes=encode_and_decode(model,tensor)
        torch.testing.assert_close(predicted,decoded,atol=0,rtol=0)
        torch.testing.assert_close(indices,codes)
        restored=load_refiner({'config':config,'model_state_dict':model.state_dict()},'cpu')
        with torch.no_grad():torch.testing.assert_close(model(tensor)[0],restored(tensor)[0],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
