import unittest

import torch
import torch.nn.functional as F

from finetuning.run_residual_refinement import admissible, allocate_refinements


def record(name, extra, error=.5, tail=.5):
    return {'sample_id':name,'base_bytes':100,'refinement_bytes':extra,'cells':1000,
            'admitted':True,'base':{'relative_rmse':1.,'ordinary_channel_p90_relative_rmse':1.},
            'refined':{'relative_rmse':error,'ordinary_channel_p90_relative_rmse':tail}}


class ResidualRefinementTests(unittest.TestCase):
    def test_nested_allocation_keeps_earlier_refinements(self):
        rows=[record('expensive',90),record('cheap',30),record('other',50)]
        first=allocate_refinements(rows,1.25)
        second=allocate_refinements(rows,1.5,first)
        self.assertTrue(first <= second)
        self.assertLessEqual(sum(r['refinement_bytes'] for r in rows if r['sample_id'] in second),150)

    def test_admission_rejects_any_tail_or_correlation_regression(self):
        base={'relative_rmse':1.,'correlation':.5,'ordinary_channel_p90_relative_rmse':1.}
        improved={'relative_rmse':.5,'correlation':.8,'ordinary_channel_p90_relative_rmse':.5}
        self.assertTrue(admissible(base,improved))
        self.assertFalse(admissible(base,{**improved,'ordinary_channel_p90_relative_rmse':2.}))
        self.assertFalse(admissible(base,{**improved,'correlation':.4}))

    def test_unadmitted_refinement_never_transmitted(self):
        row=record('rejected',10);row['admitted']=False
        self.assertEqual(allocate_refinements([row],2.),set())

    def test_residual_objective_equals_composed_reconstruction_error(self):
        target=torch.randn(2,3)
        base=torch.randn(2,3)
        predicted=torch.randn(2,3)
        mask=torch.tensor([[True,False,True],[True,True,False]])
        residual=(target-base).masked_fill(~mask,0.)
        a=.5*F.mse_loss(predicted[mask],residual[mask])+.5*F.l1_loss(predicted[mask],residual[mask])
        b=.5*F.mse_loss((base+predicted)[mask],target[mask])+.5*F.l1_loss((base+predicted)[mask],target[mask])
        torch.testing.assert_close(a,b)

    def test_only_refinement_model_receives_gradients(self):
        from finetuning.model_factory import new_model
        config=dict(architecture='native-v1',base_channels=8,embedding_dim=8,
                    num_embeddings=16,downsample_levels=4,commitment_cost=.25,
                    exact_eval_quantization=True,activation_checkpointing=True)
        base=new_model({**config,'latent_geometry':'spatial-4x'},'cpu').eval()
        for p in base.parameters():p.requires_grad_(False)
        refiner=new_model({**config,'latent_geometry':'standard'},'cpu').train()
        target=torch.randn(1,1,31,65)
        with torch.no_grad():residual=target-base(target)[0]
        predicted,vq,_,_=refiner(residual)
        (F.mse_loss(predicted,residual)+vq).backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in refiner.parameters()))


if __name__=='__main__':unittest.main()
