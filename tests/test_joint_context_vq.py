import unittest

import torch
import torch.nn.functional as F

from finetuning.joint_context_vq import JointContextVQ, context_config, assert_joint_optimizer_steps, inherited_joint_optimizer
from tests.test_analysis_context import TinyModel, TinyQuantizer
from finetuning.run_index_search import vectors
from finetuning.model_factory import architecture


class TrainableQuantizer(TinyQuantizer):
    def forward(self, features):
        quantized, loss, indices, perplexity = super().forward(features)
        exact = vectors(indices, self.embedding.weight, features.shape[-2:])
        return quantized, loss + F.mse_loss(exact, features.detach()), indices, perplexity


class SmallBackbone(TinyModel):
    def __init__(self):
        super().__init__()
        self.quantizer = TrainableQuantizer()
        self.encoder = torch.nn.Conv2d(1, 128, 1)
        with torch.no_grad():
            self.encoder.weight.fill_(.1); self.encoder.bias.fill_(.8)
        self.downsample_levels, self.num_embeddings, self.latent_geometry = 4, 2, 'spatial-4x'


class JointContextTests(unittest.TestCase):
    def test_identity_contexts_and_exact_discrete_decode(self):
        base = SmallBackbone().eval()
        target = torch.randn(1, 1, 2, 3)
        for kind in ('pointwise', 'axial'):
            model = JointContextVQ(base, kind).eval()
            with torch.no_grad():
                ids, shape = model.encode_indices(target)
                reconstructed, _, _, _ = model(target)
                replay = model.decode_indices(ids, shape, target.shape[-2:])
            self.assertTrue(torch.equal(reconstructed, replay))
            self.assertEqual(tuple(replay.shape), tuple(target.shape))
            for context in (model.analysis_context, model.synthesis_context):
                features = torch.randn(1, 128, 2, 3)
                self.assertTrue(torch.equal(context(features, (2, 3)), features))

    def test_both_sides_and_backbone_receive_gradients(self):
        model = JointContextVQ(SmallBackbone(), 'axial').train()
        target = torch.randn(1, 1, 2, 3)
        predicted, vq, _, _ = model(target)
        (predicted.square().mean() + vq).backward()
        for group in (model.base.encoder, model.base.quantizer, model.base.decoder, model.analysis_context, model.synthesis_context):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters() if p.grad is not None), 0)

    def test_versioned_config_rejects_other_geometries_and_control_doses(self):
        config = {'architecture': 'native-v1', 'latent_geometry': 'spatial-4x',
                  'embedding_dim': 128, 'num_embeddings': 2048, 'downsample_levels': 4}
        self.assertEqual(context_config(config, 'axial')['architecture'], 'joint-context-vq-v1')
        with self.assertRaises(ValueError):
            architecture(context_config(config, 'axial'))
        with self.assertRaises(ValueError):
            context_config({**config, 'latent_geometry': 'standard'}, 'axial')
        model = JointContextVQ(SmallBackbone(), 'pointwise')
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaises(ValueError):
            assert_joint_optimizer_steps(model, optimizer, 1)

    def test_optimizer_inherits_base_only_and_starts_context_states_independently(self):
        base = SmallBackbone()
        original = torch.optim.AdamW(base.parameters(), betas=(.8, .95), eps=1e-7)
        for parameter in base.parameters():
            parameter.grad = torch.ones_like(parameter)
        original.step(); original.zero_grad(set_to_none=True)
        for value in original.state.values():
            value['step'].fill_(98304.)
        source_state = original.state_dict()
        model = JointContextVQ(base, 'pointwise')
        optimizer = inherited_joint_optimizer(model, source_state)
        assert_joint_optimizer_steps(model, optimizer, 0)
        self.assertEqual(optimizer.param_groups[1]['betas'], (.8, .95))
        self.assertEqual(optimizer.param_groups[1]['eps'], 1e-7)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert_joint_optimizer_steps(model, optimizer, 1)
        self.assertEqual({float(v['step']) for v in source_state['state'].values()}, {98304.})

    def test_receiver_replay_does_not_call_encoder_or_analysis_context(self):
        model = JointContextVQ(SmallBackbone(), 'axial').eval()
        target = torch.randn(1, 1, 2, 3)
        with torch.no_grad():
            ids, shape = model.encode_indices(target)
            expected = model.decode_indices(ids, shape, target.shape[-2:])

        def forbidden(*args, **kwargs):
            raise AssertionError('Receiver must not access analysis')

        model.base.encoder.forward = forbidden
        model.analysis_context.forward = forbidden
        with torch.no_grad():
            replay = model.decode_indices(ids, shape, target.shape[-2:])
        self.assertTrue(torch.equal(replay, expected))
