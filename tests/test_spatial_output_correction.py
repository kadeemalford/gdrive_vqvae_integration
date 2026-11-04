import unittest

import torch

from compression.spatial_correction_stream import pack_frame, unpack_frame, HEADER, decode_model_frame
from finetuning.spatial_output_correction import SpatialOutputCorrection
from finetuning.run_spatial_output_study import split_train_files, channel_audit
from finetuning.run_spatial_boundary_coding import pad_spatial_replication, ReverseSpatial
from finetuning.vq_encoder_rd import frozen_forward
from finetuning.robust_escape import encode_robust
from finetuning.run_physical_guard_search import physical_admit
import numpy as np


class SpatialOutputCorrectionTests(unittest.TestCase):
    def test_physical_guards_are_strictly_nested_and_handle_missing_values(self):
        a = {'relative_rmse': .7, 'correlation': .7, 'ordinary_channel_p90_relative_rmse': 1.,
             'fft_magnitude_relative_error': .5, 'fft_phase_alignment': .8, 'reconstruction_rms_ratio': .75}
        b = {**a, 'relative_rmse': .6, 'correlation': .8}
        self.assertTrue(physical_admit(a, b))
        for key, value in [('fft_magnitude_relative_error', .51), ('fft_phase_alignment', .79),
                           ('reconstruction_rms_ratio', .74), ('correlation', None)]:
            self.assertFalse(physical_admit(a, {**b, key: value}))

    def test_full_vq_updates_all_three_components(self):
        torch.manual_seed(9)
        model = torch.nn.Module()
        model.encoder = torch.nn.Conv2d(1, 4, 1)
        model.decoder = torch.nn.Conv2d(4, 1, 1)
        model.quantizer = torch.nn.Module()
        model.quantizer.embedding = torch.nn.Embedding(8, 4)
        model.downsample_levels = 0
        model.train()
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        target = torch.randn(1, 1, 4, 7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        prediction, vq, rate, _, _, _ = frozen_forward(model, target, torch.arange(1., 9.), .5, True)
        ((prediction - target).square().mean() + vq + .1 * rate).backward()
        optimizer.step()
        for prefix in ('encoder.', 'decoder.', 'quantizer.embedding.'):
            self.assertTrue(any(not torch.equal(initial[k], v) for k, v in model.state_dict().items() if k.startswith(prefix)))

    def test_full_model_mode_selects_embedding_and_decoder_before_synthesis(self):
        packet = encode_robust(np.array([[1., 2., 3.]], dtype=np.float32))
        class DummyStream:
            def decode(self, payload):
                return np.zeros(3, dtype=np.int64), (1, 3), (1, 3), packet.side_stream
        def model(value):
            result = torch.nn.Module()
            result.quantizer = torch.nn.Module()
            result.quantizer.embedding = torch.nn.Embedding(1, 1)
            with torch.no_grad():
                result.quantizer.embedding.weight.fill_(value)
            result.decoder = torch.nn.Identity()
            return result
        base, candidate = model(0.), model(1.)
        a = decode_model_frame(pack_frame(b'codes', False), DummyStream(), base, candidate, 'cpu')
        b = decode_model_frame(pack_frame(b'codes', True), DummyStream(), base, candidate, 'cpu')
        self.assertTrue(np.all(b > a))

    def test_spatial_padding_preserves_data_and_temporal_zero_policy(self):
        x = torch.arange(3. * 5.).view(1, 1, 3, 5)
        padded = pad_spatial_replication(x)
        self.assertEqual(tuple(padded.shape), (1, 1, 16, 16))
        torch.testing.assert_close(padded[:, :, :3, :5], x, rtol=0, atol=0)
        torch.testing.assert_close(padded[:, :, 3:16, :5], x[:, :, -1:, :].expand(1, 1, 13, 5))
        self.assertEqual(float(padded[:, :, :, 5:].abs().sum()), 0.)

    def test_spatial_reversal_is_involutive(self):
        x = torch.randn(1, 1, 17, 21)
        transform = ReverseSpatial()
        torch.testing.assert_close(transform(transform(x)), x, rtol=0, atol=0)

    def test_internal_split_is_file_disjoint_and_deterministic(self):
        rows = [{'split': 'train', 'family': family, 'configuration': 'c', 'drive_id': family + str(i // 2)}
                for family in ('a', 'b') for i in range(12)]
        fit, check = split_train_files(rows)
        self.assertEqual((fit, check), split_train_files(rows))
        self.assertFalse({rows[i]['drive_id'] for i in fit} & {rows[i]['drive_id'] for i in check})
        self.assertEqual({rows[i]['family'] for i in check}, {'a', 'b'})

    def test_weak_channel_audit_preserves_all_channels(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(32, 100))
        x[:2] *= 1e-6
        r = channel_audit(x, x * .8, np.ones_like(x, dtype=bool))
        self.assertEqual(r['weak_channels'], 2)
        self.assertEqual(sum(r[key]['channels'] for key in ('edge_weak', 'edge_nonweak', 'interior_weak', 'interior_nonweak')), 32)

    def test_identity_and_equal_parameter_budget(self):
        a, b = SpatialOutputCorrection(True), SpatialOutputCorrection(False)
        x = torch.randn(1, 1, 3, 21)
        torch.testing.assert_close(a(x), x, rtol=0, atol=0)
        torch.testing.assert_close(b(x), x, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))

    def test_only_spatial_arm_sees_neighbors(self):
        x = torch.arange(4.).view(1, 1, 4, 1)
        a = SpatialOutputCorrection(True).features(x)
        b = SpatialOutputCorrection(False).features(x)
        self.assertEqual(a[0, 0, 2, 0], 0.)
        self.assertEqual(b[0, 0, 2, 0], 2.)

    def test_gradient_and_single_channel(self):
        model = SpatialOutputCorrection()
        x = torch.randn(1, 1, 1, 20)
        model(x).square().mean().backward()
        self.assertGreater(float(model.net[-1].weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(model.features(x)).all())

    def test_envelope_identity_equal_mode_cost_and_corruption(self):
        payload = b'original coded data'
        a, b = pack_frame(payload, False), pack_frame(payload, True)
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), HEADER.size + len(payload))
        self.assertEqual(unpack_frame(a), (payload, False))
        self.assertEqual(unpack_frame(b), (payload, True))
        for bad in (a[:-1], a[:4] + b'\x01' + a[5:], b'', a + b'0'):
            with self.assertRaises(ValueError):
                unpack_frame(bad)
