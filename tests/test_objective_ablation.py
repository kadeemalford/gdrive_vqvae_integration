import copy
import unittest

import torch

from finetuning.objective_ablation import reconstruction_loss, validate_lr_control_metadata


class ObjectiveAblationTests(unittest.TestCase):
    def test_objectives_exclude_escaped_cells_and_keep_gradients(self):
        predicted = torch.tensor([1., 3., 1e20], requires_grad=True)
        target = torch.zeros(3)
        valid = torch.tensor([True, True, False])
        self.assertEqual(float(reconstruction_loss(predicted, target, valid, 'mse').detach()), 5.)
        self.assertEqual(float(reconstruction_loss(predicted, target, valid, 'mixed').detach()), 3.5)
        reconstruction_loss(predicted, target, valid, 'mse').backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([1., 3., 0.]))

    def test_unknown_objectives_or_empty_masks_fail(self):
        x = torch.ones(3)
        for kind, mask in [('wrong', torch.ones(3, dtype=torch.bool)), ('mse', torch.zeros(3, dtype=torch.bool))]:
            with self.assertRaises(ValueError):
                reconstruction_loss(x, x, mask, kind)

    def test_control_requires_correct_dose_lr_and_independent_optimizer(self):
        protocol = {'kind': 'paired_update_lr_continuation_v1', 'smoke_only': False,
            'updates_per_arm': 16384, 'blocks': 8, 'block_size': 2048, 'lr_high': 1e-4, 'lr_low': 1e-5,
            'sequence': [['train'] * 2048 for _ in range(8)]}
        summary = {'arm': 'constant_high', 'updates': 16384, 'source_optimizer_preserved': True,
            'initial_optimizer_steps': [98304.] * 117, 'final_optimizer_steps': [114688.],
            'schedule_state': {'kind': 'constant_high', 'completed_updates': 16384}}
        validate_lr_control_metadata(protocol, summary)
        for key, value in [('lr_high', 5.5e-5), ('smoke_only', True), ('updates_per_arm', 8192)]:
            with self.assertRaises(ValueError):
                validate_lr_control_metadata({**protocol, key: value}, summary)
        wrong = copy.deepcopy(summary); wrong['final_optimizer_steps'] = [131072.]
        with self.assertRaises(ValueError):
            validate_lr_control_metadata(protocol, wrong)
