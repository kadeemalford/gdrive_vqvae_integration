import unittest

import torch

from finetuning.channel_objective import bounded_channel_mse, CHANNEL_WEIGHT_CAP


class ChannelObjectiveTests(unittest.TestCase):
    def test_equal_variance_channels_reduce_to_mse(self):
        target = torch.tensor([[[[-1., 1.], [-1., 1.]]]])
        prediction = target + 1
        self.assertEqual(bounded_channel_mse(prediction, target, torch.ones_like(target, dtype=torch.bool)).item(), 1.)

    def test_escaped_cells_have_no_value_or_gradient_effect(self):
        target = torch.tensor([[[[-1., 1., float("nan")]]]])
        prediction = torch.tensor([[[[0., 2., float("nan")]]]], requires_grad=True)
        mask = torch.tensor([[[[True, True, False]]]])
        loss = bounded_channel_mse(prediction, target, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(prediction.grad[0, 0, 0, -1].item(), 0.)

    def test_quiet_channel_weights_are_bounded_and_zero_variance_finite(self):
        target = torch.tensor([[[[-1., 1.], [-1e-8, 1e-8]]]])
        prediction = target + 1
        loss = bounded_channel_mse(prediction, target, torch.ones_like(target, dtype=torch.bool))
        self.assertLessEqual(loss.item(), CHANNEL_WEIGHT_CAP)
        zero = torch.zeros_like(target)
        self.assertEqual(bounded_channel_mse(zero + 1, zero, torch.ones_like(zero, dtype=torch.bool)).item(), 1.)

    def test_fully_escaped_channels_are_excluded(self):
        target = torch.tensor([[[[-1., 1.], [0., 0.]]]])
        mask = torch.tensor([[[[True, True], [False, False]]]])
        self.assertEqual(bounded_channel_mse(target + 1, target, mask).item(), 1.)
        with self.assertRaises(ValueError):
            bounded_channel_mse(target, target, torch.zeros_like(mask))
