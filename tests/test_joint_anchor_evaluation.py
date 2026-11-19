import unittest
from unittest.mock import patch

import torch

from finetuning.joint_anchor_evaluation import evaluate
from finetuning.joint_anchor_vq import JointAnchorVQ
from tests.test_joint_anchor_vq import ToyCodec


class JointAnchorEvaluationTests(unittest.TestCase):
    def test_receiver_construction_does_not_consume_training_rng(self):
        model = JointAnchorVQ(ToyCodec(), ToyCodec(), "closed-loop", 4)
        state = torch.get_rng_state().clone()
        with patch("finetuning.joint_anchor_evaluation._evaluate", side_effect=lambda *a, **k: torch.rand(5)):
            result = evaluate(model)
        self.assertEqual(result.shape, (5,))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
