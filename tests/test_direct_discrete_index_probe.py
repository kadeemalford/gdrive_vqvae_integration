import unittest

import numpy as np
import torch

from compression.full_tile_vq_stream import FullTileVQWire
from finetuning.direct_discrete_index_probe import algorithm_signature, physical_score, probe, proposals
from tests.test_joint_anchor_vq import ToyCodec


class DiscreteProposalTests(unittest.TestCase):
    def test_algorithm_signature_rejects_parameter_change_not_comments(self):
        text = "FRACTIONS=(.01,.05,.1,.2)\ndef physical_score(): return 1\ndef proposals(): return 2\ndef probe(): return 3\n"
        self.assertEqual(algorithm_signature(text), algorithm_signature("# comment\n" + text))
        self.assertNotEqual(algorithm_signature(text), algorithm_signature(text.replace(".2)", ".3)")))
        with self.assertRaises(ValueError):
            algorithm_signature("def main(): pass")
    def test_proposals_only_use_distinct_nonincreasing_cost_ids(self):
        current = torch.tensor([0, 1])
        alternatives = torch.tensor([[0, 1, 2], [0, 1, 2]])
        costs = torch.tensor([1., 2., 3.])
        ranking = torch.tensor([[-3., -2., -1.], [-1., -2., -3.]])
        for _, ids in proposals(current, alternatives, costs, ranking, True):
            self.assertTrue((costs[ids] <= costs[current]).all())
            self.assertEqual(ids.tolist(), [0, 0])

    def test_no_negative_descent_keeps_original(self):
        current = torch.tensor([0])
        for _, ids in proposals(current, torch.tensor([[0, 1]]), torch.ones(2), torch.ones(1, 2), True):
            self.assertTrue(torch.equal(current, ids))

    def test_physical_score_handles_exact_constant(self):
        raw = np.zeros((2, 17), np.float32)
        self.assertEqual(physical_score(raw, raw, 1.), 0.)
        self.assertGreater(physical_score(raw, raw + .1, 1.), 0.)

    def test_hard_probe_replays_under_cap_without_changing_model(self):
        torch.manual_seed(118)
        model = ToyCodec().eval().requires_grad_(False)
        receiver_model = ToyCodec().eval().requires_grad_(False)
        receiver_model.load_state_dict(model.state_dict())
        sender = FullTileVQWire(model, np.ones(2048, np.int64), "0" * 64)
        receiver = FullTileVQWire.from_setup(receiver_model, sender.setup_bytes())
        before = {name: value.clone() for name, value in model.state_dict().items()}
        raw = np.random.default_rng(118).normal(size=(3, 17)).astype(np.float32)
        for method in ("gradient", "latent"):
            baseline, final, original, restored, diagnostic = probe(model, receiver, sender, raw, .1, method)
            self.assertLessEqual(len(final), len(baseline))
            self.assertLessEqual(physical_score(raw, restored, .1), physical_score(raw, original, .1))
            self.assertEqual(receiver.decode(final).tobytes(), restored.tobytes())
            self.assertLessEqual(len(diagnostic["trials"]), 8)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
