from types import SimpleNamespace
import unittest

import torch

from finetuning.codebook_refresh import CodebookRefresh


class CodebookRefreshTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(7)
        embedding = torch.nn.Embedding(8, 4)
        optimizer = torch.optim.AdamW(embedding.parameters(), lr=1e-3)
        embedding.weight.square().sum().backward()
        optimizer.step()
        model = SimpleNamespace(num_embeddings=8, embedding_dim=4, quantizer=SimpleNamespace(embedding=embedding))
        return CodebookRefresh(model, optimizer, 2026, age=3, cap=2)

    def observe(self, refresh, epoch):
        refresh.begin_epoch(epoch)
        refresh.observe(torch.ones(1, 4, 2, 2) * 20, torch.zeros(4, dtype=torch.long))

    def test_only_inactive_codes_are_changed_and_moments_cleared(self):
        refresh = self.make()
        original = refresh.model.quantizer.embedding.weight.detach().clone()
        for epoch in (1, 2):
            self.observe(refresh, epoch)
            self.assertEqual(refresh.refresh(), 0)
        self.observe(refresh, 3)
        self.assertEqual(refresh.refresh(), 2)
        parameter = refresh.model.quantizer.embedding.weight
        changed = (parameter.detach() != original).any(dim=1)
        self.assertEqual(int(changed.sum()), 2)
        self.assertFalse(bool(changed[0]))
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(float(refresh.optimizer.state[parameter][key][changed].abs().sum()), 0.)
            self.assertGreater(float(refresh.optimizer.state[parameter][key][0].abs().sum()), 0.)

    def test_state_roundtrip_preserves_refresh_decision(self):
        original = self.make()
        for epoch in (1, 2):
            self.observe(original, epoch)
            original.refresh()
        resumed = self.make()
        resumed.load_state_dict(original.state_dict())
        for item in (original, resumed):
            self.observe(item, 3)
            item.refresh()
        torch.testing.assert_close(original.model.quantizer.embedding.weight, resumed.model.quantizer.embedding.weight, rtol=0, atol=0)

    def test_policy_mismatch_rejected(self):
        refresh = self.make()
        state = refresh.state_dict()
        state["cap"] = 9
        with self.assertRaises(ValueError):
            refresh.load_state_dict(state)

    def test_reservoir_is_bounded(self):
        refresh = self.make()
        refresh.begin_epoch(1)
        for _ in range(65):
            refresh.step = 0
            refresh.observe(torch.ones(1, 4, 2, 2), torch.zeros(4, dtype=torch.long))
        self.assertEqual(len(refresh.reservoir), 64)
