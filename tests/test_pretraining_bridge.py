import unittest

import torch

from finetuning.run_pretraining_bridge import balanced_source_plan, optimizer_steps


class PretrainingBridgeTests(unittest.TestCase):
    def test_external_plan_has_equal_group_dose_and_is_reproducible(self):
        rows = [{'family': family, 'source_id': f'{family}_{source}', 'sample_name': f'{family}_{source}_{i}'}
                for family in ('a', 'b', 'c', 'd') for source in range(2) for i in range(3)]
        plan = balanced_source_plan(rows, 100, 2096)
        self.assertEqual(plan, balanced_source_plan(rows, 100, 2096))
        for family in ('a', 'b', 'c', 'd'):
            self.assertEqual(sum(name.startswith(family + '_') for name in plan), 25)
        self.assertTrue(set(plan) <= {r['sample_name'] for r in rows})
        with self.assertRaises(ValueError):
            balanced_source_plan(rows, 99, 2096)

    def test_fresh_stage_optimizer_restarts_step_counter(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        source = torch.optim.AdamW([p], lr=1e-4)
        for _ in range(2):
            source.zero_grad(); p.square().sum().backward(); source.step()
        self.assertEqual(optimizer_steps(source), [2.])
        target = torch.optim.AdamW([p], lr=1e-4)
        self.assertEqual(optimizer_steps(target), [])
        target.zero_grad(); p.square().sum().backward(); target.step()
        self.assertEqual(optimizer_steps(target), [1.])
        self.assertEqual(optimizer_steps(source), [2.])
