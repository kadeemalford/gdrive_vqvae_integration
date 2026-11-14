import copy
import unittest

import numpy as np
import torch

from finetuning.update_schedule import UpdateSchedule, load_independent_optimizer_state
from finetuning.evaluate_lr_schedule_study import group_regressions


class UpdateScheduleTests(unittest.TestCase):
    def test_endpoints_mean_and_monotonicity(self):
        schedule = UpdateSchedule('cosine', 16384)
        rates = [schedule.rate(i) for i in range(16384)]
        self.assertAlmostEqual(rates[0], 1e-4)
        self.assertAlmostEqual(rates[-1], 1e-5)
        self.assertAlmostEqual(np.mean(rates), 5.5e-5)
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))

    def test_constant_controls_and_budget(self):
        for kind, expected in [('constant_high', 1e-4), ('constant_mean', 5.5e-5)]:
            s = UpdateSchedule(kind, 2)
            self.assertAlmostEqual(s.rate(0), expected)
            self.assertAlmostEqual(s.rate(1), expected)
            s.advance(); s.advance()
            with self.assertRaises(ValueError):
                s.advance()
            with self.assertRaises(ValueError):
                s.rate(2)

    def test_optimizer_and_schedule_resume_are_bit_exact(self):
        def initialize():
            p = torch.nn.Parameter(torch.tensor([.25, -.5], dtype=torch.float64))
            return p, torch.optim.AdamW([p], lr=1e-4, weight_decay=1e-5), UpdateSchedule('cosine', 10)

        def step(p, optimizer, schedule):
            lr = schedule.prepare_update(optimizer)
            optimizer.zero_grad()
            (p.square().sum() + .1 * p.sum()).backward()
            optimizer.step()
            schedule.advance()
            return lr

        p, opt, s = initialize()
        for _ in range(4):
            step(p, opt, s)
        checkpoint = copy.deepcopy((p.detach(), opt.state_dict(), s.state_dict()))
        expected_rates = [step(p, opt, s) for _ in range(6)]
        restored, opt2, s2 = initialize()
        restored.data.copy_(checkpoint[0])
        opt2.load_state_dict(checkpoint[1]); s2.load_state_dict(checkpoint[2])
        self.assertEqual([step(restored, opt2, s2) for _ in range(6)], expected_rates)
        self.assertTrue(torch.equal(p, restored))
        self.assertEqual(s.state_dict(), s2.state_dict())

    def test_resume_rejects_policy_changes(self):
        source = UpdateSchedule('cosine', 10).state_dict()
        for kind, budget in [('constant_high', 10), ('cosine', 20)]:
            with self.assertRaises(ValueError):
                UpdateSchedule(kind, budget).load_state_dict(source)
        source['completed_updates'] = 11
        with self.assertRaises(ValueError):
            UpdateSchedule('cosine', 10).load_state_dict(source)

    def test_configuration_guards_preserve_rms_distance(self):
        reference = {'family': 'single_mode', 'configuration': 'test', 'relative_rmse': .65,
                     'correlation': .75, 'ordinary_channel_p90_relative_rmse': 1.,
                     'fft_magnitude_relative_error': .5, 'fft_phase_alignment': .8,
                     'reconstruction_rms_ratio': .8}
        self.assertEqual(group_regressions([reference], [reference.copy()]), [])
        candidate = {**reference, 'relative_rmse': .6, 'reconstruction_rms_ratio': 1.3}
        failures = group_regressions([reference], [candidate])
        self.assertEqual([r['metric'] for r in failures], ['reconstruction_rms_ratio'])
        with self.assertRaises(ValueError):
            group_regressions([reference], [{**reference, 'configuration': 'different'}])

    def test_optimizer_forks_do_not_mutate_source_cpu_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        parameter.square().sum().backward(); optimizer.step()
        source = copy.deepcopy(optimizer.state_dict())
        initial_step = source['state'][0]['step'].clone()
        initial_moment = source['state'][0]['exp_avg'].clone()
        for _ in range(3):
            candidate = torch.nn.Parameter(torch.tensor([1.]))
            fork = torch.optim.AdamW([candidate], lr=1e-4)
            load_independent_optimizer_state(fork, source)
            self.assertEqual(float(fork.state[candidate]['step']), 1)
            candidate.square().sum().backward(); fork.step()
            self.assertEqual(float(fork.state[candidate]['step']), 2)
            self.assertTrue(torch.equal(source['state'][0]['step'], initial_step))
            self.assertTrue(torch.equal(source['state'][0]['exp_avg'], initial_moment))
