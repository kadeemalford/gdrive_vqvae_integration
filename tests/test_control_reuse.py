import copy
import unittest

from finetuning.control_reuse import validate_control_metadata


class ControlReuseTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {'kind': 'raw_phase_pretraining_bridge_v1', 'seed': 2096,
            'source_updates': 16384, 'final_febus_updates': 8192, 'block_size': 2048,
            'learning_rate': 1e-4, 'smoke_only': False,
            'source_febus_plan': ['train'] * 16384, 'final_febus_plan': ['train'] * 8192}
        self.summary = {'source_updates': 16384, 'final_febus_updates': 8192,
                        'final_optimizer_steps': [8192.]}
        self.history = [{'phase': phase, 'updates': step} for phase, budget in
                        [('source', 16384), ('febus', 8192)] for step in range(2048, budget + 1, 2048)]

    def test_complete_matching_control_is_admissible(self):
        validate_control_metadata(self.protocol, self.summary, self.history)

    def test_changed_learning_rate_dose_or_optimizer_is_rejected(self):
        for key, value in [('learning_rate', 5.5e-5), ('seed', 2097), ('final_febus_updates', 4096)]:
            with self.assertRaises(ValueError):
                validate_control_metadata({**self.protocol, key: value}, self.summary, self.history)
        with self.assertRaises(ValueError):
            validate_control_metadata(self.protocol, {**self.summary, 'final_optimizer_steps': [24576.]}, self.history)

    def test_smoke_or_incomplete_history_is_not_scientific_control(self):
        with self.assertRaises(ValueError):
            validate_control_metadata({**self.protocol, 'smoke_only': True}, self.summary, self.history)
        history = copy.deepcopy(self.history); history.pop(2)
        with self.assertRaises(ValueError):
            validate_control_metadata(self.protocol, self.summary, history)
