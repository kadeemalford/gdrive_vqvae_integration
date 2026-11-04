import sys
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetuning.run_index_search import FrozenIndexStream, search
from finetuning.summarize_index_search import admit_search


class IndexSearchTests(unittest.TestCase):
    def test_guard_rejects_tail_or_correlation_regression(self):
        before = {'relative_rmse': .7, 'correlation': .7, 'ordinary_channel_p90_relative_rmse': 1.}
        after = {'relative_rmse': .6, 'correlation': .8, 'ordinary_channel_p90_relative_rmse': .9}
        self.assertTrue(admit_search(before, after))
        self.assertFalse(admit_search(before, {**after, 'ordinary_channel_p90_relative_rmse': 1.01}))
        self.assertFalse(admit_search(before, {**after, 'correlation': .69}))
        self.assertFalse(admit_search(before, {**after, 'relative_rmse': .71}))

    @unittest.skipIf(importlib.util.find_spec('constriction') is None,
                     'Optional requirements-entropy.txt not installed')
    def test_frozen_search_respects_objective_and_real_bytes(self):
        torch.set_num_threads(1)
        torch.manual_seed(2078)
        model = torch.nn.Module()
        model.encoder = torch.nn.Conv2d(1, 4, 1)
        model.decoder = torch.nn.Conv2d(4, 1, 1)
        model.quantizer = torch.nn.Module()
        model.quantizer.embedding = torch.nn.Embedding(8, 4)
        model.eval().requires_grad_(False)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        counts = np.ones(2048, dtype=np.int64)
        stream = FrozenIndexStream(counts, '0' * 64)
        target = torch.randn(1, 1, 4, 7)
        valid = torch.ones_like(target, dtype=torch.bool)
        a, b, details = search(model, target, valid, 8, .05, stream, b'side')
        objective = lambda x: .5 * np.mean((x - target[0, 0].numpy()) ** 2) + .5 * np.mean(np.abs(x - target[0, 0].numpy()))
        self.assertLessEqual(objective(b), objective(a) + 1e-7)
        self.assertLessEqual(details['searched_frame_bytes'], details['baseline_frame_bytes'])
        self.assertTrue(details['exact_discrete_replay'])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], atol=0, rtol=0)
        a, b, guarded = search(model, target, valid, 8, .05, stream, b'side',
                               candidate_guard=lambda original, proposed: False)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(guarded['changed_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
