import unittest
import numpy as np

from compression.symbol_stream import SymbolCodec
from finetuning.run_roi_tail_and_bundle import balanced_modes, fixed_width_frame


def sample(name, before, after, tail_before, tail_after):
    def mode(error, tail, size):
        return {'frame_bytes': size, 'metrics': {'relative_rmse': error,
                'correlation': .5 if size == 100 else .8,
                'ordinary_channel_p90_relative_rmse': tail}}
    return {'sample_id': name, 'input_cells': 1000,
            'compact': mode(before, tail_before, 100), 'dense': mode(after, tail_after, 200)}


class TailBundleTests(unittest.TestCase):
    def test_tail_objective_admits_relative_improvement_without_unbounded_scale(self):
        rows = [sample('ordinary', 1., .9, 1., .9),
                sample('extreme_tail', 1., .9, 1e6, 1e5)]
        choices = balanced_modes(rows, 1.5)
        self.assertEqual(choices['extreme_tail'], 'dense')
        self.assertEqual(choices['ordinary'], 'compact')

    def test_tail_regression_and_excess_budget_are_rejected(self):
        choices = balanced_modes([sample('bad', 1., .5, 1., 2.)], 2.)
        self.assertEqual(choices['bad'], 'compact')
        choices = balanced_modes([sample('good', 1., .5, 1., .5)], 1.)
        self.assertEqual(choices['good'], 'compact')

    def test_fixed_baseline_preserves_indices_and_side(self):
        counts = np.ones(2048, dtype=np.int64);counts[3] = 100000
        codec = SymbolCodec.from_counts(counts, np.tile(counts, (17, 1)), 'a'*64)
        indices = np.full(100, 3)
        original, _ = codec.encode(indices, (1, 1000), (1, 100), b'side')
        fixed = fixed_width_frame(codec, original)
        self.assertGreater(len(fixed), len(original))
        np.testing.assert_array_equal(codec.decode(fixed)[0], indices)
        self.assertEqual(codec.decode(fixed)[3], b'side')


if __name__ == '__main__':
    unittest.main()
