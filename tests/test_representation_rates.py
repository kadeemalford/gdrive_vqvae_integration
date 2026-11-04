import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetuning.compare_representation_rates import allocate_modes


class RepresentationRateTests(unittest.TestCase):
    def test_actual_budget_and_tail_guard(self):
        compact = [{'sample_path': str(i), 'frame_bytes': 100,
            'metrics': {'family': 'f', 'configuration': 'c', '_aggregation_weight': 100,
                        'relative_rmse': .7, 'correlation': .7, 'ordinary_channel_p90_relative_rmse': 1.}} for i in range(3)]
        dense = [{**r, 'frame_bytes': 150, 'metrics': {**r['metrics'], 'relative_rmse': .4,
                  'correlation': .9, 'ordinary_channel_p90_relative_rmse': 1.1}} for r in compact]
        result = allocate_modes(compact, dense, 420, 20)
        self.assertEqual(result['bytes'], 420)
        self.assertEqual(result['dense_windows'], 2)
        guarded = allocate_modes(compact, dense, 420, 20, guarded=True)
        self.assertEqual(guarded['dense_windows'], 0)
        self.assertIsNone(allocate_modes(compact, dense, 319, 20))
        with self.assertRaises(ValueError):
            allocate_modes(compact, dense[::-1], 500, 20)


if __name__ == '__main__':
    unittest.main()
