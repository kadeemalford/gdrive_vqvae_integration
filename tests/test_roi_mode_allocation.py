import unittest

from finetuning.run_frozen_entropy_roi import select_modes


def record(name, compact_rmse=1., dense_rmse=.5, dense_p90=.5, size=200):
    def mode(size, error, p90):
        return {'frame_bytes': size, 'metrics': {'relative_rmse': error,
                'correlation': 1-error/2, 'ordinary_channel_p90_relative_rmse': p90}}
    return {'sample_id': name, 'input_cells': 1000,
            'compact': mode(100, compact_rmse, 1.), 'dense': mode(size, dense_rmse, dense_p90)}


class RoiAllocationTests(unittest.TestCase):
    def test_budget_and_guarded_tail(self):
        rows = [record('good'), record('bad_tail', dense_p90=2.)]
        choices, accounting = select_modes(rows, 2., 'guarded')
        self.assertEqual(choices, {'good': 'dense', 'bad_tail': 'compact'})
        self.assertLessEqual(accounting['frame_bytes'], accounting['budget_bytes'])

    def test_no_roi_when_no_gain_or_budget(self):
        rows = [record('no_gain', dense_rmse=2.), record('no_budget')]
        choices, _ = select_modes(rows, 1., 'guarded')
        self.assertEqual(set(choices.values()), {'compact'})

    def test_uniform_independent_of_distortion(self):
        rows = [record(str(i)) for i in range(10)]
        a, _ = select_modes(rows, 1.5, 'uniform')
        for row in rows:
            row['dense']['metrics']['relative_rmse'] = 100.
        b, _ = select_modes(rows, 1.5, 'uniform')
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
