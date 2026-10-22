import unittest

import numpy as np

from finetuning.summarize_generalist_frontier import dominates, metric_vector


class GeneralistFrontierTests(unittest.TestCase):
    def test_no_macro_gain_can_hide_a_component_loss(self):
        reference = np.array([1., -.8, 1.])
        self.assertTrue(dominates(np.array([.9, -.9, .9]), reference))
        self.assertFalse(dominates(np.array([.9, -.9, 1.01]), reference))
        self.assertFalse(dominates(reference, reference))

    def test_metrics_have_consistent_direction_and_order(self):
        rows = [{"family": "a", "configuration": "c", "relative_rmse": ".5", "correlation": ".8",
                 "ordinary_channel_p90_relative_rmse": ".9"}]
        np.testing.assert_array_equal(metric_vector(rows), [.5, -.8, .9])
        with self.assertRaises(ValueError):
            dominates(np.ones(2), np.ones(3))
        with self.assertRaises(ValueError):
            dominates(np.array([float("nan")]), np.ones(1))
