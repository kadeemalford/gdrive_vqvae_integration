import unittest

import numpy as np

from finetuning.full_tile_bottleneck import Moments, pair_moments, pair_result


class FullTileBottleneckTests(unittest.TestCase):
    def test_streaming_covariance_matches_numpy(self):
        values = np.random.default_rng(103).normal(size=(100, 4))
        moments = Moments(4)
        moments.update(values[:37])
        moments.update(values[37:])
        result = moments.result()
        expected = np.linalg.eigvalsh(np.cov(values, rowvar=False, bias=True))[::-1]
        np.testing.assert_allclose(result["eigenvalues"], expected, atol=1e-12)

    def test_masked_pairs_exclude_escapes_and_do_not_wrap(self):
        values = np.array([[1., 2., 10000., 4., 5.], [5., 4., 10000., 2., 1.]])
        valid = np.ones_like(values, dtype=bool)
        valid[:, 2] = False
        result = pair_result(pair_moments(values, valid, 1))
        self.assertEqual(result["pairs"], 4)
        expected = np.corrcoef([1., 4., 5., 2.], [2., 5., 4., 1.])[0, 1]
        self.assertAlmostEqual(result["correlation"], expected)

    def test_no_pairs_or_no_variance_is_undefined(self):
        self.assertIsNone(pair_result(np.zeros(6))["correlation"])
        values = np.ones((1, 5))
        self.assertIsNone(pair_result(pair_moments(values, values.astype(bool), 1))["correlation"])
        self.assertEqual(pair_result(pair_moments(values, values.astype(bool), 0))["pairs"], 0)
