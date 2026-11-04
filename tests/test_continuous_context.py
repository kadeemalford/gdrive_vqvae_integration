import unittest

import numpy as np

from finetuning.audit_continuous_context import predict_ar


class ContinuousContextTests(unittest.TestCase):
    def test_fit_never_reads_future_targets(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(2, 800))
        for t in range(1, 800):
            values[:, t] += .9 * values[:, t - 1]
        valid = np.ones(values.shape, dtype=bool)
        a, _ = predict_ar(values, valid, 400, (1, 2), 10)
        changed = values.copy(); changed[:, 400] += 100
        b, _ = predict_ar(changed, valid, 400, (1, 2), 10)
        np.testing.assert_array_equal(a[:, 0], b[:, 0])
        self.assertFalse(np.allclose(a[:, 1], b[:, 1]))
        self.assertLess(np.mean((a - values[:, 400:]) ** 2), np.mean(values[:, 400:] ** 2))

    def test_invalid_past_falls_back_without_discarding_target(self):
        values = np.ones((1, 100))
        valid = np.ones(values.shape, dtype=bool); valid[:, 49] = False
        prediction, count = predict_ar(values, valid, 50, (1,), 5)
        self.assertEqual(prediction.shape, (1, 50))
        self.assertEqual(count, 1)
        self.assertEqual(prediction[0, 0], 1.)


if __name__ == '__main__':
    unittest.main()
