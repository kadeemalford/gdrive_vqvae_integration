import unittest

from finetuning.fit_lifting_allocation import coordinate_search, file_weights


class LiftingAllocationTests(unittest.TestCase):
    def test_search_preserves_best_and_is_deterministic(self):
        def evaluate(vector):
            return sum((v - 1) ** 2 for v in vector) ** .5, sum(vector)
        a = coordinate_search(evaluate, 3, 4, .01)
        b = coordinate_search(evaluate, 3, 4, .01)
        self.assertEqual(a, b)
        self.assertEqual(a[0], [1, 1, 1])
        self.assertEqual(a[1], 0)

    def test_weighting_balances_groups_and_families(self):
        rows = [{"family": "a", "group_id": "a/1"}] * 3 + [
            {"family": "a", "group_id": "a/2"}, {"family": "b", "group_id": "b/1"}]
        weights = file_weights(rows)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertAlmostEqual(sum(weights[:4]), .5)
        self.assertAlmostEqual(sum(weights[:3]), .25)
