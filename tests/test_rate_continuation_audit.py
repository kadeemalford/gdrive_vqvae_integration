import unittest

from finetuning.audit_rate_continuation import INVARIANTS, METRICS, describe, file_bootstrap, pair_rows


class RateContinuationAuditTests(unittest.TestCase):
    def row(self, path="/data/a.npz", metric=1):
        return {"sample_path": path, **{k: "same" for k in INVARIANTS},
                **{k: str(metric) for k in METRICS}}

    def test_pairing_uses_paths_not_order(self):
        before = [self.row(), self.row("/data/b.npz", 2)]
        after = [self.row("/data/b.npz", 1), self.row(metric=2)]
        result = pair_rows(before, after)
        self.assertEqual([x["delta_relative_rmse"] for x in result], [1, -1])
        self.assertEqual(describe(result)["window_rmse_regression_fraction"], .5)

    def test_refuses_invalid_pairs(self):
        for key in INVARIANTS:
            after = self.row()
            after[key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                pair_rows([self.row()], [after])
        for after in ([], [self.row(), self.row()], [self.row("/different.npz")], [self.row(metric="nan")]):
            with self.assertRaises(ValueError):
                pair_rows([self.row()], after)

    def test_bootstrap_clusters_files_and_is_deterministic(self):
        rows = [{"drive_id": "a", "delta_relative_rmse": -1}] * 20
        self.assertFalse(file_bootstrap(rows, 100, 1)["bootstrap_available"])
        rows += [{"drive_id": "b", "delta_relative_rmse": 1}]
        result = file_bootstrap(rows, 100, 1)
        self.assertEqual(result, file_bootstrap(rows, 100, 1))
        self.assertEqual(result["file_bootstrap_delta_rmse_low"], -1)
        self.assertEqual(result["file_bootstrap_delta_rmse_high"], 1)
