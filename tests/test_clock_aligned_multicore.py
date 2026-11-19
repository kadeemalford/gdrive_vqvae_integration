import unittest

from finetuning.clock_aligned_multicore import coefficient, nominal_pairs


class MulticorePairingTests(unittest.TestCase):
    def make(self, file_id, core, ticks):
        return {"drive_id": file_id, "group_id": "multicore/g", "dataset": f"zone/data Ch{core}",
                "clock_status": "read", "clock_native": ticks}

    def headers(self, files):
        attrs = {"Spacing": [.2, .05, 1], "Origin": [0, 0, 0], "Extent": [0, 3], "GaugeLength": [1.]}
        return {row["drive_id"]: {"dataset": row["dataset"], "attributes": {"/zone": attrs}} for row in files}

    def test_pairing_is_by_exact_clock_and_pairs_are_split_as_units(self):
        files = [self.make("a", 1, [1., 2.]), self.make("b", 2, [2., 3.]),
                 self.make("c", 1, [10.]), self.make("d", 2, [10.])]
        pairs = nominal_pairs(files, self.headers(files))
        self.assertEqual(len(pairs), 2)
        self.assertEqual([row["role"] for row in pairs], ["fit", "check"])

    def test_ambiguous_components_fail_instead_of_leaking_pair_sides(self):
        files = [self.make("a", 1, [1.]), self.make("b", 2, [1.]), self.make("c", 2, [1.])]
        with self.assertRaises(ValueError):
            nominal_pairs(files, self.headers(files))

    def test_fit_is_bounded_and_degenerate_context_is_explicit(self):
        self.assertEqual(coefficient([1., 10.]), 2.)
        self.assertEqual(coefficient([0., 10.]), 0.)
