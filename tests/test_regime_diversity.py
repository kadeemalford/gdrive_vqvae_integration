import unittest

import numpy as np

from finetuning.regime_diversity import crop, nested_groups


class RegimeDiversityTests(unittest.TestCase):
    def rows(self):
        return [{"group_id": f"{family}/{index}"} for family, count in
                (("doped", 1), ("multicore", 2), ("multimode", 1), ("single_mode", 6))
                for index in range(count)]

    def test_nested_subsets_preserve_families_and_ignore_input_order(self):
        rows = self.rows()
        subsets = nested_groups(rows)
        self.assertEqual(subsets, nested_groups(list(reversed(rows))))
        for size, groups in subsets.items():
            self.assertEqual(len(groups), size)
            self.assertEqual(len({g.split("/")[0] for g in groups}), 4)
        self.assertTrue(set(subsets[4]) < set(subsets[6]) < set(subsets[10]))

    def test_crop_is_contiguous_equal_dose_and_does_not_modify_input(self):
        raw = np.arange(16000, dtype=np.float32).reshape(16, 1000)
        before = raw.tobytes()
        result, offset, start = crop(raw, np.random.default_rng(2117))
        self.assertEqual(result.shape, (8, 1000))
        np.testing.assert_array_equal(result, raw[offset:offset + 8, start:start + 1000])
        result[:] = 0
        self.assertEqual(raw.tobytes(), before)
        with self.assertRaises(ValueError):
            crop(raw[:7], np.random.default_rng(2117))
