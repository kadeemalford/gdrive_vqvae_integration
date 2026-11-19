import unittest

from finetuning.summarize_frozen_rate_match import dominators


class MeasuredRateFrontierTests(unittest.TestCase):
    def point(self, error, size, cells=100):
        return {"macro_families": error, "stream_bytes": size, "cells": cells, "groups": {"a": error}}

    def test_only_measured_points_with_nonincreasing_bytes_dominate(self):
        reference = self.point(.5, 100)
        a, b, c = self.point(.4, 101), self.point(.49, 99), self.point(.6, 80)
        self.assertEqual(dominators(reference, [a, b, c]), [b])
        self.assertEqual(dominators(reference, [reference]), [])

    def test_no_rate_interpolation_and_no_population_mismatch(self):
        reference = self.point(.5, 100)
        self.assertEqual(dominators(reference, [self.point(.4, 110), self.point(.6, 90)]), [])
        with self.assertRaises(ValueError):
            dominators(reference, [self.point(.4, 90, 101)])
