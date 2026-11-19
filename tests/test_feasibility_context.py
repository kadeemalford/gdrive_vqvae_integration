import unittest

from finetuning.feasibility_context import plan_context_windows


class FeasibilityContextTests(unittest.TestCase):
    def test_core_and_stride_are_preserved_across_all_native_widths(self):
        for channels in range(1, 1025):
            plan = plan_context_windows(channels)
            self.assertEqual(plan["windows"][-1]["channels"], channels)
            previous = None
            for window in plan["windows"]:
                self.assertEqual(window["start"] % 16, 0)
                self.assertGreaterEqual(window["start"], 0)
                self.assertLessEqual(window["stop"], channels)
                self.assertEqual(window["start"] + window["score_start_in_window"], plan["core_start"])
                self.assertEqual(window["start"] + window["score_stop_in_window"], plan["core_stop"])
                self.assertGreaterEqual(window["score_start_in_window"], 0)
                self.assertLessEqual(window["score_stop_in_window"], window["channels"])
                if previous:
                    self.assertLessEqual(window["start"], previous["start"])
                    self.assertGreaterEqual(window["stop"], previous["stop"])
                previous = window

    def test_narrow_multimode_tiles_are_retained_without_fabricated_context(self):
        plan = plan_context_windows(2)
        self.assertFalse(plan["has_context_contrast"])
        self.assertEqual(len(plan["windows"]), 1)
        self.assertEqual((plan["core_start"], plan["core_stop"]), (0, 2))

    def test_invalid_width(self):
        for width in (0, -1, True, 3.5):
            with self.assertRaises(ValueError):
                plan_context_windows(width)
