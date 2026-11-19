import unittest

from tools.choose_feasibility_training_gpu import available, snapshot


class FeasibilityGpuChoiceTests(unittest.TestCase):
    def setUp(self):
        self.inventory = (
            "0, GPU-a, NVIDIA RTX 4090, 24564, 24000, 0\n"
            "1, GPU-b, NVIDIA RTX 4090, 24564, 22000, 1\n"
        )

    def test_busy_device_is_skipped_even_if_more_memory_available(self):
        devices = snapshot(self.inventory, "GPU-a, 1234\n")
        self.assertEqual(available(devices, 8000)["uuid"], "GPU-b")
        self.assertIsNone(available(devices, 23000))

    def test_choose_largest_free_device_when_both_are_available(self):
        self.assertEqual(available(snapshot(self.inventory, ""), 8000)["index"], 0)

    def test_unknown_inventory_is_not_treated_as_idle(self):
        with self.assertRaises(ValueError):
            snapshot(self.inventory, "Not Supported")
