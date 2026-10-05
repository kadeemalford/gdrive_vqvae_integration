import csv
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from finetuning.sampling import epoch_paths, sample_groups


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.groups = {("a", "one"): [Path("a1")], ("a", "two"): [Path("a2")],
                       ("b", "one"): [Path(f"b{i}") for i in range(100)]}
        self.paths = [p for ps in self.groups.values() for p in ps]

    def test_window_covers_each_once(self):
        drawn = epoch_paths(self.paths, {}, "window", 0, random.Random(1))
        self.assertCountEqual(drawn, self.paths)

    def test_family_group_balances_families_and_groups(self):
        drawn = epoch_paths(self.paths, self.groups, "family-group", 20000, random.Random(1))
        counts = Counter("a" if str(p).startswith("a") else "b" for p in drawn)
        self.assertTrue(0.48 < counts["a"] / len(drawn) < 0.52)
        self.assertTrue(0.23 < drawn.count(Path("a1")) / len(drawn) < 0.27)
        self.assertTrue(set(drawn) <= set(self.paths))

    def test_group_is_not_family_uniform(self):
        drawn = epoch_paths(self.paths, self.groups, "group", 20000, random.Random(1))
        self.assertTrue(0.31 < drawn.count(Path("a1")) / len(drawn) < 0.36)

    def test_determinism(self):
        a = epoch_paths(self.paths, self.groups, "family-group", 100, random.Random(7))
        b = epoch_paths(self.paths, self.groups, "family-group", 100, random.Random(7))
        self.assertEqual(a, b)

    def test_manifest_rejects_test_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_path", "family", "configuration", "split"])
                writer.writeheader()
                writer.writerow(dict(sample_path="a1", family="a", configuration="one", split="test"))
            with self.assertRaises(ValueError):
                sample_groups([Path("a1")], manifest)


if __name__ == "__main__":
    unittest.main()
