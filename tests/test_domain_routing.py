import csv
import tempfile
import unittest
import random
from pathlib import Path

from finetuning.domain_routing import FAMILIES, route_label, family_paths
from finetuning.sampling import epoch_paths


class DomainRoutingTests(unittest.TestCase):
    def test_balanced_control_has_exact_family_dose_without_config_reweighting(self):
        groups = {("doped", "one"): [Path("d0"), Path("d1")],
                  ("multimode", "one"): [Path("m0"), Path("m1"), Path("m2"), Path("m3")]}
        paths = [p for values in groups.values() for p in values]
        result = epoch_paths(paths, groups, "family-window", 8, random.Random(2026))
        self.assertEqual(sum(p.name.startswith("d") for p in result), 4)
        self.assertEqual(sum(p.name.startswith("m") for p in result), 4)
        self.assertEqual(len({p for p in result if p.name.startswith("m")}), 4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            epoch_paths(paths, groups, "family-window", 7, random.Random(2026))

    def test_routes_depend_only_on_metadata_and_unknown_has_fallback(self):
        for family in FAMILIES:
            self.assertEqual(route_label(family), f"expert_{family}")
        self.assertEqual(route_label(None), "shared")
        self.assertEqual(route_label("new_family"), "shared")

    def test_family_selection_preserves_original_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            paths = [root / "a.npz", root / "b.npz", root / "c.npz"]
            with manifest.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_path", "family", "split"])
                writer.writeheader()
                writer.writerows([dict(sample_path=str(paths[0]), family="doped", split="train"),
                                  dict(sample_path=str(paths[1]), family="multimode", split="train"),
                                  dict(sample_path=str(paths[2]), family="doped", split="val")])
            self.assertEqual(family_paths(paths[:2], manifest, "doped", "train"), paths[:1])
            with self.assertRaisesRegex(ValueError, "split boundary"):
                family_paths(paths, manifest, "doped", "train")
            with self.assertRaisesRegex(ValueError, "no samples"):
                family_paths(paths[:2], manifest, "single_mode", "train")
