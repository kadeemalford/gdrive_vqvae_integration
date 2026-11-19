import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from finetuning.audit_physical_metadata import header_only, native


class HeaderOnlyAuditTests(unittest.TestCase):
    def test_attributes_are_read_without_dataset_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.h5"
            with h5py.File(path, "w") as source:
                dataset = source.create_dataset("zone/strain", shape=(3, 10), dtype="f4")
                dataset.parent.attrs["GaugeLength"] = np.array([5.])
                dataset.parent.attrs["Spacing"] = np.array([.2, 1.])
            with patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("Dataset values forbidden")):
                result = header_only(path, "zone/strain")
            self.assertEqual(result["shape"], [3, 10])
            self.assertEqual(result["physical_fields"]["gauge"][0]["value"], [5.])

    def test_metadata_conversion_keeps_strings_and_missing_values_explicit(self):
        self.assertEqual(native(np.array([b"unit"])), ["unit"])
        self.assertEqual(native(float("nan")), "nan")
