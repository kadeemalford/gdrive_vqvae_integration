import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from finetuning.audit_train_provenance import candidate_pairs, full_block_hash, sample_block


class TrainProvenanceTests(unittest.TestCase):
    def test_samples_do_not_prove_complete_block_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "test.h5", "w") as source:
                values = np.random.default_rng(111).normal(size=(1, 1024, 4)).astype("f4")
                a = source.create_dataset("a", data=values)
                values[0, 300, 1] += 1
                b = source.create_dataset("b", data=values)
                self.assertEqual(sample_block(a, 0, [0, 1, 2, 3])["sample_sha256"],
                                 sample_block(b, 0, [0, 1, 2, 3])["sample_sha256"])
                self.assertNotEqual(full_block_hash(a, 0)["sha256"], full_block_hash(b, 0)["sha256"])

    def test_constant_samples_and_same_file_are_not_copy_candidates(self):
        record = {"drive_id": "a", "block": 0, "sample_sha256": "same", "sample_constant_bits": True}
        self.assertEqual(candidate_pairs([record, {**record, "drive_id": "b"}]), [])
        record["sample_constant_bits"] = False
        self.assertEqual(candidate_pairs([record, {**record, "block": 1}]), [])
        self.assertEqual(len(candidate_pairs([record, {**record, "drive_id": "b"}])), 1)
