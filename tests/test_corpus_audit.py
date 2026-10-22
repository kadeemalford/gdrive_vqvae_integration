import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np

from finetuning.audit_febus_corpus import content_descriptors, interval_union_length, scan_dataset, source_geometry, window_key


class CorpusAuditTests(unittest.TestCase):
    def test_streaming_hash_chunk_invariant_and_invalid_channels_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.h5"
            data = np.arange(128, dtype=np.float32).reshape(2, 16, 4)
            data[0, 4, 1] = np.nan
            data[1, 8, 2] = np.float32(1e30)
            with h5py.File(path, "w") as handle:
                dataset = handle.create_dataset("Strain Rate", data=data)
                first = scan_dataset(dataset, 5)
                second = scan_dataset(dataset, 7)
                self.assertEqual(first[1], second[1])
                self.assertFalse(first[0][0, 1])
                self.assertTrue(first[0][1].all())
                self.assertEqual(first[2], 1)
                self.assertEqual(first[3], 0)

    def test_overlap_not_counted_twice(self):
        self.assertAlmostEqual(interval_union_length([(0, 1.05), (1, 2.05), (2, 3.05)]), 3.05)
        self.assertEqual(interval_union_length([(0, 1000), (1052, 2052)]), 2000)

    def test_geometry_matches_preparation(self):
        self.assertEqual(source_geometry((5, 21000, 61)), (5, 21000, 61, True))
        self.assertEqual(source_geometry((1000, 40)), (1, 1000, 40, True))
        self.assertEqual(source_geometry((40, 1000)), (1, 1000, 40, False))

    def test_coordinate_key_ignores_fold_copy_path_and_role(self):
        row = dict(drive_id="id", block_index="1", time_start="500", time_samples="1000", channels="40")
        self.assertEqual(window_key({**row, "split": "train", "sample_path": "a"}), window_key({**row, "split": "test", "sample_path": "b"}))

    def test_broadband_has_higher_entropy_than_tone(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(size=(64, 1000)).astype(np.float32)
        tone = np.tile(np.sin(2 * np.pi * .1 * np.arange(1000)), (64, 1)).astype(np.float32)
        a, b = content_descriptors(noise), content_descriptors(tone)
        self.assertGreater(a["temporal_spectral_entropy"], b["temporal_spectral_entropy"])
        self.assertGreater(a["temporal_spectral_flatness"], b["temporal_spectral_flatness"])
        self.assertAlmostEqual(b["in_phase_common_mode_fraction"], 1)

    def test_channel_energy_concentration_is_detected(self):
        rng = np.random.default_rng(2)
        signal = rng.normal(size=(40, 1000))
        signal[0] *= 100
        result = content_descriptors(signal)
        self.assertGreater(result["top_10pct_channels_energy_fraction"], .99)
        self.assertLess(result["effective_channels_fraction"], .04)


if __name__ == "__main__":
    unittest.main()
