import unittest
import numpy as np
from finetuning.channel_geometry import selected_indices, select_spatial, spatial_metadata, map_identifier


class ChannelGeometryTests(unittest.TestCase):
    def test_selection_matches_legacy_and_preserves_gap(self):
        raw = np.arange(24, dtype=np.float32).reshape(6, 4)
        raw[1, 0] = np.nan
        raw[4, 0] = np.inf
        selected, mapping = select_spatial(raw, 3, {"Spacing": [.2, .05], "Origin": [0, 0], "Extent": [100, 105]})
        legacy = raw[np.isfinite(raw).all(axis=1)]
        start = (len(legacy) - 3) // 2
        np.testing.assert_array_equal(selected, legacy[start:start + 3])
        self.assertEqual(mapping["source_channel_indices"], [0, 2, 3])
        self.assertFalse(mapping["selected_rows_contiguous"])
        self.assertEqual(mapping["missing_source_rows_inside_selection"], 1)
        np.testing.assert_allclose(mapping["metadata_positions_m"], [20, 20.4, 20.6])

    def test_unknown_coordinates_not_invented(self):
        mapping = spatial_metadata([1, 2], 4, {"Spacing": [.2], "Extent": [100, 999]})
        self.assertIsNone(mapping["metadata_positions_m"])
        self.assertIsNone(mapping["source_extent_channel_indices"])
        self.assertEqual(mapping["dataset_relative_positions_m"], [.2, .4])

    def test_crop_preserves_original_index_offset(self):
        np.testing.assert_array_equal(selected_indices(np.ones(1012, dtype=bool), 512), np.arange(250, 762))

    def test_empty_finite_selection_safe(self):
        selected, mapping = select_spatial(np.full((4, 10), np.nan), 3, {})
        self.assertEqual(selected.shape, (0, 10))
        self.assertEqual(mapping["source_channel_indices"], [])

    def test_hash_independent_of_key_order(self):
        self.assertEqual(map_identifier({"a": 1, "b": 2}), map_identifier({"b": 2, "a": 1}))

    def test_invalid_indices_rejected(self):
        for indices in ([2, 1], [-1, 2], [1, 1], [0, 4]):
            with self.assertRaises(ValueError):
                spatial_metadata(indices, 4, {})
