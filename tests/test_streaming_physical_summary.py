import unittest

import numpy as np

from finetuning.evaluate_lifting_curve import summarize
from finetuning.streaming_physical_summary import StreamingPhysicalSummary


class StreamingSummaryTests(unittest.TestCase):
    def test_streaming_matches_existing_window_pooling(self):
        rows = [{"group_id": "family/group", "drive_id": "file", "channel_map_id": "map",
                 "dt_s": .00005}] * 3
        maps = {"map": {"source_channel_indices": [10, 11]}}
        scales = {"family/group": .01}
        pairs = [(np.arange(34.).reshape(2, 17) + i, np.arange(34.).reshape(2, 17) + i + .1)
                 for i in range(3)]
        expected = summarize(rows, pairs, [b"abc"] * 3, scales, maps, 5)
        streaming = StreamingPhysicalSummary(scales, maps)
        for row, (raw, predicted) in zip(rows, pairs):
            streaming.update(row, raw, predicted, 3)
        result = streaming.result(5)
        for key in ("macro_families", "cells", "stream_bytes", "groups", "families"):
            self.assertEqual(expected[key], result[key])
        self.assertEqual(expected["records"][0]["channels"], result["records"][0]["channels"])
        self.assertEqual(result["records"][0]["source_channel_indices"], [10, 11])

    def test_empty_fails_instead_of_silent_zero_score(self):
        with self.assertRaises(ValueError):
            StreamingPhysicalSummary({}, {}).result(0)
