import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from finetuning.evaluate_lifting_curve import select_receiver, summarize


class LiftingCurveTests(unittest.TestCase):
    def test_selection_uses_registered_real_stream_objective(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "protocol.json").write_text(json.dumps({"target_proxy_bpc": .8}))
            for update, error, bpc in ((0, .5, 3.), (2048, .6, .8)):
                (root / f"check_{update:06d}.json").write_text(json.dumps(
                    {"update": update, "macro_families": error, "stream_bpc": bpc}))
            path, update = select_receiver(root)
            self.assertEqual(update, 2048)
            self.assertEqual(path.name, "receiver_002048.pt")

    def test_physical_channel_ids_not_tile_offsets(self):
        rows = [{"group_id": "family/group", "drive_id": "file", "channel_map_id": key}
                for key in ("left", "right")]
        maps = {"left": {"source_channel_indices": [10]},
                "right": {"source_channel_indices": [11]}}
        targets = [(np.array([[0., 2.]]), np.array([[0., 2.]])),
                   (np.array([[100., 102.]]), np.array([[100., 102.]]))]
        result = summarize(rows, targets, [b"a", b"bc"], {"family/group": .1}, maps, 5)
        self.assertEqual(result["records"][0]["nrmse_star_macro_channels"], 0)
        self.assertEqual(len(result["records"][0]["channels"]), 2)
        self.assertEqual(result["stream_bytes"], 8)
        self.assertEqual(result["stream_bpc"], 16)
