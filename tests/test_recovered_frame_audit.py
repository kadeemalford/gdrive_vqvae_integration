import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from compression.full_tile_vq_stream import FullTileVQWire
from tests.test_joint_anchor_vq import ToyCodec
from tools.audit_recovered_validation_frames import check_symbols, inventory


class RecoveredFrameAuditTests(unittest.TestCase):
    def test_inventory_and_paid_symbols_replay_without_waveform_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "mature_reference"
            (folder / "frames").mkdir(parents=True)
            wire = FullTileVQWire(ToyCodec(), np.ones(2048, np.int64), "0" * 64)
            raw = np.random.default_rng(106).normal(size=(3, 17)).astype(np.float32)
            frame = wire.encode(raw)
            (folder / "registry.zlib").write_bytes(wire.setup_bytes())
            (folder / "frames/sample.frame").write_bytes(frame)
            (folder / "frames_manifest.jsonl").write_text(json.dumps({"frame": "sample.frame", "bytes": len(frame)}) + "\n")
            self.assertEqual(inventory(root)["mature_reference"]["frames"], 1)
            self.assertGreater(check_symbols(folder), 0)
            (folder / "frames/sample.frame").write_bytes(frame[:-1] + bytes([frame[-1] ^ 1]))
            with self.assertRaises(ValueError):
                check_symbols(folder)

    def test_inventory_rejects_partial_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "model"
            (folder / "frames").mkdir(parents=True)
            (folder / "frames/sample.frame").write_bytes(b"x")
            (folder / "frames_manifest.jsonl").write_text(json.dumps({"frame": "sample.frame", "bytes": 2}) + "\n")
            with self.assertRaises(ValueError):
                inventory(root)
