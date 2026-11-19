import hashlib
import unittest

import numpy as np

from compression.bounded_residual_stream import audit, decode as decode_bound, encode as encode_bound
from compression.multicore_magnitude_stream import (
    SHAPES, MagnitudeModel, encode_buckets, features, pack_pair, restore_buckets, unpack_pair,
)
from compression.predictive_coefficient_stream import pack_predictive, unpack_predictive


class MulticoreMagnitudeTests(unittest.TestCase):
    def model(self, mode="aligned"):
        weights = {key: np.zeros(shape, np.float32) for key, shape in SHAPES.items()}
        weights["w0"][1, 0] = 1
        weights["w1"][0, 0] = 1
        weights["w2"][0, 0] = 1
        model = MagnitudeModel(weights, mode, ["a", "b"], np.array([[0., 1., 1.], [0., 1., 1.]]),
                               np.tile(np.arange(1, 8, dtype=np.float32) / 8, (2, 1)))
        return MagnitudeModel.from_setup(model.setup_bytes())

    def test_context_controls_and_shift_are_fixed(self):
        context = np.arange(51, dtype=np.float32).reshape(3, 17)
        aligned = features(context, 0, 0., 1., "aligned")
        shifted = features(context, 0, 0., 1., "shifted")
        expected = features(np.roll(context, 8, axis=1), 0, 0., 1., "aligned")
        np.testing.assert_array_equal(shifted, expected)
        self.assertFalse(np.array_equal(aligned, shifted))
        self.assertTrue(np.all(features(context, 0, 0., 1., "none")[:, :6] == 0))

    def test_joint_roundtrip_charges_both_cores_and_preserves_physical_bits(self):
        rng = np.random.default_rng(99)
        raw_a = rng.normal(size=(3, 17)).astype(np.float32)
        raw_b = rng.normal(size=(3, 17)).astype(np.float32)
        zero = np.zeros_like(raw_a)
        a, b = encode_bound(raw_a, zero, .03), encode_bound(raw_b, zero, .03)
        metadata = {"shape": list(raw_a.shape), "group_id": "a", "left_id": "left", "right_id": "right"}
        baseline = pack_pair(a, b, metadata)
        model = self.model()
        frame = pack_pair(a, b, metadata, model)
        left, right, _, _ = unpack_pair(frame, self.model())
        self.assertEqual(left.tobytes(), decode_bound(a, zero).tobytes())
        self.assertEqual(right.tobytes(), decode_bound(b, zero).tobytes())
        audit(raw_a, left, .03)
        audit(raw_b, right, .03)
        self.assertGreater(len(baseline), len(a) + len(b))
        self.assertLessEqual(len(frame), len(baseline))
        with self.assertRaises(ValueError):
            unpack_pair(frame[:-1], model)

    def test_nonfallback_bucket_reassembly_and_wrong_model_rejection(self):
        context = np.arange(51, dtype=np.float32).reshape(3, 17) / 20
        model = self.model()
        q = np.arange(51, dtype=np.int32).reshape(context.shape)
        bucket = model.buckets(context, "a")
        present = sorted(np.unique(bucket).tolist())
        metadata = {"kind": "neural-magnitude-buckets-v1", "original": {"kind": "test"},
                    "group_id": "a", "shape": list(q.shape), "present": present,
                    "model_sha256": model.identifier, "q_sha256": hashlib.sha256(q.astype("<i4").tobytes()).hexdigest()}
        frame = pack_predictive([q[bucket == i] for i in present], metadata, b"")
        restored, _, _ = unpack_predictive(restore_buckets(frame, context, "a", model))
        np.testing.assert_array_equal(restored[0], q)
        with self.assertRaises(ValueError):
            restore_buckets(frame, context, "b", model)
