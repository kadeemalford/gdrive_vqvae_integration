import hashlib
import unittest

import numpy as np

from compression.bounded_residual_stream import audit, decode as physical_decode, encode as physical_encode
from compression.neural_residual_stream import IntegerPredictor, decode, encode, features
from compression.predictive_coefficient_stream import pack_predictive, unpack_predictive


class NeuralResidualTests(unittest.TestCase):
    def predictor(self):
        rng = np.random.default_rng(103)
        shapes = {"w0": (8, 32), "b0": (32,), "w1": (32, 32), "b1": (32,),
                  "w2": (32, 1), "b2": (1,)}
        weights = {key: (rng.normal(size=shape) * .05).astype(np.float32) for key, shape in shapes.items()}
        return IntegerPredictor(weights, True)

    def test_features_cannot_access_current_or_future_coefficients(self):
        base = np.ones((3, 17), np.float32)
        q = np.arange(51, dtype=np.int32).reshape(3, 17)
        original = features(base, q, 1., True)
        q[1:] += 123
        np.testing.assert_array_equal(original[:2], features(base, q, 1., True)[:2])
        control = features(base, q, 1., False)
        self.assertTrue(np.all(control[..., 4:7] == 0))

    def test_replay_coefficients_and_physical_bits(self):
        predictor = self.predictor()
        setup = predictor.setup_bytes()
        receiver = IntegerPredictor.from_setup(setup)
        identifier = hashlib.sha256(setup).hexdigest()
        raw = np.random.default_rng(103).normal(size=(8, 64)).astype(np.float32)
        base = raw * .75
        original = physical_encode(raw, base, .03)
        frame = encode(original, base, predictor, identifier, 3.)
        arrays, descriptor, side = decode(frame, base, receiver, identifier)
        original_arrays, original_descriptor, original_side = unpack_predictive(original)
        np.testing.assert_array_equal(arrays[0], original_arrays[0])
        self.assertEqual(descriptor, original_descriptor)
        self.assertEqual(side, original_side)
        restored = physical_decode(pack_predictive(arrays, descriptor, side), base)
        self.assertEqual(restored.tobytes(), physical_decode(original, base).tobytes())
        audit(raw, restored, .03)
        self.assertLessEqual(len(frame), len(original))

    def test_nonfallback_causal_replay(self):
        predictor = self.predictor()
        identifier = "1" * 64
        q = np.arange(60, dtype=np.int32).reshape(3, 20)
        base = np.zeros_like(q, np.float32)
        inputs = features(base, q, 1., True)
        delta = q - np.stack([predictor.predict(row, 1.) for row in inputs])
        metadata = {"kind": "neural-causal-integer-v1", "original": {"kind": "test"},
                    "scale": 1., "model_sha256": identifier,
                    "q_sha256": hashlib.sha256(q.astype("<i4").tobytes()).hexdigest()}
        frame = pack_predictive([delta.astype(np.int32)], metadata, b"")
        arrays, _, _ = decode(frame, base, predictor, identifier)
        np.testing.assert_array_equal(arrays[0], q)
        with self.assertRaises(ValueError):
            decode(frame, base, predictor, "2" * 64)
