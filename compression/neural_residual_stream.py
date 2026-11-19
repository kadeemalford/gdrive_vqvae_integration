"""Lossless causal neural prediction of bounded-correction integer coefficients."""
import hashlib
import io

import numpy as np

from compression.predictive_coefficient_stream import pack_predictive, unpack_predictive


def feature_row(current_base, previous_base, previous_q, denominator, scale, available, contextual):
    def shifts(values):
        return (np.r_[0., values[:-1]], values, np.r_[values[1:], 0.])
    current = np.asarray(current_base, np.float64) / denominator
    previous = np.asarray(previous_base, np.float64) / denominator
    q = np.asarray(previous_q, np.float64) / scale if contextual else np.zeros_like(current)
    features = np.stack((*shifts(current), previous, *shifts(q),
                         np.full_like(current, float(available))), axis=1)
    return np.clip(features, -32, 32).astype(np.float32)


def features(base, q, scale, contextual):
    finite = np.where(np.isfinite(base), base, 0).astype(np.float64)
    denominator = max(float(finite.std()), 1e-20)
    rows = []
    zeros = np.zeros(base.shape[1], np.float64)
    for index in range(len(base)):
        rows.append(feature_row(finite[index], finite[index - 1] if index else zeros,
                                q[index - 1] if index else zeros,
                                denominator, scale, index > 0, contextual))
    return np.stack(rows)


class IntegerPredictor:
    def __init__(self, weights, contextual):
        self.weights = weights
        self.contextual = contextual

    def setup_bytes(self):
        buffer = io.BytesIO()
        np.savez(buffer, contextual=np.array(self.contextual, np.uint8), **self.weights)
        return buffer.getvalue()

    @classmethod
    def from_setup(cls, data):
        with np.load(io.BytesIO(data), allow_pickle=False) as saved:
            shapes = {"w0": (8, 32), "b0": (32,), "w1": (32, 32), "b1": (32,),
                      "w2": (32, 1), "b2": (1,)}
            weights = {key: saved[key].copy() for key in shapes}
            if (any(value.shape != shapes[key] or value.dtype != np.float32 or not np.isfinite(value).all()
                    for key, value in weights.items())
                    or saved["contextual"].shape != () or int(saved["contextual"]) not in (0, 1)):
                raise ValueError("Invalid integer predictor setup")
            return cls(weights, bool(saved["contextual"]))

    def predict(self, values, scale):
        w = self.weights
        values = np.maximum(values @ w["w0"] + w["b0"], 0)
        values = np.maximum(values @ w["w1"] + w["b1"], 0)
        values = (values @ w["w2"] + w["b2"])[:, 0].astype(np.float64) * scale
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite integer prediction")
        return np.clip(np.rint(values), -(2**30), 2**30).astype(np.int64)


def encode(original_frame, base, predictor, model_identifier, scale):
    arrays, descriptor, side = unpack_predictive(original_frame)
    q, = arrays
    values = features(base, q, scale, predictor.contextual)
    prediction = np.stack([predictor.predict(row, scale) for row in values])
    residual = q.astype(np.int64) - prediction
    if np.any(residual < -(2**31)) or np.any(residual > 2**31 - 1):
        return original_frame
    metadata = {"kind": "neural-causal-integer-v1", "original": descriptor,
                "model_sha256": model_identifier, "scale": float(scale),
                "q_sha256": hashlib.sha256(q.astype("<i4").tobytes()).hexdigest()}
    candidate = pack_predictive([residual.astype(np.int32)], metadata, side)
    return candidate if len(candidate) < len(original_frame) else original_frame


def decode(frame, base, predictor, model_identifier):
    arrays, descriptor, side = unpack_predictive(frame)
    if descriptor.get("kind") != "neural-causal-integer-v1":
        return arrays, descriptor, side
    scale = descriptor.get("scale")
    if (descriptor.get("model_sha256") != model_identifier
            or type(scale) not in (float, int) or not np.isfinite(scale) or scale <= 0
            or len(arrays) != 1 or arrays[0].shape != base.shape):
        raise ValueError("Invalid causal residual metadata")
    residual, = arrays
    q = np.zeros_like(residual)
    finite = np.where(np.isfinite(base), base, 0).astype(np.float64)
    denominator = max(float(finite.std()), 1e-20)
    zeros = np.zeros(base.shape[1], np.float64)
    for index in range(len(base)):
        inputs = feature_row(finite[index], finite[index - 1] if index else zeros,
                             q[index - 1] if index else zeros, denominator, scale,
                             index > 0, predictor.contextual)
        row = residual[index].astype(np.int64) + predictor.predict(inputs, scale)
        if np.any(row < -(2**31)) or np.any(row > 2**31 - 1):
            raise ValueError("Causal coefficient overflow")
        q[index] = row.astype(np.int32)
    if hashlib.sha256(q.astype("<i4").tobytes()).hexdigest() != descriptor.get("q_sha256"):
        raise ValueError("Causal coefficient identity check failed")
    return [q], descriptor["original"], side
