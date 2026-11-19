"""Closed-loop spatial predictor with transmitted anchors AND innovations."""
import numpy as np

from compression.research_coefficient_stream import pack, quantize, unpack
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust


def neighbors(even, odd_count):
    left = even[:odd_count]
    right = even[np.minimum(np.arange(odd_count) + 1, len(even) - 1)]
    return (left + right) * .5


def encode_frame(raw, step, coefficient, predictive=True):
    packet = encode_robust(raw)
    values = packet.values.astype(np.float64)
    if predictive and len(values) > 1:
        anchors = quantize(values[::2], step)
        prediction = coefficient * neighbors(anchors.astype(np.float64) * step, len(values) // 2)
        innovation = quantize(values[1::2] - prediction, step)
        arrays = [anchors, innovation]
    else:
        arrays = [quantize(values, step)]
        predictive = False
    frame = pack(arrays, {"kind": "closed-loop-spatial-v1", "predictive": predictive,
                         "step": step, "coefficient": coefficient, "shape": list(raw.shape)},
                 packet.side_stream)
    return frame


def decode_frame(frame):
    arrays, descriptor, side = unpack(frame)
    if descriptor["kind"] != "closed-loop-spatial-v1":
        raise ValueError("Unexpected spatial codec")
    step = descriptor["step"]
    shape = descriptor["shape"]
    if descriptor["predictive"]:
        anchors, innovation = arrays
        values = np.empty(shape, np.float64)
        values[::2] = anchors.astype(np.float64) * step
        values[1::2] = (innovation.astype(np.float64) * step
                       + descriptor["coefficient"] * neighbors(values[::2], shape[0] // 2))
    else:
        values = arrays[0].astype(np.float64) * step
    return decode(values.astype(np.float32), side)
