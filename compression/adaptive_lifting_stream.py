"""Explicit per-band quantization profiles for frozen lifting, FRC1/FRC2 replay."""
import numpy as np
import torch

from compression.adaptive_coefficient_stream import pack_best, unpack_any
from compression.lifting_stream import model_digest
from compression.research_coefficient_stream import quantize
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust


def descriptor(model, shape, global_steps):
    global_steps = np.asarray(global_steps, np.float64)
    if (global_steps.shape != (len(model.log_step),) or not np.isfinite(global_steps).all()
            or np.any(global_steps <= 0)):
        raise ValueError("Positive finite step for each global band required")
    return {"kind": model.VERSION, "shape": list(shape),
            "steps": [float(global_steps[index]) for index in model.band_indices(shape)],
            "model_sha256": model_digest(model)}


@torch.no_grad()
def encode_frame(raw, model, global_steps):
    packet = encode_robust(raw)
    bits = raw.view(np.uint32)
    if np.all(bits == bits.flat[0]):
        return pack_best([np.array([bits.flat[0]], np.uint32).view(np.int32)],
                         {"kind": "literal-float32-v1", "shape": list(raw.shape)}, b"")
    metadata = descriptor(model, raw.shape, global_steps)
    tensor = torch.from_numpy(packet.values)[None, None].to(next(model.parameters()).device)
    bands = model.analysis(tensor)
    arrays = [quantize(band.cpu().numpy(), step) for band, step in zip(bands, metadata["steps"])]
    return pack_best(arrays, metadata, packet.side_stream)


@torch.no_grad()
def decode_frame(frame, model):
    arrays, metadata, side = unpack_any(frame)
    shape = metadata.get("shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)
            or shape[0] * shape[1] > 16 * 1024 * 1024):
        raise ValueError("Invalid adaptive lifting geometry")
    if metadata["kind"] == "literal-float32-v1":
        if len(arrays) != 1 or arrays[0].shape != (1,) or side:
            raise ValueError("Invalid literal frame")
        return np.full(shape, arrays[0].view(np.uint32)[0], np.uint32).view(np.float32)
    if metadata["kind"] != model.VERSION or metadata["model_sha256"] != model_digest(model):
        raise ValueError("Adaptive lifting receiver mismatch")
    h, w = shape
    expected = []
    for axis, _, length in model.schedule(shape):
        expected.append((1, 1, h // 2, w) if axis == 2 else (1, 1, h, w // 2))
        if axis == 2:
            h = (length + 1) // 2
        else:
            w = (length + 1) // 2
    expected.append((1, 1, h, w))
    steps = metadata.get("steps", [])
    if (len(arrays) != len(expected) or [a.shape for a in arrays] != expected
            or len(steps) != len(arrays) or not np.isfinite(steps).all() or min(steps) <= 0):
        raise ValueError("Invalid coefficient geometry/steps")
    bands = [torch.from_numpy(a.astype(np.float32)).to(next(model.parameters()).device) * step
             for a, step in zip(arrays, steps)]
    values = model.synthesis(bands, shape)[0, 0].cpu().numpy()
    return decode(values, side)
