"""Independent receiver replay for the prospective lifting codec."""
import hashlib
import io

import numpy as np
import torch

from compression.research_coefficient_stream import pack, quantize, unpack
from finetuning.learned_lifting import LiftingCodec
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust


def receiver_state(model):
    return {key: value for key, value in model.state_dict().items()
            if model.learned or not key.startswith(("predict.", "update."))}


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(receiver_state(model).items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    digest.update(str(sorted(model.config.items())).encode())
    return digest.hexdigest()


def receiver_bytes(model):
    stream = io.BytesIO()
    torch.save({"config": model.config, "state_dict": receiver_state(model),
                "version": model.VERSION}, stream)
    return stream.getvalue()


def load_receiver(data, device="cpu"):
    payload = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
    if payload["version"] != LiftingCodec.VERSION:
        raise ValueError("Unexpected lifting version")
    model = LiftingCodec(**payload["config"]).to(device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=model.learned)
    if unexpected or any(not key.startswith(("predict.", "update.")) for key in missing):
        raise ValueError("Incomplete lifting receiver state")
    return model.eval()


@torch.no_grad()
def encode_frame(raw, model):
    packet = encode_robust(raw)
    tensor = torch.from_numpy(packet.values)[None, None].to(next(model.parameters()).device)
    bands = model.analysis(tensor)
    indices = model.band_indices(raw.shape)
    steps = [float(model.steps[index]) for index in indices]
    arrays = [quantize(band.cpu().numpy(), step) for band, step in zip(bands, steps)]
    # Literal constants are a common engineering mode, not an architectural gain.
    bits = raw.view(np.uint32)
    if np.all(bits == bits.flat[0]):
        return pack([np.array([int(bits.flat[0])], dtype=np.uint32).view(np.int32)],
                    {"kind": "literal-float32-v1", "shape": list(raw.shape)}, b"")
    return pack(arrays, {"kind": model.VERSION, "shape": list(raw.shape),
                        "steps": steps, "model_sha256": model_digest(model)}, packet.side_stream)


@torch.no_grad()
def decode_frame(frame, model):
    arrays, descriptor, side = unpack(frame)
    shape = descriptor.get("shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)
            or shape[0] * shape[1] > 16 * 1024 * 1024):
        raise ValueError("Invalid lifting geometry")
    if descriptor["kind"] == "literal-float32-v1":
        if len(arrays) != 1 or arrays[0].shape != (1,) or side:
            raise ValueError("Invalid literal payload")
        value = arrays[0].view(np.uint32)[0]
        raw = np.full(descriptor["shape"], value, dtype=np.uint32).view(np.float32)
        return raw
    if descriptor["kind"] != model.VERSION or descriptor["model_sha256"] != model_digest(model):
        raise ValueError("Receiver model mismatch")
    steps = descriptor.get("steps", [])
    if (len(steps) != len(arrays) or len(arrays) != len(model.schedule(shape)) + 1
            or not np.isfinite(steps).all() or min(steps) <= 0):
        raise ValueError("Invalid lifting quantization steps")
    device = next(model.parameters()).device
    bands = [torch.from_numpy(array.astype(np.float32)).to(device) * step
             for array, step in zip(arrays, steps)]
    values = model.synthesis(bands, descriptor["shape"])[0, 0].cpu().numpy()
    return decode(values, side)
