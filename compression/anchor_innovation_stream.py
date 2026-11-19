"""FAI1: frozen neural VQ anchors plus transmitted closed-loop Haar innovation."""
import json
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from compression.adaptive_coefficient_stream import pack_best, unpack_any
from compression.adaptive_lifting_stream import decode_frame as decode_literal
from compression.lifting_stream import model_digest
from compression.research_coefficient_stream import quantize
from finetuning.reversible_normalization import decode, parse_side_stream
from finetuning.robust_escape import encode_robust
from finetuning.run_index_search import nearest, vectors

MAGIC = b"FAI1"
HEADER = struct.Struct("<4sIIIII")
MAX_FRAME = 128 * 1024 * 1024


def predict(anchors, odd_count, coefficients):
    if len(coefficients) != 2 or not np.isfinite(coefficients).all():
        raise ValueError("Two finite predictor coefficients required")
    left = anchors[:odd_count].astype(np.float64)
    right = anchors[np.minimum(np.arange(odd_count) + 1, len(anchors) - 1)].astype(np.float64)
    return coefficients[0] * left + coefficients[1] * right


class AnchorInnovationCodec:
    def __init__(self, model, receiver, stream, residual_model, checkpoint_sha256):
        self.model = model.eval()
        self.receiver = receiver.eval()
        self.stream = stream
        self.residual_model = residual_model.eval()
        self.checkpoint_sha256 = checkpoint_sha256

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def analyze(self, raw):
        packet = encode_robust(raw)
        anchors = packet.values[::2].copy()
        tensor = torch.from_numpy(anchors)[None, None].to(self.device)
        h, w = anchors.shape
        z = self.model.encoder(F.pad(tensor, (0, (-w) % 16, 0, (-h) % 16)))
        indices = nearest(z, self.model.quantizer.embedding.weight)
        shape = tuple(z.shape[-2:])
        decoded = self.receiver.decoder(vectors(indices, self.receiver.quantizer.embedding.weight, shape))
        return packet, indices, shape, decoded[0, 0, :h, :w].cpu().numpy()

    @torch.no_grad()
    def decode_anchors(self, frame):
        recovered, shape, latent, side = self.stream.decode(frame)
        if side:
            raise ValueError("Anchor stream must not carry a second normalization")
        indices = torch.from_numpy(recovered.astype(np.int64)).to(self.device)
        decoded = self.receiver.decoder(vectors(indices, self.receiver.quantizer.embedding.weight, latent))
        return decoded[0, 0, :shape[0], :shape[1]].cpu().numpy()

    @torch.no_grad()
    def context(self, raw):
        packet, indices, latent, direct = self.analyze(raw)
        anchor_shape = (len(raw[::2]), raw.shape[1])
        frame = self.stream.encode(indices, anchor_shape, latent, b"")
        reconstructed = self.decode_anchors(frame)
        if direct.tobytes() != reconstructed.tobytes():
            raise AssertionError("Anchor code-vector and stream replay differ")
        return packet, frame, reconstructed

    @torch.no_grad()
    def encode_context(self, raw, context, coefficients, step):
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Positive innovation quantization step required")
        packet, anchor_frame, anchors = context
        if packet.values.shape != raw.shape or anchors.shape != ((len(raw) + 1) // 2, raw.shape[1]):
            raise ValueError("Cached anchor context geometry mismatch")
        bits = raw.view(np.uint32)
        if np.all(bits == bits.flat[0]):
            return pack_best([np.array([bits.flat[0]], np.uint32).view(np.int32)],
                             {"kind": "literal-float32-v1", "shape": list(raw.shape)}, b"")
        prediction = predict(anchors, raw.shape[0] // 2, coefficients)
        residual_frame = b""
        if raw.shape[0] > 1:
            residual = (packet.values[1::2].astype(np.float64) - prediction).astype(np.float32)
            tensor = torch.from_numpy(residual)[None, None].to(self.device)
            bands = self.residual_model.analysis(tensor)
            arrays = [quantize(band.cpu().numpy(), step) for band in bands]
            residual_frame = pack_best(arrays, {
                "kind": "haar-innovation-v1", "shape": list(residual.shape), "step": float(step),
                "model_sha256": model_digest(self.residual_model)}, b"")
        metadata = json.dumps({
            "kind": "vq-anchors-haar-innovation-v1", "shape": list(raw.shape),
            "coefficients": list(map(float, coefficients)), "checkpoint_sha256": self.checkpoint_sha256,
            "residual_model_sha256": model_digest(self.residual_model)},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        body = metadata + packet.side_stream + anchor_frame + residual_frame
        return HEADER.pack(MAGIC, len(metadata), len(packet.side_stream), len(anchor_frame),
                           len(residual_frame), zlib.crc32(body)) + body

    def encode(self, raw, coefficients, step):
        return self.encode_context(raw, self.context(raw), coefficients, step)

    @torch.no_grad()
    def decode(self, frame):
        if frame[:4] in (b"FRC1", b"FRC2"):
            arrays, metadata, _ = unpack_any(frame)
            if metadata.get("kind") != "literal-float32-v1":
                raise ValueError("Only literal fallback allowed outside FAI1")
            return decode_literal(frame, self.residual_model)
        if len(frame) < HEADER.size or len(frame) > MAX_FRAME:
            raise ValueError("Invalid anchor innovation frame length")
        magic, nm, ns, na, nr, checksum = HEADER.unpack_from(frame)
        body = frame[HEADER.size:]
        if magic != MAGIC or nm + ns + na + nr != len(body) or zlib.crc32(body) != checksum:
            raise ValueError("Anchor innovation frame checksum/length mismatch")
        metadata = json.loads(body[:nm])
        side = body[nm:nm + ns]
        shape, _, _, _, _ = parse_side_stream(side)
        if shape[0] * shape[1] > 16 * 1024 * 1024:
            raise ValueError("Anchor innovation allocation limit")
        if (metadata.get("kind") != "vq-anchors-haar-innovation-v1"
                or metadata.get("shape") != list(shape)
                or metadata.get("checkpoint_sha256") != self.checkpoint_sha256
                or metadata.get("residual_model_sha256") != model_digest(self.residual_model)):
            raise ValueError("Anchor innovation receiver identity/geometry mismatch")
        anchors = self.decode_anchors(body[nm + ns:nm + ns + na])
        if anchors.shape != ((shape[0] + 1) // 2, shape[1]):
            raise ValueError("Invalid decoded anchor geometry")
        values = np.empty(shape, np.float32)
        values[::2] = anchors
        if shape[0] > 1:
            arrays, descriptor, extra = unpack_any(body[nm + ns + na:])
            residual_shape = (shape[0] // 2, shape[1])
            step = descriptor.get("step")
            if (descriptor.get("kind") != "haar-innovation-v1" or extra
                    or descriptor.get("shape") != list(residual_shape)
                    or descriptor.get("model_sha256") != model_digest(self.residual_model)
                    or not isinstance(step, (int, float)) or not np.isfinite(step) or step <= 0):
                raise ValueError("Invalid innovation stream")
            expected, h, w = [], *residual_shape
            for axis, _, length in self.residual_model.schedule(residual_shape):
                expected.append((1, 1, h // 2, w) if axis == 2 else (1, 1, h, w // 2))
                if axis == 2:
                    h = (length + 1) // 2
                else:
                    w = (length + 1) // 2
            expected.append((1, 1, h, w))
            if [array.shape for array in arrays] != expected:
                raise ValueError("Invalid innovation coefficient shapes")
            bands = [torch.from_numpy(a.astype(np.float32)).to(self.device) * step for a in arrays]
            residual = self.residual_model.synthesis(bands, residual_shape)[0, 0].cpu().numpy()
            values[1::2] = (residual.astype(np.float64) +
                           predict(anchors, shape[0] // 2, metadata["coefficients"])).astype(np.float32)
        elif nr:
            raise ValueError("Unexpected innovation for a single channel")
        return decode(values, side)
