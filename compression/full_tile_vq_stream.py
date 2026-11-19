"""Frozen full-tile VQ comparator with the same compact FAK1 registry contract."""
import json
import zlib

import numpy as np
import torch

from compression.adaptive_coefficient_stream import pack_best, unpack_any
from compression.alphabet_stream import AlphabetStream
from finetuning.joint_anchor_vq import exact_branch_forward
from finetuning.reversible_normalization import decode
from finetuning.robust_escape import encode_robust


class FullTileVQWire:
    def __init__(self, model, counts, identifier):
        self.model = model.eval()
        self.counts = np.asarray(counts, np.int64)
        if self.counts.shape != (2048,) or np.any(self.counts < 0) or np.any(self.counts > 2**40):
            raise ValueError("Invalid full-tile codebook counts")
        self.identifier = identifier
        self.stream = AlphabetStream(self.counts, identifier)

    def setup_bytes(self):
        metadata = json.dumps({"kind": "full-tile-vq-fak1-v1", "identifier": self.identifier, "K": 2048},
                              sort_keys=True, separators=(",", ":")).encode()
        return zlib.compress(metadata + b"\n" + self.counts.astype("<i8").tobytes(), 6)

    @classmethod
    def from_setup(cls, model, data):
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(data, 32769)
        except zlib.error as error:
            raise ValueError("Corrupt full-tile registry") from error
        if len(raw) > 32768 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or b"\n" not in raw:
            raise ValueError("Invalid full-tile registry length")
        metadata, counts = raw.split(b"\n", 1)
        metadata = json.loads(metadata)
        if len(counts) != 2048 * 8 or metadata.get("K") != 2048 or metadata.get("kind") != "full-tile-vq-fak1-v1":
            raise ValueError("Invalid full-tile registry schema")
        return cls(model, np.frombuffer(counts, "<i8"), metadata["identifier"])

    @torch.no_grad()
    def encode(self, raw):
        bits = raw.view(np.uint32)
        if np.all(bits == bits.flat[0]):
            return pack_best([np.array([bits.flat[0]], np.uint32).view(np.int32)],
                             {"kind": "literal-float32-v1", "shape": list(raw.shape)}, b"")
        packet = encode_robust(raw)
        x = torch.from_numpy(packet.values)[None, None].to(next(self.model.parameters()).device)
        _, _, indices, latent, _ = exact_branch_forward(self.model, x)
        return self.stream.encode(indices.cpu().numpy(), raw.shape, latent, packet.side_stream)

    @torch.no_grad()
    def decode(self, frame):
        if frame[:4] in (b"FRC1", b"FRC2"):
            arrays, metadata, side = unpack_any(frame)
            shape = metadata.get("shape")
            if (metadata.get("kind") != "literal-float32-v1" or side or len(arrays) != 1 or arrays[0].shape != (1,)
                    or not isinstance(shape, list) or len(shape) != 2
                    or any(type(n) is not int or n < 1 for n in shape) or shape[0] * shape[1] > 16 * 1024 * 1024):
                raise ValueError("Invalid full-tile literal")
            return np.full(shape, arrays[0].view(np.uint32)[0], np.uint32).view(np.float32)
        indices, shape, latent, side = self.stream.decode(frame)
        device = next(self.model.parameters()).device
        weight = self.model.quantizer.embedding.weight
        q = torch.nn.functional.embedding(torch.from_numpy(indices.astype(np.int64)).to(device), weight)
        q = q.view(1, *latent, weight.shape[1]).permute(0, 3, 1, 2).contiguous()
        values = self.model.decoder(q)[0, 0, :shape[0], :shape[1]].cpu().numpy()
        return decode(values, side)
