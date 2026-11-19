"""FJH1: exact two-book H2 replay, compact charged FAK1 count registries."""
import hashlib
import json
import struct
import zlib

import numpy as np
import torch

from compression.adaptive_coefficient_stream import pack_best, unpack_any
from compression.alphabet_stream import AlphabetStream
from finetuning.joint_anchor_vq import VERSION
from finetuning.reversible_normalization import decode, parse_side_stream
from finetuning.robust_escape import encode_robust

MAGIC = b"FJH1"
HEADER = struct.Struct("<4sIIIII")
MAX_CELLS = 16 * 1024 * 1024


class JointAnchorWire:
    def __init__(self, model, counts, identifier):
        counts = np.asarray(counts, np.int64)
        if (counts.shape != (2, 2048) or np.any(counts < 0)
                or np.any(counts > 2**40) or len(bytes.fromhex(identifier)) != 32):
            raise ValueError("Invalid two-book registry")
        self.model = model.eval()
        self.counts = counts.copy()
        self.identifier = identifier
        self.streams = [AlphabetStream(row, hashlib.sha256(bytes.fromhex(identifier) + bytes([index])).hexdigest())
                        for index, row in enumerate(counts)]

    def setup_bytes(self):
        descriptor = {"kind": VERSION, "mode": self.model.mode, "identifier": self.identifier,
                      "coding": "FAK1-fixed-range-two-book", "shape": [2, 2048]}
        data = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        return zlib.compress(data + b"\n" + self.counts.astype("<i8").tobytes(), 6)

    @classmethod
    def from_setup(cls, model, data):
        codec = zlib.decompressobj()
        try:
            raw = codec.decompress(data, 65537)
        except zlib.error as error:
            raise ValueError("Corrupt H2 registry") from error
        if (len(raw) > 65536 or not codec.eof or codec.unused_data or codec.unconsumed_tail
                or b"\n" not in raw):
            raise ValueError("Invalid H2 registry compression")
        metadata, counts = raw.split(b"\n", 1)
        metadata = json.loads(metadata)
        if (len(counts) != 2 * 2048 * 8 or metadata.get("kind") != VERSION
                or metadata.get("mode") != model.mode or metadata.get("shape") != [2, 2048]
                or metadata.get("coding") != "FAK1-fixed-range-two-book"):
            raise ValueError("Incompatible H2 registry")
        return cls(model, np.frombuffer(counts, "<i8").reshape(2, 2048), metadata["identifier"])

    @torch.no_grad()
    def encode(self, raw):
        bits = raw.view(np.uint32)
        if np.all(bits == bits.flat[0]):
            return pack_best([np.array([bits.flat[0]], np.uint32).view(np.int32)],
                             {"kind": "literal-float32-v1", "shape": list(raw.shape)}, b"")
        packet = encode_robust(raw)
        device = next(self.model.parameters()).device
        _, _, info = self.model(torch.from_numpy(packet.values)[None, None].to(device))
        anchor_shape = ((len(raw) + 1) // 2, raw.shape[1])
        anchor = self.streams[0].encode(info["anchor_ids"].cpu().numpy(), anchor_shape, info["anchor_latent"], b"")
        innovation = b""
        if len(raw) > 1:
            shape = (len(raw) // 2, raw.shape[1])
            innovation = self.streams[1].encode(
                info["innovation_ids"].cpu().numpy(), shape, info["innovation_latent"], b"")
        metadata = json.dumps({"kind": VERSION, "mode": self.model.mode,
                               "shape": list(raw.shape), "identifier": self.identifier},
                              sort_keys=True, separators=(",", ":")).encode()
        body = metadata + packet.side_stream + anchor + innovation
        return HEADER.pack(MAGIC, len(metadata), len(packet.side_stream), len(anchor),
                           len(innovation), zlib.crc32(body)) + body

    @torch.no_grad()
    def decode(self, frame):
        if frame[:4] in (b"FRC1", b"FRC2"):
            arrays, metadata, side = unpack_any(frame)
            shape = metadata.get("shape")
            if (metadata.get("kind") != "literal-float32-v1" or side or len(arrays) != 1
                    or arrays[0].shape != (1,) or not isinstance(shape, list) or len(shape) != 2
                    or any(type(n) is not int or n < 1 for n in shape)
                    or shape[0] * shape[1] > MAX_CELLS):
                raise ValueError("Invalid H2 literal")
            return np.full(shape, arrays[0].view(np.uint32)[0], np.uint32).view(np.float32)
        if len(frame) < HEADER.size:
            raise ValueError("Truncated joint anchor frame")
        magic, nm, ns, na, nr, checksum = HEADER.unpack_from(frame)
        body = frame[HEADER.size:]
        if magic != MAGIC or nm + ns + na + nr != len(body) or zlib.crc32(body) != checksum:
            raise ValueError("Joint anchor frame checksum/length mismatch")
        metadata = json.loads(body[:nm])
        side = body[nm:nm + ns]
        shape, _, _, _, _ = parse_side_stream(side)
        if (shape[0] * shape[1] > MAX_CELLS or metadata.get("shape") != list(shape)
                or metadata.get("kind") != VERSION or metadata.get("mode") != self.model.mode
                or metadata.get("identifier") != self.identifier):
            raise ValueError("Joint anchor receiver identity/geometry mismatch")
        anchor, anchor_shape, anchor_latent, extra = self.streams[0].decode(body[nm + ns:nm + ns + na])
        if anchor_shape != ((shape[0] + 1) // 2, shape[1]) or extra:
            raise ValueError("Invalid anchor payload")
        innovation = innovation_latent = None
        if shape[0] > 1:
            innovation, odd_shape, innovation_latent, extra = self.streams[1].decode(body[nm + ns + na:])
            if odd_shape != (shape[0] // 2, shape[1]) or extra:
                raise ValueError("Invalid innovation payload")
        elif nr:
            raise ValueError("Unexpected innovation for one channel")
        device = next(self.model.parameters()).device
        output = self.model.decode_indices(
            torch.from_numpy(anchor.astype(np.int64)).to(device), anchor_latent,
            torch.from_numpy(innovation.astype(np.int64)).to(device) if innovation is not None else None,
            innovation_latent, shape)
        return decode(output[0, 0].cpu().numpy(), side)
