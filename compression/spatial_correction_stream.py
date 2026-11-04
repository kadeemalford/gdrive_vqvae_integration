"""Explicit mode envelope for unchanged source frames and shared output correction."""
import struct
import zlib

HEADER = struct.Struct('<4sBII')
MAGIC = b'FSC1'


def pack_frame(payload, use_correction):
    if not isinstance(payload, bytes) or not payload:
        raise ValueError('Nonempty source frame bytes required')
    mode = int(use_correction)
    if mode not in (0, 1):
        raise ValueError('Invalid correction mode')
    checksum = zlib.crc32(bytes([mode]) + payload)
    return HEADER.pack(MAGIC, mode, len(payload), checksum) + payload


def unpack_frame(frame):
    if len(frame) < HEADER.size:
        raise ValueError('Truncated correction frame')
    magic, mode, size, checksum = HEADER.unpack_from(frame)
    payload = frame[HEADER.size:]
    if magic != MAGIC or mode not in (0, 1) or len(payload) != size or not payload:
        raise ValueError('Invalid correction frame')
    if zlib.crc32(bytes([mode]) + payload) != checksum:
        raise ValueError('Correction frame checksum mismatch')
    return payload, bool(mode)


def decode_frame(frame, stream, base_model, correction, device):
    """Decode with shared models and stream tables only; no waveform/target access."""
    import numpy as np
    import torch
    from finetuning.run_index_search import vectors
    from finetuning.reversible_normalization import decode

    payload, use_correction = unpack_frame(frame)
    indices, shape, latent, side = stream.decode(payload)
    with torch.no_grad():
        prediction = base_model.decoder(vectors(torch.from_numpy(indices.astype(np.int64)).to(device),
            base_model.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]]
        if use_correction:
            prediction = correction(prediction)
    return decode(prediction[0, 0].cpu().numpy(), side)


def decode_model_frame(frame, stream, base_model, candidate_model, device):
    """Decode FSC1 with a registry selecting a full candidate model, not a filter.

    The inner frame identifies the frozen alphabet/tables. The outer registry
    must bind both waveform checkpoint hashes and declare full-model routing.
    """
    import numpy as np
    import torch
    from finetuning.run_index_search import vectors
    from finetuning.reversible_normalization import decode

    payload, use_candidate = unpack_frame(frame)
    model = candidate_model if use_candidate else base_model
    indices, shape, latent, side = stream.decode(payload)
    with torch.no_grad():
        prediction = model.decoder(vectors(torch.from_numpy(indices.astype(np.int64)).to(device),
            model.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]]
    return decode(prediction[0, 0].cpu().numpy(), side)
