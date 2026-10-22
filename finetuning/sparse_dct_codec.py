"""Transparent rate control: sparse 2D orthonormal DCT + fixed-width scalar codes.

Data-dependent coefficient selection is transmitted explicitly. No trained model,
temporal decimation, clipping of the original, or free oracle reconstruction.
"""
import math
import struct

import numpy as np
from scipy.fft import dctn, idctn

try:
    from .reversible_normalization import encode, decode, parse_side_stream
except ImportError:
    from reversible_normalization import encode, decode, parse_side_stream

HEADER = struct.Struct("<4sIBIf")
MAGIC = b"FSD1"


def pack(values, width):
    values = np.asarray(values, dtype=np.uint32).reshape(-1)
    bits = ((values[:, None] >> np.arange(width - 1, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder="big").tobytes()


def unpack(stream, width, count):
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8), bitorder="big")
    expected = width * count
    if len(stream) != math.ceil(expected / 8) or (bits[expected:] != 0).any():
        raise ValueError("Invalid packed length/padding")
    if count == 0:
        return np.empty(0, dtype=np.uint32)
    return np.sum(bits[:expected].reshape(count, width).astype(np.uint32) << np.arange(width - 1, -1, -1, dtype=np.uint32), axis=1, dtype=np.uint32)


def compress(raw, budget_bytes, quant_bits=8):
    if not 2 <= quant_bits <= 16 or budget_bytes < 1:
        raise ValueError("Positive budget and 2–16 quantization bits required")
    packet = encode(raw)
    coefficients = dctn(packet.values, type=2, norm="ortho", workers=1).reshape(-1)
    index_bits = max(1, math.ceil(math.log2(len(coefficients))))
    entry_bits = index_bits + quant_bits
    overhead = HEADER.size + len(packet.side_stream)
    if budget_bytes < overhead:
        raise ValueError("Budget does not fit mandatory metadata")
    count = min(len(coefficients), (budget_bytes - overhead) * 8 // entry_bits)
    if count == len(coefficients):
        indices = np.arange(count, dtype=np.uint32)
    elif count:
        indices = np.sort(np.argpartition(np.abs(coefficients), len(coefficients) - count)[-count:]).astype(np.uint32)
    else:
        indices = np.empty(0, dtype=np.uint32)
    selected = coefficients[indices].astype(np.float64)
    scale = np.float32(max(float(np.abs(selected).max()) if count else 0., 1e-30))
    qmax = 2 ** (quant_bits - 1) - 1
    quantized = (np.rint(selected / float(scale) * qmax).astype(np.int32) + qmax).astype(np.uint32)
    entries = (indices.astype(np.uint32) << quant_bits) | quantized
    # With larger shapes index_bits+quant_bits can exceed uint32: reject explicitly.
    if entry_bits > 32:
        raise ValueError("Window/quantizer exceeds packed uint32 format")
    stream = HEADER.pack(MAGIC, count, quant_bits, len(packet.side_stream), float(scale)) + packet.side_stream + pack(entries, entry_bits)
    if len(stream) > budget_bytes:
        raise AssertionError("Codec exceeds budget")
    total_energy = float(np.square(coefficients.astype(np.float64)).sum())
    stats = {"encoded_bytes": len(stream), "budget_bytes": int(budget_bytes), "retained_coefficients": int(count),
             "total_coefficients": len(coefficients), "index_bits_per_coefficient": index_bits, "quant_bits": quant_bits,
             "prequantization_energy_fraction_retained": float(np.square(selected).sum() / total_energy) if total_energy else 1.}
    return stream, stats


def decompress(stream):
    if len(stream) < HEADER.size:
        raise ValueError("Truncated codec header")
    magic, count, quant_bits, side_size, scale = HEADER.unpack_from(stream)
    if magic != MAGIC or not 2 <= quant_bits <= 16 or not np.isfinite(scale) or scale <= 0 or HEADER.size + side_size > len(stream):
        raise ValueError("Invalid codec header")
    side = stream[HEADER.size:HEADER.size + side_size]
    shape, _, _, _, _ = parse_side_stream(side)
    total = math.prod(shape)
    index_bits = max(1, math.ceil(math.log2(total)))
    entry_bits = index_bits + quant_bits
    if count > total or entry_bits > 32:
        raise ValueError("Invalid coefficient count/width")
    entries = unpack(stream[HEADER.size + side_size:], entry_bits, count)
    indices = entries >> quant_bits
    quantized = entries & (2 ** quant_bits - 1)
    qmax = 2 ** (quant_bits - 1) - 1
    if np.any(indices >= total) or (count > 1 and np.any(indices[1:] <= indices[:-1])) or np.any(quantized > 2 * qmax):
        raise ValueError("Invalid sparse coefficient entries")
    coefficients = np.zeros(total, dtype=np.float32)
    coefficients[indices] = ((quantized.astype(np.float64) - qmax) * scale / qmax).astype(np.float32)
    normalized = idctn(coefficients.reshape(shape), type=2, norm="ortho", workers=1)
    return decode(normalized, side)
