"""Frozen fit-only probability tables for the report-107 diagnostic."""
import hashlib
import json
import struct
import zlib

import numpy as np


TOTAL = 1 << 20
SUPPORTS = (15, 63, 255, 1023, 4095)
RESET = 1000
MAX_CELLS = 2_000_000
MAX_TABLE_BYTES = 128 << 20
MODES = ("marginal", "temporal", "spatiotemporal")


def quantize(values, epsilon):
    """Quantize once. Physical escapes use q=0 and an unavailable context."""
    x = np.asarray(values)
    if x.dtype != np.float32 or x.ndim != 2 or not x.size:
        raise ValueError("Expected a nonempty float32 [time, channel] array")
    if x.size > MAX_CELLS or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid epsilon or region exceeds the 2M-cell memory cap")
    if not np.isfinite(2 * epsilon):
        raise ValueError("Nonfinite quantization step")
    bits = x.view(np.uint32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        candidate = np.rint(x.astype(np.float64) / (2 * epsilon))
        valid = (np.isfinite(candidate) & (candidate >= -(1 << 31))
                 & (candidate <= (1 << 31) - 1))
        q = np.where(valid, candidate, 0).astype(np.int32)
        reconstruction = (q.astype(np.float64) * (2 * epsilon)).astype(np.float32)
        valid &= np.isfinite(x) & np.isfinite(reconstruction)
        valid &= np.abs(x.astype(np.float64) - reconstruction.astype(np.float64)) <= epsilon
    # Signed zeros and nonfinite payloads are preserved, not normalized away.
    valid &= ~((x == 0) & ((bits >> 31) != 0))
    q[~valid] = 0
    positions = np.flatnonzero(~valid).astype(np.uint32)
    return q, positions, bits.reshape(-1)[positions].copy()


def reconstruct(q, positions, bits, epsilon):
    with np.errstate(over="ignore", invalid="ignore"):
        x = (q.astype(np.float64) * (2 * epsilon)).astype(np.float32)
    x.view(np.uint32).reshape(-1)[positions] = bits
    return x


def bucket(q):
    q = int(q)
    if q == 0:
        return 0
    return 1 + (8 if q < 0 else 0) + min(7, abs(q).bit_length() - 1)


def contexts(q, positions, channel_ids, mode):
    """Encoder-only vectorization of strictly causal decoder rules."""
    if mode not in MODES:
        raise ValueError("Unknown context mode")
    ids = np.asarray(channel_ids)
    if (ids.shape != (q.shape[1],) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(np.diff(ids) <= 0)):
        raise ValueError("Channel IDs must be strictly increasing physical indices")
    if mode == "marginal":
        return np.zeros(q.shape, dtype=np.int32)
    a = np.abs(q.astype(np.int64))
    exponent = np.zeros(q.shape, dtype=np.int32)
    nonzero = a > 0
    exponent[nonzero] = np.minimum(7, np.floor(np.log2(a[nonzero]))).astype(np.int32)
    b = np.where(nonzero, 1 + 8 * (q < 0) + exponent, 0)
    b.reshape(-1)[positions] = 17
    previous = np.full(q.shape, 17, dtype=np.int32)
    previous[1:] = b[:-1]
    previous[::RESET] = 17
    if mode == "temporal":
        return previous
    spatial = np.full(q.shape, 17, dtype=np.int32)
    spatial[:, 1:] = np.where(np.diff(ids) == 1, b[:, :-1], 17)
    return previous * 18 + spatial


def integer_frequencies(probabilities):
    """Largest-remainder apportionment with exact symbol-ID tie breaking."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid probability vector")
    if not np.isclose(p.sum(), 1., atol=1e-12, rtol=0) or len(p) >= TOTAL:
        raise ValueError("Probability vector must sum to one")
    scaled = (TOTAL - len(p)) * p
    floor = np.floor(scaled)
    f = 1 + floor.astype(np.uint32)
    remaining = TOTAL - int(f.sum())
    order = np.lexsort((np.arange(len(p)), -(scaled - floor)))
    f[order[:remaining]] += 1
    assert int(f.sum()) == TOTAL and np.all(f > 0)
    return f


def symbols(q, k):
    return np.where(np.abs(q.astype(np.int64)) <= k, q.astype(np.int64) + k,
                    2 * k + 1).astype(np.int32)


class Tables:
    """Installation contains integer tables, backoff map and complete geometry."""

    def __init__(self, k, mode, epsilon, channel_ids, frequencies, mapping, probabilities=None):
        self.k, self.mode, self.epsilon = int(k), mode, float(epsilon)
        self.channel_ids = [int(i) for i in channel_ids]
        self.frequencies = np.asarray(frequencies, dtype=np.uint32)
        self.mapping = np.asarray(mapping, dtype=np.uint32)
        self.probabilities = probabilities
        self.cdf = np.concatenate((
            np.zeros((len(self.frequencies), 1), dtype=np.uint64),
            np.cumsum(self.frequencies, axis=1, dtype=np.uint64)), axis=1)
        self.cdf_lists = self.cdf.tolist()

    def dump(self):
        config = json.dumps({
            "version": 1, "k": self.k, "mode": self.mode, "epsilon": self.epsilon,
            "channel_ids": self.channel_ids, "reset": RESET, "total": TOTAL,
            "rows": len(self.frequencies), "contexts": len(self.mapping),
            "bucket": "zero,sign*min(7,floor(log2(abs(q)))),unavailable",
            "smoothing": "Jeffreys; n<256 backoff; otherwise n/(n+256)",
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        raw = self.mapping.astype("<u4").tobytes() + self.frequencies.astype("<u4").tobytes()
        body = b"FTB1" + struct.pack("<I", len(config)) + config + zlib.compress(raw, 6)
        return body + hashlib.sha256(body).digest()

    @classmethod
    def load(cls, blob):
        if len(blob) < 40 or hashlib.sha256(blob[:-32]).digest() != blob[-32:]:
            raise ValueError("Installation integrity failure")
        if blob[:4] != b"FTB1":
            raise ValueError("Unknown installation")
        length, = struct.unpack_from("<I", blob, 4)
        try:
            cfg = json.loads(blob[8:8 + length])
            k, mode = cfg["k"], cfg["mode"]
            nctx = {"marginal": 1, "temporal": 18, "spatiotemporal": 324}[mode]
            rows = cfg["rows"]
            if (k not in SUPPORTS or cfg["version"] != 1 or cfg["total"] != TOTAL
                    or cfg["reset"] != RESET or cfg["contexts"] != nctx
                    or type(rows) is not int or not 1 <= rows <= nctx + 1
                    or not np.isfinite(cfg["epsilon"]) or cfg["epsilon"] <= 0):
                raise ValueError("Invalid installation parameters")
            ids = cfg["channel_ids"]
            if (not ids or len(ids) > MAX_CELLS or any(type(i) is not int or i < 0 for i in ids)
                    or any(b <= a for a, b in zip(ids, ids[1:]))):
                raise ValueError("Invalid channel geometry")
            expected = 4 * (nctx + rows * (2 * k + 2))
            dec = zlib.decompressobj()
            raw = dec.decompress(blob[8 + length:-32], expected + 1)
            if len(raw) != expected or not dec.eof or dec.unused_data or dec.unconsumed_tail:
                raise ValueError("Invalid table blob length")
            a = np.frombuffer(raw, dtype="<u4")
            mapping, frequencies = a[:nctx], a[nctx:].reshape(rows, 2 * k + 2)
            if (np.any(mapping >= rows) or np.any(frequencies == 0)
                    or np.any(frequencies.sum(axis=1, dtype=np.uint64) != TOTAL)):
                raise ValueError("Invalid integer CDF")
            return cls(k, mode, cfg["epsilon"], ids, frequencies, mapping)
        except (KeyError, TypeError, OverflowError, zlib.error, struct.error) as exc:
            raise ValueError("Malformed installation") from exc


def fit_tables(q, positions, epsilon, channel_ids, k, mode):
    if k not in SUPPORTS or mode not in MODES:
        raise ValueError("Invalid fit configuration")
    c = contexts(q, positions, channel_ids, mode).reshape(-1)
    s = symbols(q, k).reshape(-1)
    nctx = {"marginal": 1, "temporal": 18, "spatiotemporal": 324}[mode]
    alphabet = 2 * k + 2
    if nctx * alphabet * 24 > MAX_TABLE_BYTES:
        raise ValueError("Fit table memory ceiling exceeded")
    marginal_counts = np.bincount(s, minlength=alphabet)
    marginal = (marginal_counts + .5) / (len(s) + .5 * alphabet)
    probabilities = [marginal]
    mapping = np.zeros(nctx, dtype=np.uint32)
    if mode != "marginal":
        counts = np.bincount(c * alphabet + s, minlength=nctx * alphabet).reshape(nctx, alphabet)
        for context, row in enumerate(counts):
            n = int(row.sum())
            if n >= 256:
                p = (row + .5) / (n + .5 * alphabet)
                p = n / (n + 256) * p + 256 / (n + 256) * marginal
                mapping[context] = len(probabilities)
                probabilities.append(p)
    p = np.asarray(probabilities)
    return Tables(k, mode, epsilon, channel_ids,
                  np.asarray([integer_frequencies(row) for row in p]), mapping, p)


def score(tables, q, positions):
    c = contexts(q, positions, tables.channel_ids, tables.mode).reshape(-1)
    s = symbols(q, tables.k).reshape(-1)
    rows = tables.mapping[c]
    cell_bits = -np.log2(tables.frequencies[rows, s] / TOTAL)
    integer_bits = float(cell_bits.sum())
    float_bits = (float(-np.log2(tables.probabilities[rows, s]).sum())
                  if tables.probabilities is not None else None)
    return {"float_nll_bits": float_bits, "integer_cdf_nll_bits": integer_bits,
            "reset_first_sample_integer_bits": float(cell_bits.reshape(q.shape)[::RESET].sum()),
            "reset_first_sample_cells": int(q[::RESET].size),
            "nonreset_integer_bits": float(cell_bits.reshape(q.shape)[np.arange(len(q)) % RESET != 0].sum()),
            "nonreset_cells": int(q[np.arange(len(q)) % RESET != 0].size),
            "alphabet_escape_count": int(np.count_nonzero(s == 2 * tables.k + 1)),
            "physical_escape_count": len(positions)}
