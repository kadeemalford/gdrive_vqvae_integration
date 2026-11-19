"""Byte-exact rANS backend and framed direct-quantization diagnostic.

No constriction model is used: its floating categorical API requantizes.
All rANS interval arithmetic below uses transmitted 20-bit integer CDFs.
"""
import bisect
import hashlib
import io
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np

from compression.feasibility_tables import (
    MAX_CELLS, MODES, RESET, SUPPORTS, TOTAL, Tables, bucket, contexts,
    fit_tables, reconstruct, symbols,
)


LOWER = 1 << 31
FRAME = struct.Struct("<BIIIII")  # mode, times, physical count, q escapes, payload bytes, reserved


def portable_object(transport, installation):
    """Actual executable object; declared platform is Python 3 standard library.

    ZIP_STORED avoids adding a second entropy coder to the byte contrast.
    Every object includes independent decoder source and its full installation.
    """
    buffer = io.BytesIO()
    source = Path(__file__).with_name("feasibility_portable.py").read_bytes()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in (("__main__.py", source), ("installation.ftb", installation),
                           ("transport.fqs", transport)):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 17, 0, 0, 0))
            archive.writestr(info, data)
    return buffer.getvalue()


def ans_encode(s, rows, tables):
    state = LOWER
    emitted = bytearray()
    frequencies, cdf = tables.frequencies, tables.cdf
    for symbol, row in zip(s[::-1], rows[::-1]):
        f, start = int(frequencies[row, symbol]), int(cdf[row, symbol])
        ceiling = ((LOWER >> 20) << 8) * f
        while state >= ceiling:
            emitted.append(state & 255)
            state >>= 8
        state = ((state // f) << 20) + state % f + start
    return struct.pack("<Q", state) + bytes(reversed(emitted))


def _frame(q, positions, bits, tables, mode):
    flat = q.reshape(-1)
    if mode == 0:
        payload = flat.astype("<i4").tobytes()
        tail = b""
        nesc = 0
    else:
        c = contexts(q, positions, tables.channel_ids, tables.mode if mode == 2 else "marginal")
        rows = tables.mapping[c.reshape(-1)] if mode == 2 else np.zeros(flat.size, dtype=np.uint32)
        s = symbols(q, tables.k).reshape(-1)
        escaped = s == 2 * tables.k + 1
        tail = flat[escaped].astype("<i4").tobytes()
        nesc = int(escaped.sum())
        payload = ans_encode(s, rows, tables)
    physical = np.column_stack((positions, bits)).astype("<u4").tobytes()
    body = FRAME.pack(mode, q.shape[0], len(positions), nesc, len(payload), 0)
    body += physical + tail + payload
    return body + struct.pack("<I", zlib.crc32(body))


def encode(q, positions, bits, tables, forced=None):
    """Return transport and exact, nonoverlapping byte ledger.

    `forced` is for diagnostics/fixtures only; normal mode pays a per-frame ID
    and chooses the shortest of contextual, marginal and raw q.
    """
    if (q.dtype != np.int32 or q.ndim != 2 or q.shape[1] != len(tables.channel_ids)
            or not q.size or q.size > MAX_CELLS):
        raise ValueError("Invalid quantized region")
    positions, bits = np.asarray(positions), np.asarray(bits)
    if (positions.ndim != 1 or bits.shape != positions.shape
            or np.any(positions >= q.size) or np.any(positions[1:] <= positions[:-1])
            or np.any(q.reshape(-1)[positions] != 0)):
        raise ValueError("Invalid physical escapes")
    if forced is not None and forced not in (0, 1, 2):
        raise ValueError("Unknown forced mode")
    installation = tables.dump()
    frames = []
    counts = {"raw": 0, "marginal": 0, "contextual": 0}
    ledger = {"object_header_and_integrity": 76, "frame_headers_and_integrity": 0,
              "physical_escapes": 0, "alphabet_escapes": 0, "entropy_or_raw": 0,
              "entropy_finalization": 0}
    channels = q.shape[1]
    for start in range(0, q.shape[0], RESET):
        block = q[start:start + RESET]
        lo, hi = start * channels, (start + len(block)) * channels
        selected = (positions >= lo) & (positions < hi)
        pos, rawbits = positions[selected] - lo, bits[selected]
        modes = (forced,) if forced is not None else ((0, 1) if tables.mode == "marginal" else (0, 1, 2))
        candidates = [(_frame(block, pos, rawbits, tables, mode), mode) for mode in modes]
        frame, mode = min(candidates, key=lambda item: (len(item[0]), item[1]))
        _, _, nphys, nesc, payload, _ = FRAME.unpack_from(frame)
        ledger["frame_headers_and_integrity"] += FRAME.size + 4
        ledger["physical_escapes"] += nphys * 8
        ledger["alphabet_escapes"] += nesc * 4
        ledger["entropy_or_raw"] += payload - (8 if mode else 0)
        ledger["entropy_finalization"] += 8 if mode else 0
        counts[("raw", "marginal", "contextual")[mode]] += 1
        frames.append(frame)
    body = b"FQS1" + hashlib.sha256(installation).digest() + struct.pack("<II", len(q), len(frames))
    body += b"".join(frames)
    transport = body + hashlib.sha256(body).digest()
    assert sum(ledger.values()) == len(transport)
    return transport, {"transport_bytes": len(transport), "installation_bytes": len(installation),
                       "table_object_bytes": len(transport) + len(installation),
                       "components": ledger, "frame_modes": counts}


def decode(transport, installation, max_cells=MAX_CELLS, trace=None):
    """Independent causal decoder. Receives no raw/check statistics.

    Optional trace records (position, context, q) after decoding each symbol.
    Integrity is accidental-corruption detection, not authentication.
    """
    tables = Tables.load(installation)
    if (len(transport) < 76 or transport[:4] != b"FQS1"
            or hashlib.sha256(transport[:-32]).digest() != transport[-32:]
            or transport[4:36] != hashlib.sha256(installation).digest()):
        raise ValueError("Transport integrity or installation mismatch")
    times, nframes = struct.unpack_from("<II", transport, 36)
    channels = len(tables.channel_ids)
    if (not times or times * channels > min(max_cells, MAX_CELLS)
            or nframes != (times + RESET - 1) // RESET):
        raise ValueError("Invalid geometry or frame count")
    q = np.empty((times, channels), dtype=np.int32)
    physical_positions, physical_bits = [], []
    offset, t = 44, 0
    for frame_id in range(nframes):
        if offset + FRAME.size + 4 > len(transport) - 32:
            raise ValueError("Truncated frame header")
        mode, nt, nphys, nesc, nbytes, reserved = FRAME.unpack_from(transport, offset)
        cells = nt * channels
        if (mode > 2 or reserved or nt != min(RESET, times - t) or nphys > cells
                or nesc > cells or (mode == 0 and (nesc or nbytes != 4 * cells))):
            raise ValueError("Invalid frame metadata")
        end = offset + FRAME.size + 8 * nphys + 4 * nesc + nbytes
        if end + 4 > len(transport) - 32:
            raise ValueError("Truncated frame")
        if struct.unpack_from("<I", transport, end)[0] != zlib.crc32(transport[offset:end]):
            raise ValueError("Frame checksum failure")
        cursor = offset + FRAME.size
        physical = np.frombuffer(transport, dtype="<u4", count=2 * nphys, offset=cursor).reshape(-1, 2)
        positions = physical[:, 0]
        if np.any(positions >= cells) or np.any(positions[1:] <= positions[:-1]):
            raise ValueError("Invalid physical escape positions")
        special = set(int(i) for i in positions)
        cursor += 8 * nphys
        escapes = np.frombuffer(transport, dtype="<i4", count=nesc, offset=cursor)
        cursor += 4 * nesc
        payload = transport[cursor:end]
        block = q[t:t + nt].reshape(-1)
        if mode == 0:
            block[:] = np.frombuffer(payload, dtype="<i4")
        else:
            if len(payload) < 8:
                raise ValueError("Truncated ANS state")
            state, = struct.unpack_from("<Q", payload)
            if not LOWER <= state < (LOWER << 8):
                raise ValueError("Invalid initial ANS state")
            ptr, ei = 8, 0
            previous = [17] * channels
            for i in range(cells):
                ch = i % channels
                spatial = (bucket(block[i - 1]) if ch and tables.channel_ids[ch] == tables.channel_ids[ch - 1] + 1
                           and i - 1 not in special else 17)
                context = (0 if mode == 1 or tables.mode == "marginal" else
                           previous[ch] if tables.mode == "temporal" else previous[ch] * 18 + spatial)
                row = 0 if mode == 1 else int(tables.mapping[context])
                cdf = tables.cdf_lists[row]
                residue = state & (TOTAL - 1)
                symbol = bisect.bisect_right(cdf, residue) - 1
                f, start = cdf[symbol + 1] - cdf[symbol], cdf[symbol]
                state = f * (state >> 20) + residue - start
                while state < LOWER:
                    if ptr >= len(payload):
                        raise ValueError("Truncated ANS payload")
                    state = (state << 8) | payload[ptr]
                    ptr += 1
                if symbol == 2 * tables.k + 1:
                    if ei >= nesc:
                        raise ValueError("Missing alphabet escape")
                    value = int(escapes[ei])
                    ei += 1
                    if -tables.k <= value <= tables.k:
                        raise ValueError("Noncanonical alphabet escape")
                else:
                    value = symbol - tables.k
                block[i] = value
                previous[ch] = 17 if i in special else bucket(value)
                if trace is not None:
                    trace.append((t * channels + i, context, value))
            if ptr != len(payload) or state != LOWER or ei != nesc:
                raise ValueError("Noncanonical final ANS state or unused bytes")
        if np.any(block[positions] != 0):
            raise ValueError("Physical escape must carry placeholder q=0")
        physical_positions.extend((positions.astype(np.uint64) + t * channels).tolist())
        physical_bits.extend(physical[:, 1].tolist())
        t += nt
        offset = end + 4
    if offset != len(transport) - 32:
        raise ValueError("Trailing frame bytes")
    positions = np.asarray(physical_positions, dtype=np.uint32)
    bits = np.asarray(physical_bits, dtype=np.uint32)
    return q, reconstruct(q, positions, bits, tables.epsilon)


def select_support(q, positions, bits, epsilon, channel_ids):
    """Evaluate actual fit-object bytes, never contextual/check performance."""
    candidates = []
    for k in SUPPORTS:
        tables = fit_tables(q, positions, epsilon, channel_ids, k, "marginal")
        _, cost = encode(q, positions, bits, tables)
        candidates.append({"k": k, **cost})
    best = min(candidates, key=lambda c: (c["table_object_bytes"], c["k"]))
    return best["k"], candidates
