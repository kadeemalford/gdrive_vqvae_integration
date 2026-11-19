"""Independent standard-library-only decoder for FQS1/FTB1.

When placed as __main__.py in an object zip, writes reconstructed float32 bytes
and q int32 bytes. Python 3 with its standard library is the declared platform;
no numpy, scipy, encoder, original waveform or fit/check statistics is needed.
"""
import bisect
import hashlib
import json
import math
import struct
import sys
import zipfile
import zlib


TOTAL = 1 << 20
LOWER = 1 << 31
MAX_CELLS = 2_000_000
FRAME = struct.Struct("<BIIIII")


def _bucket(value):
    return 0 if value == 0 else 1 + (8 if value < 0 else 0) + min(7, abs(value).bit_length() - 1)


def decode_bytes(transport, installation):
    """Return (q little-endian int32 bytes, waveform float32 bytes)."""
    if (len(installation) < 40 or installation[:4] != b"FTB1"
            or hashlib.sha256(installation[:-32]).digest() != installation[-32:]):
        raise ValueError("Invalid installation integrity")
    length, = struct.unpack_from("<I", installation, 4)
    cfg = json.loads(installation[8:8 + length])
    k, mode, rows, ids = cfg["k"], cfg["mode"], cfg["rows"], cfg["channel_ids"]
    nctx = {"marginal": 1, "temporal": 18, "spatiotemporal": 324}.get(mode)
    epsilon = cfg["epsilon"]
    if (k not in (15, 63, 255, 1023, 4095) or cfg["version"] != 1
            or cfg["reset"] != 1000 or cfg["total"] != TOTAL or cfg["contexts"] != nctx
            or type(rows) is not int or not 1 <= rows <= nctx + 1
            or not math.isfinite(epsilon) or epsilon <= 0
            or not ids or len(ids) > MAX_CELLS
            or any(type(i) is not int or i < 0 for i in ids)
            or any(b <= a for a, b in zip(ids, ids[1:]))):
        raise ValueError("Invalid installation parameters")
    alphabet = 2 * k + 2
    size = 4 * (nctx + rows * alphabet)
    dec = zlib.decompressobj()
    raw = dec.decompress(installation[8 + length:-32], size + 1)
    if len(raw) != size or not dec.eof or dec.unused_data or dec.unconsumed_tail:
        raise ValueError("Invalid installation payload")
    mapping = struct.unpack_from(f"<{nctx}I", raw)
    if any(row >= rows for row in mapping):
        raise ValueError("Invalid backoff map")
    cdfs = []
    for row in range(rows):
        frequencies = struct.unpack_from(f"<{alphabet}I", raw, 4 * (nctx + row * alphabet))
        cdf = [0]
        for frequency in frequencies:
            if not frequency:
                raise ValueError("Zero frequency")
            cdf.append(cdf[-1] + frequency)
        if cdf[-1] != TOTAL:
            raise ValueError("Invalid CDF total")
        cdfs.append(cdf)
    if (len(transport) < 76 or transport[:4] != b"FQS1"
            or hashlib.sha256(transport[:-32]).digest() != transport[-32:]
            or transport[4:36] != hashlib.sha256(installation).digest()):
        raise ValueError("Invalid transport integrity")
    times, nframes = struct.unpack_from("<II", transport, 36)
    channels = len(ids)
    if not times or times * channels > MAX_CELLS or nframes != (times + 999) // 1000:
        raise ValueError("Invalid geometry")
    qbytes, waveform = bytearray(), bytearray()
    offset, t = 44, 0
    for frame_id in range(nframes):
        if offset + FRAME.size + 4 > len(transport) - 32:
            raise ValueError("Truncated frame header")
        coding, nt, nphys, nesc, nbytes, reserved = FRAME.unpack_from(transport, offset)
        cells = nt * channels
        if (coding > 2 or reserved or nt != min(1000, times - t) or nphys > cells
                or nesc > cells or (coding == 0 and (nesc or nbytes != 4 * cells))):
            raise ValueError("Invalid frame")
        end = offset + FRAME.size + 8 * nphys + 4 * nesc + nbytes
        if end + 4 > len(transport) - 32:
            raise ValueError("Truncated frame")
        if struct.unpack_from("<I", transport, end)[0] != zlib.crc32(transport[offset:end]):
            raise ValueError("Invalid frame CRC")
        cursor = offset + FRAME.size
        specials = {}
        last = -1
        for _ in range(nphys):
            position, bits = struct.unpack_from("<II", transport, cursor)
            if not last < position < cells:
                raise ValueError("Invalid physical position")
            specials[position] = bits
            last = position
            cursor += 8
        escapes = struct.unpack_from(f"<{nesc}i", transport, cursor)
        cursor += 4 * nesc
        payload = transport[cursor:end]
        if coding == 0:
            values = struct.unpack(f"<{cells}i", payload)
        else:
            if len(payload) < 8:
                raise ValueError("Truncated rANS state")
            state, = struct.unpack_from("<Q", payload)
            if not LOWER <= state < LOWER << 8:
                raise ValueError("Invalid rANS state")
            values = []
            previous, ptr, ei = [17] * channels, 8, 0
            for i in range(cells):
                ch = i % channels
                spatial = (_bucket(values[-1]) if ch and ids[ch] == ids[ch - 1] + 1
                           and i - 1 not in specials else 17)
                context = (0 if coding == 1 or mode == "marginal" else
                           previous[ch] if mode == "temporal" else previous[ch] * 18 + spatial)
                cdf = cdfs[0 if coding == 1 else mapping[context]]
                residue = state & (TOTAL - 1)
                symbol = bisect.bisect_right(cdf, residue) - 1
                state = (cdf[symbol + 1] - cdf[symbol]) * (state >> 20) + residue - cdf[symbol]
                while state < LOWER:
                    if ptr >= len(payload):
                        raise ValueError("Truncated rANS payload")
                    state = state << 8 | payload[ptr]
                    ptr += 1
                if symbol == alphabet - 1:
                    if ei >= nesc:
                        raise ValueError("Missing alphabet escape")
                    value = escapes[ei]
                    ei += 1
                    if -k <= value <= k:
                        raise ValueError("Noncanonical alphabet escape")
                else:
                    value = symbol - k
                values.append(value)
                previous[ch] = 17 if i in specials else _bucket(value)
            if state != LOWER or ptr != len(payload) or ei != nesc:
                raise ValueError("Invalid rANS finalization")
        qbytes.extend(struct.pack(f"<{cells}i", *values))
        for i, value in enumerate(values):
            if i in specials:
                if value != 0:
                    raise ValueError("Invalid physical placeholder")
                waveform.extend(struct.pack("<I", specials[i]))
            else:
                try:
                    waveform.extend(struct.pack("<f", value * (2 * epsilon)))
                except OverflowError as exc:
                    raise ValueError("Nonfinite reconstruction") from exc
        offset, t = end + 4, t + nt
    if offset != len(transport) - 32:
        raise ValueError("Trailing bytes")
    return bytes(qbytes), bytes(waveform)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python object.pyz output.q.int32 output.waveform.float32")
    with zipfile.ZipFile(sys.argv[0]) as archive:
        q, waveform = decode_bytes(archive.read("transport.fqs"), archive.read("installation.ftb"))
    for name, data in zip(sys.argv[1:], (q, waveform)):
        with open(name, "xb") as stream:
            stream.write(data)


if __name__ == "__main__":
    main()
