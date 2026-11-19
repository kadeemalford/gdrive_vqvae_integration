import hashlib
import struct
import zlib
import unittest

import numpy as np

from compression.full_tile_compact_bundle import (
    ENTRY, HEADER, Reader, compact_tile, pack,
)
from compression.partition_bundle import pack_bundle
from compression.research_coefficient_stream import pack as literal_pack
from finetuning.audit_feasibility_compact_bundle import independent_proof


def bundle(bits=0, overlay_bits=(), positions=(), shape=(2, 3)):
    frame = literal_pack([np.array([bits], dtype=np.uint32).view(np.int32)],
                         {"kind": "literal-float32-v1", "shape": list(shape)}, b"")
    raw = struct.pack(f"<{len(positions)}I", *positions) + struct.pack(f"<{len(overlay_bits)}I", *overlay_bits)
    body = frame + zlib.compress(raw)
    overlay = struct.pack("<4sIII", b"FZC1", len(frame), len(positions), zlib.crc32(body)) + body
    return pack_bundle(shape, [(0, 0, shape[0], overlay)]), frame


def test_literal_empty_and_identical_overlay():
    for bits in (0, 0x80000000, 0x3f800000):
        positions = (0, 4) if bits in (0, 0x80000000) else ()
        original, frame = bundle(bits, (bits,) * len(positions), positions)
        mode, stored = compact_tile(original)
        assert mode == 1 and stored == frame
        proof = independent_proof(original, mode, (2, 3), stored)
        assert proof["branch"] == "literal_decoded_bits_equal"


def test_signed_zero_mismatch_is_intact_fallback():
    original, _ = bundle(0, (0x80000000,), (0,))
    assert compact_tile(original) == (0, original)
    encoded = pack([original], b"opaque geometry")
    reader = Reader(encoded)
    assert reader.geometry == b"opaque geometry"
    assert reader.tile(0) == (0, (2, 3), original)


def test_multi_fragment_fallback():
    original, _ = bundle()
    from compression.partition_bundle import unpack_bundle
    payload = unpack_bundle(original)[1][0][3]
    multi = pack_bundle((4, 3), [(0, 0, 2, payload), (0, 2, 2, payload)])
    assert compact_tile(multi) == (0, multi)


def test_index_access_and_integrity():
    originals = [bundle()[0], bundle(0, (0x80000000,), (0,))[0]]
    data = pack(originals, b"geometry")
    reader = Reader(data)
    assert reader.tile(1)[0] == 0
    assert reader.tile(0)[0] == 1
    for position in (0, HEADER.size, len(data) - 33, len(data) - 1):
        corrupt = bytearray(data)
        corrupt[position] ^= 1
        with unittest.TestCase().assertRaises(ValueError):
            Reader(bytes(corrupt))


def test_invalid_index_even_with_recomputed_hashes():
    data = bytearray(pack([bundle()[0]], b"g"))
    struct.pack_into("<Q", data, HEADER.size, 0)
    data[12:44] = hashlib.sha256(data[HEADER.size:HEADER.size + ENTRY.size]).digest()
    data[-32:] = hashlib.sha256(data[:-32]).digest()
    with unittest.TestCase().assertRaisesRegex(ValueError, "offset"):
        Reader(bytes(data))


def test_bad_overlay_rejected():
    original, frame = bundle(0, (0,), (6,))
    with unittest.TestCase().assertRaisesRegex(ValueError, "positions"):
        compact_tile(original)


def test_unknown_future_frame_preserves_fallback():
    body = b"FUT1unknown" + zlib.compress(b"")
    outer = struct.pack("<4sIII", b"FZC1", 11, 0, zlib.crc32(body)) + body
    original = pack_bundle((2, 3), [(0, 0, 2, outer)])
    assert compact_tile(original) == (0, original)


def test_fak1_empty_overlay_symbolic_proof_and_nonempty_fallback():
    from compression.full_tile_compact_bundle import FAK
    symbols = np.zeros(1, dtype="<u2")
    payload = struct.pack("<I", zlib.crc32(symbols.tobytes())) + b"\0\0"
    fields = [b"FAK1", 1, 0, 2048, 2, 3, 1, 1, 0, b"x" * 32, len(payload), 0]
    fields[-1] = zlib.crc32(FAK.pack(*fields) + payload)
    frame = FAK.pack(*fields) + payload
    for count in (0, 1):
        body = frame + zlib.compress(b"" if not count else struct.pack("<II", 0, 0))
        outer = struct.pack("<4sIII", b"FZC1", len(frame), count, zlib.crc32(body)) + body
        original = pack_bundle((2, 3), [(0, 0, 2, outer)])
        mode, stored = compact_tile(original)
        if count:
            assert (mode, stored) == (0, original)
        else:
            assert stored == frame and mode == 1
            proof = independent_proof(original, mode, (2, 3), stored)
            assert not proof["waveform_roundtrip_performed"]


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
