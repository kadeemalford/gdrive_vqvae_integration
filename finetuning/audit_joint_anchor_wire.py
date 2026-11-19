"""Decompose measured H2 bytes into symbol positions, payload and framing."""
import argparse
import json
import zlib
from pathlib import Path

from compression.alphabet_stream import HEADER as ALPHABET_HEADER
from compression.joint_anchor_vq_stream import HEADER as JOINT_HEADER
from finetuning.audit_prospective_preparation import write_json


def alphabet_stats(frame):
    fields = list(ALPHABET_HEADER.unpack_from(frame))
    magic, version, _, _, _, _, lh, lw, side, _, payload, crc = fields
    if magic != b"FAK1" or version != 1 or ALPHABET_HEADER.size + side + payload != len(frame):
        raise ValueError("Invalid alphabet framing")
    fields[-1] = 0
    if zlib.crc32(ALPHABET_HEADER.pack(*fields) + frame[ALPHABET_HEADER.size:]) != crc:
        raise ValueError("Alphabet audit checksum mismatch")
    return lh * lw, payload - 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--selected-update", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    totals = {"full_positions": 0, "joint_positions": 0, "full_payload_bytes": 0,
              "joint_payload_bytes": 0, "full_frame_bytes": 0, "joint_frame_bytes": 0, "frames": 0}
    for path in sorted((args.expanded / f"check_{args.selected_update:06d}").glob("*.fjh")):
        frame = path.read_bytes()
        magic, nm, ns, na, nr, crc = JOINT_HEADER.unpack_from(frame)
        body = frame[JOINT_HEADER.size:]
        if magic != b"FJH1" or nm + ns + na + nr != len(body) or zlib.crc32(body) != crc:
            raise ValueError("Joint audit checksum mismatch")
        for packet in (body[nm + ns:nm + ns + na], body[nm + ns + na:]):
            if packet:
                positions, payload = alphabet_stats(packet)
                totals["joint_positions"] += positions
                totals["joint_payload_bytes"] += payload
        reference = (args.expanded / "practical_reference" / (path.stem + ".fak")).read_bytes()
        positions, payload = alphabet_stats(reference)
        totals["full_positions"] += positions
        totals["full_payload_bytes"] += payload
        totals["full_frame_bytes"] += len(reference)
        totals["joint_frame_bytes"] += len(frame)
        totals["frames"] += 1
    if not totals["frames"]:
        raise ValueError("No paired scientific frames")
    totals.update(
        position_increase=totals["joint_positions"] / totals["full_positions"] - 1,
        full_payload_bits_per_position=8 * totals["full_payload_bytes"] / totals["full_positions"],
        joint_payload_bits_per_position=8 * totals["joint_payload_bytes"] / totals["joint_positions"],
        full_framing_side_bytes=totals["full_frame_bytes"] - totals["full_payload_bytes"],
        joint_framing_side_bytes=totals["joint_frame_bytes"] - totals["joint_payload_bytes"])
    write_json(args.output, {"totals": totals, "parameter_updates": 0, "waveform_access": False,
                            "interpretation": "Measured count/layout tax, not a promise that latents can be discarded losslessly"})
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
