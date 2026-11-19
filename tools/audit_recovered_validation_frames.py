"""Check recovered bytes against remote SHA inventory and replay paid symbols."""
import argparse
import hashlib
import json
import shlex
import zlib
from pathlib import Path

import numpy as np

from compression.alphabet_stream import AlphabetStream
from compression.joint_anchor_vq_stream import HEADER
from finetuning.audit_prospective_preparation import write_json
from finetuning.reversible_normalization import parse_side_stream
from tools.sync_research_remote import REMOTE, command, connect


def inventory(root):
    result = {}
    for folder in sorted(root.iterdir()):
        manifest = folder / "frames_manifest.jsonl"
        if not folder.is_dir() or not manifest.exists():
            continue
        digest = hashlib.sha256()
        total, frames = 0, 0
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            content = (folder / "frames" / row["frame"]).read_bytes()
            if len(content) != row["bytes"]:
                raise ValueError("Recovered frame length mismatch")
            digest.update(row["frame"].encode() + b"\0" + hashlib.sha256(content).digest())
            frames += 1
            total += len(content)
        result[folder.name] = {"frames": frames, "bytes": total, "ordered_sha256_inventory": digest.hexdigest()}
    return result


def check_symbols(folder):
    metadata, raw_counts = zlib.decompress((folder / "registry.zlib").read_bytes()).split(b"\n", 1)
    metadata = json.loads(metadata)
    identifier = metadata["identifier"]
    if folder.name == "mature_reference":
        streams = [AlphabetStream(np.frombuffer(raw_counts, "<i8"), identifier)]
    else:
        counts = np.frombuffer(raw_counts, "<i8").reshape(2, 2048)
        streams = [AlphabetStream(row, hashlib.sha256(bytes.fromhex(identifier) + bytes([i])).hexdigest())
                   for i, row in enumerate(counts)]
    positions = 0
    for line in (folder / "frames_manifest.jsonl").read_text().splitlines():
        frame = (folder / "frames" / json.loads(line)["frame"]).read_bytes()
        if frame[:4] == b"FAK1":
            ids, shape, _, side = streams[0].decode(frame)
            if parse_side_stream(side)[0] != shape:
                raise ValueError("Full-tile normalization geometry mismatch")
            positions += len(ids)
        elif frame[:4] == b"FJH1":
            _, nm, ns, na, nr, crc = HEADER.unpack_from(frame)
            body = frame[HEADER.size:]
            if nm + ns + na + nr != len(body) or zlib.crc32(body) != crc:
                raise ValueError("Recovered joint wrapper CRC mismatch")
            descriptor = json.loads(body[:nm])
            shape = parse_side_stream(body[nm:nm + ns])[0]
            if descriptor["identifier"] != identifier or descriptor["shape"] != list(shape):
                raise ValueError("Recovered joint descriptor mismatch")
            for index, packet in enumerate((body[nm + ns:nm + ns + na], body[nm + ns + na:])):
                if packet:
                    ids, branch_shape, _, side = streams[index].decode(packet)
                    expected = ((shape[0] + 1) // 2 if index == 0 else shape[0] // 2, shape[1])
                    if branch_shape != expected or side:
                        raise ValueError("Recovered branch geometry mismatch")
                    positions += len(ids)
        else:
            # This population explicitly contains no literal windows.
            raise ValueError("Unexpected recovered frame mode")
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    local = inventory(args.root)
    # The same inventory function runs remotely; only frame bytes/manifests are read.
    import inspect
    script = "import hashlib,json\nfrom pathlib import Path\n" + inspect.getsource(inventory)
    script += f"\nprint(json.dumps(inventory(Path({str(args.root)!r}))))\n"
    client = connect()
    try:
        remote = json.loads(command(client, f"cd {REMOTE}\n.venv/bin/python -c {shlex.quote(script)}"))
    finally:
        client.close()
    if local != remote or len(local) != 5:
        raise ValueError("Local and remote recovered frame SHA inventories differ")
    positions = {name: check_symbols(args.root / name) for name in local}
    result = {"passed": True, "local_remote_sha256_inventories_equal": True, "candidates": local,
              "decoded_index_positions": positions, "symbol_and_frame_crc_verified": True,
              "acquired_waveforms_reopened": False, "neural_inference": False}
    write_json(args.root / "RECOVERED_FRAME_INTEGRITY.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
