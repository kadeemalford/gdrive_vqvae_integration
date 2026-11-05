#!/usr/bin/env python3
"""CPU-only FPI1 reader: no learned assets or source waveform required."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.predictive_stream import decode_normalized, decompress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frame = args.frame.read_bytes()
    _, _, metadata = decode_normalized(frame)
    output = decompress(frame)
    np.save(args.output, output)
    print(json.dumps({'shape': list(output.shape), 'device': 'cpu', 'source_waveforms_read': False,
                      'learned_assets_read': False, **metadata}))


if __name__ == '__main__':
    main()
