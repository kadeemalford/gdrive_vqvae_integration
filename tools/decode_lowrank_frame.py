#!/usr/bin/env python3
"""Standalone CPU low-rank reader: only transmitted factors/side, no original data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.lowrank_stream import decode_normalized, decompress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.frame.read_bytes()
    _, _, metadata = decode_normalized(data)
    result = decompress(data)
    np.save(args.output, result)
    print(json.dumps({'shape': list(result.shape), 'device': 'cpu', 'original_waveforms_read': False,
                      'learned_weights_read': False, **metadata}))


if __name__ == '__main__':
    main()
