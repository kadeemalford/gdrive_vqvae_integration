#!/usr/bin/env python3
"""Independent CPU reader for FNE1/FAS1 and exact-numeric affine research frames."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.affine_codec_study import decode_affine_frame
from finetuning.model_factory import build_model
from finetuning.representation_study import digest
from finetuning.run_index_search import FrozenIndexStream
from compression.partition_bundle import decode_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite existing decoded output')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    descriptor = json.loads((args.package / 'descriptor.json').read_text())
    source_hash = digest(args.checkpoint)
    if source_hash != descriptor['source_checkpoint_sha256'] or digest(args.package / 'counts.npy') != descriptor['counts_sha256']:
        raise ValueError('Affine package model/table identity mismatch')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cpu').eval().requires_grad_(False)
    stream = FrozenIndexStream(np.load(args.package / 'counts.npy', allow_pickle=False), source_hash)
    decoder = decode_bundle if descriptor.get('format') == 'FPB1' else decode_affine_frame
    decoded = decoder(args.frame.read_bytes(), stream, model, 'cpu')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        np.save(output, decoded, allow_pickle=False)
    print(json.dumps({'shape': list(decoded.shape), 'device': 'cpu', 'source_waveforms_read': False,
                      'nonfinite_cells_preserved': int((~np.isfinite(decoded)).sum()), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
