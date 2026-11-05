#!/usr/bin/env python3
"""Decode experimental FAK1 on CPU from shared weights/tables, without source data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from finetuning.model_factory import build_model
from finetuning.representation_study import digest
from finetuning.run_index_search import vectors
from finetuning.reversible_normalization import decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--frame', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    descriptor = json.loads((args.package / 'descriptor.json').read_text())
    checkpoint = args.package / 'selected.pt'
    counts = args.package / 'counts.npy'
    if descriptor['format'] != 'FAK1' or digest(checkpoint) != descriptor['checkpoint_sha256'] or digest(counts) != descriptor['counts_sha256']:
        raise ValueError('Incompatible package fingerprint')
    stream = AlphabetStream(np.load(counts, allow_pickle=False), descriptor['checkpoint_sha256'])
    if stream.identifier.hex() != descriptor['identifier']:
        raise ValueError('Incompatible alphabet dictionary')
    indices, shape, latent, side = stream.decode(args.frame.read_bytes())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cpu').eval().requires_grad_(False)
    with torch.no_grad():
        restored = model.decoder(vectors(torch.from_numpy(indices.astype(np.int64)), model.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]][0, 0].numpy()
    output = decode(restored, side)
    if output.shape != shape:
        raise ValueError('Output geometry mismatch')
    np.save(args.output, output)
    print(json.dumps({'shape': shape, 'alphabet': stream.k, 'device': 'cpu', 'source_waveforms_read': False}))


if __name__ == '__main__':
    main()
