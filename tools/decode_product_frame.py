#!/usr/bin/env python3
"""Standalone CPU reader for experimental product-VQ FAK1 packages, no waveform input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from compression.product_quantization import replay_product, validate_codebooks
from finetuning.model_factory import build_model
from finetuning.representation_study import digest
from finetuning.reversible_normalization import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    descriptor = json.loads((args.package / 'descriptor.json').read_text())
    source_hash = digest(args.checkpoint)
    if (descriptor['format'] != 'FAK1' or descriptor['representation'] not in ('product-vq-2-v1', 'pca-product-vq-2-v1')
            or descriptor['source_checkpoint_sha256'] != source_hash or descriptor['groups'] != 2):
        raise ValueError('Incompatible product package or source checkpoint')
    books_path, counts_path = args.package / 'codebooks.npy', args.package / 'counts.npy'
    book_hash = digest(books_path)
    if book_hash != descriptor['codebooks_sha256'] or digest(counts_path) != descriptor['counts_sha256']:
        raise ValueError('Incompatible shared codebooks or symbol counts')
    basis = center = None
    fingerprint = bytes.fromhex(source_hash) + bytes.fromhex(book_hash) + descriptor['representation'].encode()
    if descriptor['representation'] == 'pca-product-vq-2-v1':
        basis_path, center_path = args.package / 'basis.npy', args.package / 'center.npy'
        if digest(basis_path) != descriptor['basis_sha256'] or digest(center_path) != descriptor['center_sha256']:
            raise ValueError('Incompatible shared product transform')
        basis = torch.from_numpy(np.load(basis_path, allow_pickle=False))
        center = torch.from_numpy(np.load(center_path, allow_pickle=False))
        fingerprint += bytes.fromhex(descriptor['basis_sha256']) + bytes.fromhex(descriptor['center_sha256'])
    binding = hashlib.sha256(fingerprint).hexdigest()
    if binding != descriptor['binding_sha256']:
        raise ValueError('Product representation fingerprint mismatch')
    stream = AlphabetStream(np.load(counts_path, allow_pickle=False), binding)
    if stream.identifier.hex() != descriptor['identifier']:
        raise ValueError('Product stream fingerprint mismatch')
    books = torch.from_numpy(np.load(books_path, allow_pickle=False))
    groups, alphabet, width = validate_codebooks(books)
    if groups != 2 or alphabet != descriptor['alphabet'] or books.dtype != torch.float32:
        raise ValueError('Invalid product codebook layout')
    indices, shape, symbols, side = stream.decode(args.frame.read_bytes())
    if symbols[1] % groups:
        raise ValueError('Nonintegral product latent grid')
    latent = symbols[0], symbols[1] // groups
    q = replay_product(torch.from_numpy(indices.astype(np.int64)).view(-1, groups), latent, books, basis, center)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['config']['embedding_dim'] != groups * width:
        raise ValueError('Source decoder embedding width differs')
    model = build_model(saved['config'], saved, 'cpu').eval().requires_grad_(False)
    with torch.no_grad():
        prediction = model.decoder(q)[..., :shape[0], :shape[1]][0, 0].numpy()
    restored = decode(prediction, side)
    if restored.shape != shape:
        raise ValueError('Invalid decoded waveform shape')
    np.save(args.output, restored)
    print(json.dumps({'shape': shape, 'groups': groups, 'alphabet_per_group': alphabet,
                      'device': 'cpu', 'source_waveforms_read': False}))


if __name__ == '__main__':
    main()
