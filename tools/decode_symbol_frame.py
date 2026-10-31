#!/usr/bin/env python3
"""Standalone frame decoder; no access to the source DAS window is required."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.symbol_stream import SymbolCodec
from finetuning.model_factory import build_model
from finetuning.latent_geometry import latent_positions
from finetuning.reversible_normalization import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; refusing to overwrite')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    sys.path.insert(0, str(args.repo_root))
    codec = SymbolCodec.from_bytes(args.table.read_bytes())
    with args.checkpoint.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if codec.spec['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Checkpoint differs from the shared model used to encode this frame')
    indices, shape, latent, side = codec.decode(args.frame.read_bytes())
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved.get('finetune', {}).get('normalization') != 'robust-escape-v1':
        raise ValueError('Unsupported normalization policy')
    model = build_model({**saved['config'], 'exact_eval_quantization': True}, saved, args.device).eval()
    del saved
    if indices.size != latent_positions(model, *shape):
        raise ValueError('Frame index count differs from the model geometry')
    with torch.no_grad():
        symbols = torch.from_numpy(indices.astype(np.int64)).to(args.device)
        vectors = model.quantizer.embedding(symbols).view(1, *latent, model.embedding_dim)
        vectors = vectors.permute(0, 3, 1, 2).contiguous()
        predicted = model.decoder(vectors)[0, 0, :shape[0], :shape[1]].cpu().numpy()
    restored = decode(predicted, side)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'checkpoint_sha256': checkpoint_hash, 'table_sha256': codec.identifier.hex(),
                'shape': list(shape), 'latent_shape': list(latent), 'index_count': int(indices.size),
                'normalization': 'robust-escape-v1', 'source_window_required': False,
                'note': 'Physical coordinates are in the separately transmitted channel-map/acquisition metadata'}
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, data=restored, metadata_json=json.dumps(metadata))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
