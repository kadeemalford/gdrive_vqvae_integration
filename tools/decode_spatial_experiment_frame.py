#!/usr/bin/env python3
"""Decode one FSC1 research frame without source NPZ/HDF5 or target waveforms."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.spatial_correction_stream import decode_frame, decode_model_frame
from finetuning.model_factory import build_model
from finetuning.representation_study import digest
from finetuning.run_index_search import FrozenIndexStream
from finetuning.run_spatial_boundary_coding import ReverseSpatial
from finetuning.spatial_output_correction import SpatialOutputCorrection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--kind', choices=('correction', 'orientation', 'full-model'), required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing decoded output')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    root = args.experiment_dir
    protocol = json.loads((root / 'protocol.json').read_text())
    if digest(args.base_checkpoint) != protocol['source_sha256']:
        raise ValueError('Base checkpoint identity mismatch')
    source_hash = digest(args.base_checkpoint)
    saved = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    base = build_model(saved['config'], saved, 'cpu').eval().requires_grad_(False)
    del saved
    stream = FrozenIndexStream(np.load(root / 'counts.npy', allow_pickle=False), source_hash)
    frame = args.frame.read_bytes()
    if args.kind == 'orientation':
        registry = json.loads((root / 'registry.json').read_text())
        if registry['source_sha256'] != source_hash or registry['builtin_transform_code_sha256'] != digest(ROOT / 'finetuning/run_spatial_boundary_coding.py'):
            raise ValueError('Orientation registry mismatch')
        restored = decode_frame(frame, stream, base, ReverseSpatial(), 'cpu')
    else:
        registry = json.loads((root / args.arm / 'registry.json').read_text())
        path = root / f'{args.arm}_selected.pt'
        if args.kind == 'correction':
            if registry['source_sha256'] != source_hash or registry['correction_sha256'] != digest(path):
                raise ValueError('Correction registry mismatch')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            correction = SpatialOutputCorrection(saved['spatial'], saved['hidden'])
            correction.load_state_dict(saved['state_dict'], strict=True)
            correction.eval().requires_grad_(False)
            restored = decode_frame(frame, stream, base, correction, 'cpu')
        else:
            if registry['kind'] != 'full_model_routing' or registry['mode_zero_model_sha256'] != source_hash or registry['mode_one_model_sha256'] != digest(path):
                raise ValueError('Full model registry mismatch')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            candidate = build_model(saved['config'], saved, 'cpu').eval().requires_grad_(False)
            restored = decode_model_frame(frame, stream, base, candidate, 'cpu')
    if not np.isfinite(restored).all():
        raise ValueError('Nonfinite decoded waveform')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        np.save(output, restored, allow_pickle=False)
    print(json.dumps({'output': str(args.output), 'shape': list(restored.shape), 'device': 'cpu', 'original_waveforms_required': False}))


if __name__ == '__main__':
    main()
