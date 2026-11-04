#!/usr/bin/env python3
"""Decode an E2 validation bundle without source NPZ/HDF5 waveforms."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.scalar_stream import ScalarStream
from finetuning.scalar_rd import ScalarRDCodec
from finetuning.reversible_normalization import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-frames', type=int, default=0, help='0 decodes every frame')
    args = parser.parse_args()
    if args.max_frames < 0:
        raise ValueError('Frame limit must be nonnegative')
    sys.path.insert(0, str(args.repo_root.resolve()))
    experiment = args.experiment_dir.resolve()
    if not (experiment / 'COMPLETE').exists():
        raise ValueError('Completed experiment required')
    result = json.loads((experiment / 'results.json').read_text())
    checkpoint = experiment / 'best.pt'
    with checkpoint.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != result['best_checkpoint_sha256']:
        raise ValueError('Decoder checkpoint fingerprint mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['config']
    if config['architecture'] != ScalarRDCodec.VERSION:
        raise ValueError('Unsupported scalar decoder version')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = ScalarRDCodec(**{key: config[key] for key in ('base_channels', 'latent_channels', 'geometry')})
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.to(args.device).eval()
    bundle = experiment / 'full_validation'
    prior = np.frombuffer((bundle / 'prior.f32').read_bytes(), dtype='<f4').reshape(config['latent_channels'], 512)
    codec = ScalarStream(prior)
    maps = {m['channel_map_id']: m for m in json.loads((bundle / 'maps.json').read_text())}
    records = json.loads((bundle / 'transport.json').read_text())
    if args.max_frames:
        records = records[:args.max_frames]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(records):
        # Only generated flat frame names are valid; never follow paths outside the bundle.
        if Path(row['frame']).name != row['frame'] or Path(row['frame']).suffix != '.srd':
            raise ValueError('Invalid frame path')
        symbols, shape, side, mapping = codec.decode((bundle / row['frame']).read_bytes())
        if mapping != row['channel_map_id'] or mapping not in maps:
            raise ValueError('Missing or inconsistent channel map')
        with torch.no_grad():
            tensor = torch.from_numpy(symbols.copy())[None].to(args.device).float()
            predicted = model.synthesis(tensor, shape)[0, 0].cpu().numpy()
        restored = decode(predicted, side)
        np.savez_compressed(args.output_dir / f'{index:05d}.npz', data=restored,
            source_channel_indices=np.asarray(maps[mapping]['source_channel_indices']),
            provenance_json=np.asarray(json.dumps(row, sort_keys=True)))
    (args.output_dir / 'decoder.json').write_text(json.dumps({'frames': len(records),
        'checkpoint_sha256': actual, 'device': args.device, 'source_waveform_access': False}, indent=2) + '\n')
    print(json.dumps({'decoded_frames': len(records), 'source_waveform_access': False}))


if __name__ == '__main__':
    main()
