#!/usr/bin/env python3
"""Train-only tile-size gradient diagnostics; never update model parameters."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.model_factory import build_model
from finetuning.representation_study import (digest, distortion, load_packet, read_manifest,
                                            tensor_packet, write_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train = read_manifest(args.samples_dir)['train']
    selected = {}
    for family in sorted({r['family'] for r in train}):
        for lo, hi in ((1, 3), (4, 32), (33, 128), (129, 512)):
            choices = [r for r in train if r['family'] == family and lo <= int(r['channels']) <= hi]
            if choices:
                row = min(choices, key=lambda r: r['sample_path'])
                selected[row['sample_path']] = row
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['config'], checkpoint, 'cuda').train()
    version_counters = {name: parameter._version for name, parameter in model.named_parameters()}
    records = []
    for row in selected.values():
        raw, packet = load_packet(row)
        tensor, valid = tensor_packet(packet)
        model.zero_grad(set_to_none=True)
        recon, vq, _, _ = model(tensor)
        temporal = distortion(recon, tensor, valid)
        (temporal + vq).backward()
        norms = {}
        for name in ('encoder', 'decoder', 'quantizer'):
            norms[name] = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                for p in getattr(model, name).parameters() if p.grad is not None))
        assert all(math.isfinite(value) for value in norms.values())
        records.append({'sample_path': row['sample_path'], 'family': row['family'],
            'configuration': row['configuration'], 'channels': raw.shape[0], 'cells': raw.size,
            'temporal_loss': float(temporal.detach()), 'vq_loss': float(vq.detach()), 'gradient_l2': norms})
    assert all(p._version == version_counters[name] for name, p in model.named_parameters())
    write_json(args.output_dir / 'results.json', {'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': digest(args.checkpoint), 'source_sha256': digest(Path(__file__)),
        'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'objective': '0.5*MSE + 0.5*L1 + VQ; ordinary cells; robust-escape-v1',
        'parameter_updates': 0, 'no_test_inference': True,
        'selection': 'first train metadata path per family and channel-size bin: 1-3,4-32,33-128,129-512',
        'limitations': 'Stratified descriptive sample, not mean dataset gradient contributions or a causal sampler comparison',
        'records': records})
    (args.output_dir / 'COMPLETE').touch()
    print(json.dumps({'windows': len(records), 'updates': 0}), flush=True)


if __name__ == '__main__':
    main()
