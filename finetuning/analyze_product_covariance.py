#!/usr/bin/env python3
"""Train-only covariance diagnosis; fixed rotation opportunity gate, no waveform fitting."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.model_factory import build_model
from finetuning.representation_study import digest, read_manifest, load_packet, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one remote RTX 4090')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads((args.probe / 'protocol.json').read_text())
    assert (args.probe / 'COMPLETE').exists()
    assert digest(args.source) == protocol['source_checkpoint_sha256']
    path = args.probe / 'covariance_diagnostic.json'
    if path.exists():
        raise FileExistsError('Preserve existing covariance diagnosis')
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    del saved
    pooled = []
    with torch.no_grad():
        for sample in protocol['fit_train']:
            row = train[sample]
            assert digest(ROOT / sample) == protocol['input_sha256'][sample]
            _, packet = load_packet(row)
            target = torch.from_numpy(packet.values)[None, None].cuda()
            h, w = target.shape[-2:]
            z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
            flat = z.permute(0, 2, 3, 1).reshape(-1, 128)
            positions = torch.linspace(0, len(flat) - 1, min(256, len(flat)), device='cuda').long()
            pooled.append(flat[positions].double())
        x = torch.cat(pooled)
        center = x.mean(0)
        centered = x - center
        covariance = centered.T @ centered / len(x)
        eigenvalues, basis = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues, basis = eigenvalues[order].clamp_min(0), basis[:, order]
        # Canonical signs, then alternate principal axes between equal-width subspaces.
        sign = torch.sign(basis[basis.abs().argmax(0), torch.arange(128, device='cuda')])
        basis = basis * sign[None, :]
        balanced = torch.cat((torch.arange(0, 128, 2, device='cuda'), torch.arange(1, 128, 2, device='cuda')))
        rotation = basis[:, balanced].float()
        mass = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
        cross = float(2 * covariance[:64, 64:].square().sum() / covariance.square().sum().clamp_min(1e-30))
        top32 = float(mass[:32].sum())
        # Diagnostic opportunity criterion, not a codec/physical acceptance threshold.
        allow_rotation_probe = cross >= .10 or top32 >= .80
        np.save(args.probe / 'balanced_pca_basis.npy', rotation.cpu().numpy())
        np.save(args.probe / 'balanced_pca_center.npy', center.float().cpu().numpy())
    assert digest(args.source) == protocol['source_checkpoint_sha256']
    result = {'code_sha256': digest(Path(__file__)), 'source_sha256': protocol['source_checkpoint_sha256'],
        'fit_windows': len(protocol['fit_train']), 'fit_vectors': len(x), 'validation_signal_windows_read': 0,
        'neural_parameter_updates': 0, 'top8_variance_fraction': float(mass[:8].sum()),
        'top16_variance_fraction': float(mass[:16].sum()), 'top32_variance_fraction': top32,
        'cross_subspace_covariance_squared_mass': cross,
        'effective_rank': float(torch.exp(-(mass * mass.clamp_min(1e-30).log()).sum())),
        'basis_sha256': digest(args.probe / 'balanced_pca_basis.npy'),
        'center_sha256': digest(args.probe / 'balanced_pca_center.npy'),
        'rotation_probe_rule': 'one fixed balanced-PCA control iff cross covariance mass >=0.10 OR top32 variance >=0.80; no OPQ or rotation sweep',
        'rotation_probe_allowed': allow_rotation_probe,
        'limitations': ['latent covariance is not waveform fidelity or decoder-available improvement',
            'orthogonal transform preserves distances only before quantization; all trained shared assets must be charged',
            'same fit-only vectors, no whitening, no waveform-loss optimization or new training examples']}
    write_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
