#!/usr/bin/env python3
"""Read-only scope, train-only selection, checkpoint and nested wire audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.representation_study import digest, read_manifest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    assert (root / 'COMPLETE').exists() and (root / 'CODEBOOKS_READY').exists()
    protocol = json.loads((root / 'protocol.json').read_text())
    result = json.loads((root / 'results.json').read_text())
    selection = json.loads((root / 'selection.json').read_text())
    assert digest(args.source) == protocol['source_checkpoint_sha256']
    assert result['source_state_after_sha256'] == protocol['source_state_before_sha256']
    assert result['source_network_parameter_updates'] == result['validation_signal_windows_read'] == 0
    assert result['centroid_parameter_updates'] == 2 * protocol['budget_per_arm']
    hashes = 0
    for f, expected in protocol['code_sha256'].items():
        assert digest(root / 'code_snapshot' / f) == expected
        hashes += 1
    for f, expected in protocol['input_sha256'].items():
        assert digest(ROOT / f) == expected
        hashes += 1
    assert digest(args.samples_dir / 'samples_manifest.csv') == protocol['manifest_sha256']
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    fit, checks = protocol['fit_train'], protocol['check_train']
    assert all(f in train for f in [*fit, *checks])
    assert not {train[f]['drive_id'] for f in fit} & {train[f]['drive_id'] for f in checks}
    assert set(protocol['sequence']) <= set(fit)
    assert set(protocol['selection_checks']) <= set(checks)
    assert len(protocol['sequence']) == protocol['budget_per_arm']
    for arm in ('latent', 'waveform'):
        folder = root / arm
        history = json.loads((folder / 'history.json').read_text())
        best = min(history, key=lambda row: row['macro_check_reconstruction_distortion'])
        entry = selection[arm]
        assert best['step'] == entry['selected_step']
        assert best['macro_check_reconstruction_distortion'] == entry['selected_check_distortion']
        assert history[-1]['step'] == entry['updates'] == protocol['budget_per_arm']
        assert entry['selected_sha256'] == digest(folder / 'selected.npy')
        books = np.load(folder / 'selected.npy', allow_pickle=False)
        assert books.shape == (2, 32, 64) and books.dtype == np.float32 and np.isfinite(books).all()
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert last['step'] == protocol['budget_per_arm'] and tuple(last['centroids'].shape) == books.shape
        assert len(last['optimizer']['param_groups']) == 1 and len(last['optimizer']['param_groups'][0]['params']) == 1
        assert last['optimizer']['param_groups'][0]['lr'] == 1e-3
        assert last['torch_rng'] is not None and last['cuda_rng'] is not None
        assert result['selection'][arm] == entry
    evaluation = root / 'evaluation'
    assert digest(evaluation / 'results.json') == result['evaluation_results_sha256']
    eval_protocol = json.loads((evaluation / 'protocol.json').read_text())
    assert eval_protocol['external_training_protocol_sha256'] == digest(root / 'protocol.json')
    assert eval_protocol['external_selection_sha256'] == digest(root / 'selection.json')
    assert eval_protocol['fit_train'] == fit and eval_protocol['check_train'] == checks
    for arm in ('latent', 'waveform'):
        assert digest(evaluation / arm / 'codebooks.npy') == selection[arm]['selected_sha256']
        descriptor = json.loads((evaluation / arm / 'descriptor.json').read_text())
        assert descriptor['basis_sha256'] == protocol['basis_sha256']
        assert descriptor['center_sha256'] == protocol['center_sha256']
    subprocess.run([sys.executable, str(ROOT / 'finetuning/audit_product_quantization.py'), '--experiment', str(evaluation),
                    '--source', str(args.source), '--samples-dir', str(args.samples_dir)], check=True)
    wire_audit = json.loads((evaluation / 'audit.json').read_text())
    evaluated = json.loads((evaluation / 'results.json').read_text())
    assert evaluated['external_centroid_parameter_updates'] == result['centroid_parameter_updates']
    if not protocol['smoke_only']:
        for arm in ('latent', 'waveform'):
            assert evaluated['arms'][arm]['macro_normalized_distortion'] == selection[arm]['selected_check_distortion']
    wave, latent = evaluated['arms']['waveform'], evaluated['arms']['latent']
    gain = 1 - wave['guarded_macro_normalized_distortion'] / latent['guarded_macro_normalized_distortion']
    assert result['waveform_guarded_gain_vs_latent'] == gain
    assert result['advance_gate_met'] == (wave['advance_gate_met'] and wave['guarded_macro_normalized_distortion'] <= .99 * latent['guarded_macro_normalized_distortion'])
    output = {'source_network_updates': 0, 'centroid_updates_verified': result['centroid_parameter_updates'],
        'validation_signal_windows_accessed': 0, 'source_state_hash_unchanged': True,
        'snapshot_and_input_hashes_verified': hashes + wire_audit['snapshot_and_input_hashes_verified'],
        'frames_verified': wire_audit['frames_verified'], 'check_windows': len(checks), 'fit_windows': len(fit),
        'advance_gate_met': result['advance_gate_met'], 'auditor_sha256': digest(Path(__file__))}
    write_json(root / 'audit.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
