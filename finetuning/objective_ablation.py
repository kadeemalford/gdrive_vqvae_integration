"""Objective-only continuation helpers and conservative historical-control checks."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from finetuning.representation_study import digest


def reconstruction_loss(predicted, target, valid, kind):
    if kind not in ('mixed', 'mse'):
        raise ValueError('Unknown reconstruction objective')
    if predicted.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool or not valid.any():
        raise ValueError('Matching tensors and a nonempty Boolean ordinary-cell mask required')
    squared = F.mse_loss(predicted[valid], target[valid])
    return squared if kind == 'mse' else .5 * squared + .5 * F.l1_loss(predicted[valid], target[valid])


def validate_lr_control_metadata(protocol, summary):
    expected = {'kind': 'paired_update_lr_continuation_v1', 'smoke_only': False,
                'updates_per_arm': 16384, 'blocks': 8, 'block_size': 2048,
                'lr_high': 1e-4, 'lr_low': 1e-5}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('Historical LR control protocol differs')
    if (summary.get('arm') != 'constant_high' or summary.get('updates') != 16384
            or not summary.get('source_optimizer_preserved')
            or set(summary.get('initial_optimizer_steps', [])) != {98304.}
            or summary.get('final_optimizer_steps') != [114688.]):
        raise ValueError('Historical control exposure/optimizer differs')
    if len(protocol['sequence']) != 8 or any(len(block) != 2048 for block in protocol['sequence']):
        raise ValueError('Historical control batch plan differs')
    schedule = summary['schedule_state']
    if schedule['kind'] != 'constant_high' or schedule['completed_updates'] != 16384:
        raise ValueError('Historical control did not use constant high LR throughout')


def audit_lr_control(study, source, root, vendor):
    study, source, root, vendor = map(Path, (study, source, root, vendor))
    if not (study / 'COMPLETE').is_file() or not (study / 'wire_validation/COMPLETE').is_file():
        raise ValueError('Historical training and full wire evaluation must be complete')
    protocol = json.loads((study / 'protocol.json').read_text())
    summary = json.loads((study / 'constant_high/summary.json').read_text())
    validate_lr_control_metadata(protocol, summary)
    if digest(source) != protocol['source_sha256']:
        raise ValueError('Historical source checkpoint differs')
    for name, expected in protocol['code_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Historical implementation changed: {name}')
    for name, expected in protocol['vendor_sha256'].items():
        if digest(vendor / 'vqvae' / name) != expected:
            raise ValueError(f'Historical vendor changed: {name}')
    for name, expected in protocol['input_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Historical train/pilot input changed: {name}')
    if (digest(study / 'constant_high/selected.pt') != summary['selected_sha256']
            or digest(study / 'constant_high/last.pt') != summary['last_sha256']):
        raise ValueError('Historical retained control weights changed')
    saved = torch.load(study / 'constant_high/last.pt', map_location='cpu', weights_only=False)
    if ({float(state['step']) for state in saved['optimizer_state_dict']['state'].values()} != {114688.}
            or saved['schedule_protocol_sha256'] != digest(study / 'protocol.json')):
        raise ValueError('Historical retained optimizer/lineage differs')
    del saved
    wire = study / 'wire_validation'
    artifacts = {}
    for label in ('source', 'constant_high_selected', 'constant_high_endpoint'):
        result = json.loads((wire / f'{label}.json').read_text())
        folder = wire / label
        if (not result['independent_bundle_roundtrip']
                or (folder / 'validation.fsb').stat().st_size != result['complete_stream_bytes']
                or digest(folder / 'decoder_model_weights.pt') != result['model_sha256']):
            raise ValueError('Historical wire/model artifact integrity differs')
        artifacts[label] = {str(path.relative_to(wire)): digest(path) for path in
            (wire / f'{label}.json', folder / 'groups.csv', folder / 'validation.fsb', folder / 'decoder_model_weights.pt')}
    return {'study': str(study.resolve()), 'protocol_sha256': digest(study / 'protocol.json'),
            'wire_protocol_sha256': digest(wire / 'protocol.json'), 'wire_artifacts': artifacts,
            'selected_sha256': summary['selected_sha256'], 'endpoint_sha256': summary['last_sha256'],
            'control_retrained': False, 'control_reevaluated': False}
