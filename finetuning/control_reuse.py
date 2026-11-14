"""Conservative ledger checks before reusing a completed FEBUS-only bridge control."""
import json
from pathlib import Path

import torch

from finetuning.representation_study import digest


def validate_control_metadata(protocol, summary, history, *, allow_smoke=False):
    expected = {'kind': 'raw_phase_pretraining_bridge_v1', 'seed': 2096,
                'source_updates': 16384, 'final_febus_updates': 8192,
                'block_size': 2048, 'learning_rate': 1e-4}
    if protocol.get('smoke_only'):
        if not allow_smoke:
            raise ValueError('A smoke control cannot substitute for a scientific control')
        expected.update(source_updates=4, final_febus_updates=2, block_size=1)
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('Control optimizer/dose/protocol differs')
    if (len(protocol['source_febus_plan']) != expected['source_updates']
            or len(protocol['final_febus_plan']) != expected['final_febus_updates']):
        raise ValueError('Control training sequence length differs')
    if (summary.get('source_updates') != expected['source_updates']
            or summary.get('final_febus_updates') != expected['final_febus_updates']
            or summary.get('final_optimizer_steps') != [float(expected['final_febus_updates'])]):
        raise ValueError('Control stage reset or completed dose differs')
    size = expected['block_size']
    for phase, budget in [('source', expected['source_updates']), ('febus', expected['final_febus_updates'])]:
        completed = [r['updates'] for r in history if r['phase'] == phase]
        if completed != list(range(size, budget + 1, size)):
            raise ValueError('Control history has missing/duplicate blocks')


def audit_reusable_febus_control(study, root, vendor, *, allow_smoke=False):
    study, root, vendor = Path(study), Path(root), Path(vendor)
    if not (study / 'COMPLETE').is_file():
        raise ValueError('Complete training AND wire evaluation required before control reuse')
    protocol = json.loads((study / 'protocol.json').read_text())
    folder = study / 'febus'
    summary = json.loads((folder / 'summary.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    validate_control_metadata(protocol, summary, history, allow_smoke=allow_smoke)
    if digest(study / 'initial.pt') != protocol['source_sha256']:
        raise ValueError('Original compact initialization changed')
    if digest(folder / 'selected.pt') != summary['selected_sha256'] or digest(folder / 'last.pt') != summary['endpoint_sha256']:
        raise ValueError('Retained control checkpoint changed')
    for name, expected in protocol['code_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Control dependency changed: {name}')
    for name, expected in protocol['vendor_sha256'].items():
        if digest(vendor / 'vqvae' / name) != expected:
            raise ValueError(f'Control vendor implementation changed: {name}')
    for name, expected in protocol['input_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Control target input changed: {name}')
    saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    expected = int(protocol['final_febus_updates'])
    if saved['phase'] != 'febus' or saved['phase_updates'] != expected:
        raise ValueError('Control retained wrong stage')
    steps = {float(value['step']) for value in saved['optimizer_state_dict']['state'].values()}
    if steps != {float(expected)}:
        raise ValueError('Control AdamW counter audit failed')
    del saved
    wire_artifacts = {}
    for label in ('initial', 'febus_selected', 'febus_endpoint'):
        result_path = study / 'wire_validation' / f'{label}.json'
        directory = study / 'wire_validation' / label
        result = json.loads(result_path.read_text())
        if (not result['independent_bundle_roundtrip']
                or (directory / 'validation.fsb').stat().st_size != result['complete_stream_bytes']
                or digest(directory / 'decoder_model_weights.pt') != result['model_sha256']):
            raise ValueError('Retained control wire/model artifacts differ')
        wire_artifacts[label] = {'result_sha256': digest(result_path),
            'groups_sha256': digest(directory / 'groups.csv'),
            'bundle_sha256': digest(directory / 'validation.fsb'),
            'deployment_model_sha256': result['model_sha256']}
    return {'control_study': str(study.resolve()), 'protocol_sha256': digest(study / 'protocol.json'),
            'initial_sha256': protocol['source_sha256'],
            'selected_sha256': summary['selected_sha256'], 'endpoint_sha256': summary['endpoint_sha256'],
            'history_sha256': digest(folder / 'history.json'),
            'wire_results_sha256': digest(study / 'wire_validation/results.json'),
            'wire_protocol_sha256': digest(study / 'wire_validation/protocol.json'),
            'wire_artifacts': wire_artifacts,
            'final_optimizer_steps': sorted(steps), 'smoke_only': protocol['smoke_only'],
            'reuse_scope': 'Same original weights/target corpus/batches/optimizer reset/objective/precision; matched updates, not FLOPs. Public-source arm must independently match these settings.'}
