"""Explicit compatibility exceptions for frozen scalar inference, never for resumed training."""
import json
from pathlib import Path

from finetuning.representation_study import digest

# Inspected historical C4 differences: warm-start CLI/records and safe int32 bounds.
# The old training runner and old ScalarStream.encode are not called by this audit.
APPROVED_NONRUNTIME_PAIRS = {
    'finetuning/run_scalar_rd_study.py': (
        '16b4f30e8e54e5596b92031d53a4fcad5c8aa716bb16fa20b903a70a20bd0009',
        '23c14e2ff666b97eafa3b3d4c8fbe64109da210da067d8c3f3a68dc5679ef5ce'),
    'compression/scalar_stream.py': (
        '7e8820be585206ec2ca00290f9dfc4066f6697bf4cc6a88cc1f2631490ad6f35',
        '1f3344729c6a1423619e65837339e5b1e8f5fc330957d7a8649988286e46b7f5'),
}


def audit_scalar_history(source_directory, root):
    source_directory, root = Path(source_directory), Path(root)
    protocol = json.loads((source_directory / 'protocol.json').read_text())
    findings = {}
    for name, expected in protocol['code_sha256'].items():
        current = digest(root / name)
        if current == expected:
            findings[name] = {'status': 'identical', 'sha256': current}
            continue
        snapshot = source_directory / 'code_snapshot' / name
        if digest(snapshot) != expected or (expected, current) != APPROVED_NONRUNTIME_PAIRS.get(name):
            raise ValueError(f'Unreviewed historical scalar implementation change: {name}')
        findings[name] = {'status': 'explicitly_reviewed_nonruntime_change',
            'historical_sha256': expected, 'current_sha256': current, 'snapshot_verified': True,
            'scope': 'Frozen inference audit only; not approval to resume the historical training run'}
    return findings
