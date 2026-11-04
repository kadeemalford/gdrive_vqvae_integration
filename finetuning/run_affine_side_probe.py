#!/usr/bin/env python3
"""Train-only mechanism check: affine restoration on already frozen source/E1 codes."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.affine_restoration import candidates, restore
from finetuning.affine_codec_study import score_restored, temporal_distortion, decode_affine_frame
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, load_packet, read_manifest, write_json
from finetuning.run_index_search import FrozenIndexStream, vectors
from finetuning.run_physical_guard_search import physical_admit
from finetuning.run_spatial_output_study import macro_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bounded-ols', action='store_true')
    parser.add_argument('--reference-probe', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paired = json.loads((args.paired_dir / 'protocol.json').read_text())
    source_hash = digest(args.source)
    assert paired['checkpoint_sha256']['source'] == source_hash
    assert digest(args.calibration_dir / 'counts.npy') == paired['counts_sha256']
    lookup = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    rows = [lookup[p] for p in paired['selection']['train']]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_affine_side_probe.py', 'finetuning/affine_codec_study.py',
            'compression/affine_restoration.py', 'finetuning/run_physical_guard_search.py',
            'finetuning/run_index_search.py']
    write_json(out / 'protocol.json', {'kind': 'train_affine_restoration_probe_v1', 'parameter_updates': 0,
        'gpu': torch.cuda.get_device_name(), 'source_sha256': source_hash,
        'paired_protocol_sha256': digest(args.paired_dir / 'protocol.json'),
        'selection': [r['sample_path'] for r in rows], 'waveform_splits': ['train'],
        'code_sha256': {p: digest(ROOT / p) for p in code},
        'fitting': 'per-window OLS gain+offset and unit-RMS+offset; original available at encoder; coefficients serialized as FAS1 output offset/scale',
        'bounded_ols': args.bounded_ols, 'reference_probe_sha256': digest(args.reference_probe / 'results.json') if args.reference_probe else None,
        'wire_contract': 'FAS1 side has same byte layout/size as FNE1 but distinct output-restoration semantics; descriptor declares both; no new weights or free scalars',
        'guard': 'per-window source real bytes, RMSE, correlation, p90, FFT magnitude, phase, abs(RMS ratio-1)',
        'selector': 'minimum mixed ordinary normalized distortion among eligible source/raw E1/affine candidates',
        'advance_rule': 'positive macro normalized distortion gain and nonzero admission on train permit one fixed fresh-development comparison; no validation-based tuning',
        'limitations': ['stored best E1 codes only; not all search candidates',
                        'per-window encoder fitting, not shared model training or global calibration',
                        'relative train RMSE is dominated by constant tiles; use macro normalized distortion for this mechanism check']})
    for p in code:
        dest = out / 'code_snapshot' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p, dest)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    del saved
    stream = FrozenIndexStream(np.load(args.calibration_dir / 'counts.npy', allow_pickle=False), source_hash)
    records, measured, losses = [], {n: [] for n in ('source', 'stored_search', 'strict_no_affine', 'strict_affine')}, {n: [] for n in ('source', 'strict_no_affine', 'strict_affine')}
    for i, row in enumerate(rows):
        assert row['split'] == 'train' and digest(ROOT / row['sample_path']) == paired['sample_sha256'][row['sample_path']]
        raw, packet = load_packet(row)
        decoded = {}
        for arm in ('source', 'search'):
            frame = (args.paired_dir / 'train' / f'{i:05d}.{arm}.frame').read_bytes()
            indices, shape, latent, side = stream.decode(frame)
            assert shape == raw.shape and side == packet.side_stream
            with torch.no_grad():
                prediction = model.decoder(vectors(torch.from_numpy(indices.astype(np.int64)).cuda(),
                    model.quantizer.embedding.weight, latent))[..., :shape[0], :shape[1]][0, 0].cpu().numpy()
            restored = restore(prediction, side)
            decoded[arm] = (frame, indices, shape, latent, prediction, restored,
                            score_restored(row, raw, packet, restored), temporal_distortion(restored, packet))
        baseline = decoded['source']
        before, budget = baseline[6], len(baseline[0])
        best, strict = (baseline[7], baseline[0], before, 'source'), (baseline[7], baseline[0], before, 'source')
        attempted = []
        for arm, (frame, indices, shape, latent, prediction, restored, metric, value) in decoded.items():
            if len(frame) <= budget and physical_admit(before, metric) and value < strict[0]:
                strict = (value, frame, metric, arm)
            if len(frame) <= budget and physical_admit(before, metric) and value < best[0]:
                best = (value, frame, metric, arm)
            for fit in candidates(packet.values, prediction, packet.valid, packet.side_stream,
                                  baseline_rms=before['reconstruction_rms_ratio'] if args.bounded_ols else None):
                frame_new = stream.encode(torch.from_numpy(indices.astype(np.int64)).cuda(), shape, latent, fit['side'])
                assert len(frame_new) == len(frame)
                after_array = restore(prediction, fit['side'])
                after = score_restored(row, raw, packet, after_array)
                loss = temporal_distortion(after_array, packet)
                eligible = len(frame_new) <= budget and physical_admit(before, after)
                attempted.append({'from': arm, 'fit': fit['kind'], 'gain': fit['gain'], 'offset': fit['offset'], 'admitted': bool(eligible)})
                if eligible and loss < best[0]:
                    best = (loss, frame_new, after, arm + '_' + fit['kind'])
        for name, item in [('strict_no_affine', strict), ('strict_affine', best)]:
            measured[name].append(item[2])
            losses[name].append((row['family'], row['configuration'], item[0]))
            (out / f'{i:05d}.{name}.frame').write_bytes(item[1])
            replay = decode_affine_frame(item[1], stream, model, 'cuda')
            assert score_restored(row, raw, packet, replay) == item[2]
        measured['source'].append(before)
        measured['stored_search'].append(decoded['search'][6])
        losses['source'].append((row['family'], row['configuration'], baseline[7]))
        records.append({'sample_path': row['sample_path'], 'source': before, 'strict_no_affine': strict[2], 'strict_affine': best[2],
            'chosen': best[3], 'frame_bytes': len(best[1]), 'source_frame_bytes': budget, 'attempts': attempted})
    write_json(out / 'records.json', records)
    result = {'parameter_updates': 0, 'windows': len(rows), 'arms': {}, 'macro_normalized_distortion': {k: macro_loss(v) for k, v in losses.items()}}
    for name, values in measured.items():
        macro, groups = aggregate_metrics(values)
        result['arms'][name] = {'macro': macro, 'groups': groups}
    result['affine_selected_windows'] = sum(r['chosen'] not in ('source', 'search') for r in records)
    result['distortion_gain_vs_strict_no_affine'] = 1 - result['macro_normalized_distortion']['strict_affine'] / result['macro_normalized_distortion']['strict_no_affine']
    result['advance_rule_met'] = result['affine_selected_windows'] > 0 and result['distortion_gain_vs_strict_no_affine'] > 0
    if args.reference_probe:
        reference = json.loads((args.reference_probe / 'results.json').read_text())
        old_protocol = json.loads((args.reference_probe / 'protocol.json').read_text())
        assert old_protocol['selection'] == [r['sample_path'] for r in rows]
        result['gain_vs_two_affine_proposals'] = 1 - result['macro_normalized_distortion']['strict_affine'] / reference['macro_normalized_distortion']['strict_affine']
        result['advance_rule_met'] = result['advance_rule_met'] and result['gain_vs_two_affine_proposals'] > 0
    assert digest(args.source) == source_hash
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}, indent=2))


if __name__ == '__main__':
    main()
