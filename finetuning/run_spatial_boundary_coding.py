#!/usr/bin/env python3
"""Frozen spatial orientation/padding coding probes under source frame ceilings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.spatial_correction_stream import pack_frame, decode_frame
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json, guarded_admit
from finetuning.representation_study import digest, distortion, load_packet, read_manifest, score, write_json
from finetuning.reversible_normalization import decode
from finetuning.run_index_search import FrozenIndexStream, nearest, vectors


class ReverseSpatial(torch.nn.Module):
    def forward(self, decoded):
        return decoded.flip(2)


def pad_spatial_replication(target, multiple=16):
    h, w = target.shape[-2:]
    target = F.pad(target, (0, 0, 0, (-h) % multiple), mode='replicate')
    return F.pad(target, (0, (-w) % multiple, 0, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('This campaign requires the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paired = json.loads((args.paired_dir / 'protocol.json').read_text())
    assert (args.paired_dir / 'COMPLETE').exists()
    source_hash = digest(args.source)
    assert source_hash == paired['checkpoint_sha256']['source']
    assert digest(args.samples_dir / 'samples_manifest.csv') == paired['manifest_sha256']
    assert digest(args.calibration_dir / 'counts.npy') == paired['counts_sha256']
    lookup = {row['sample_path']: row for rows in read_manifest(args.samples_dir).values() for row in rows}
    selected = {split: [lookup[p] for p in paths] for split, paths in paired['selection'].items()}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code_paths = ['finetuning/run_spatial_boundary_coding.py', 'compression/spatial_correction_stream.py',
                  'finetuning/paired_search_diagnostics.py', 'finetuning/run_index_search.py',
                  'finetuning/representation_study.py', 'finetuning/model_factory.py']
    protocol = {'kind': 'frozen_spatial_boundary_coding_v1', 'parameter_updates': 0,
        'gpu': torch.cuda.get_device_name(), 'device': 'cuda', 'math_policy': 'FP32, TF32 disabled, deterministic cuDNN',
        'source_sha256': source_hash, 'paired_protocol_sha256': digest(args.paired_dir / 'protocol.json'),
        'code_sha256': {p: digest(ROOT / p) for p in code_paths}, 'selection': paired['selection'],
        'hypotheses': ['reverse spatial order before source encoder, undo after exact source decoder',
                       'replicate only extra spatial input rows to the existing multiple of 16; temporal extra columns remain zero'],
        'selection_policy': 'fixed two alternatives plus source; choose lowest ordinary normalized 0.5MSE+0.5L1 among candidates satisfying source bytes/RMSE/correlation/p90 guards',
        'guard_undefined_policy': 'retain original source frame if a guarded metric is undefined',
        'wire_contract': 'same symbol count/grid/side; FSC1 mode 0=direct and 1=reverse output; 13 bytes per envelope in every policy including source; no learned correction weights',
        'limitations': ['joint change of encoder input and spatial ordering; not a proof of padding causality',
                        'known development files only; no external acquisitions or test access',
                        'replication influences only shapes requiring extra spatial input rows; internal convolutions unchanged'],
        'investment_gate': '>=5% guarded RMSE reduction vs source at nonincreasing common-envelope frame bytes; confirmation required before promotion'}
    write_json(out / 'protocol.json', protocol)
    for path in code_paths:
        target = out / 'code_snapshot' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    model = build_model({**saved['config'], 'exact_eval_quantization': True}, saved, 'cuda').eval().requires_grad_(False)
    del saved
    stream = FrozenIndexStream(np.load(args.calibration_dir / 'counts.npy', allow_pickle=False), source_hash)
    reverse = ReverseSpatial().cuda().eval()
    results = {'parameter_updates': 0, 'splits': {}}
    with torch.no_grad():
        for split, rows in selected.items():
            folder = out / split
            folder.mkdir()
            measured, records = defaultdict(list), []
            for index, row in enumerate(rows):
                assert row['split'] == split
                assert digest(ROOT / row['sample_path']) == paired['sample_sha256'][row['sample_path']]
                raw, packet = load_packet(row)
                h, w = raw.shape
                target = torch.from_numpy(packet.values)[None, None].cuda()
                valid = torch.from_numpy(packet.valid)[None, None].cuda()
                source_payload = (args.paired_dir / split / f'{index:05d}.source.frame').read_bytes()
                original, shape, latent, side = stream.decode(source_payload)
                assert shape == raw.shape and side == packet.side_stream
                source_prediction = model.decoder(vectors(torch.from_numpy(original.astype(np.int64)).cuda(),
                    model.quantizer.embedding.weight, latent))[..., :h, :w]
                before = finite_json(score(row, raw, packet, source_prediction[0, 0].cpu().numpy()))
                best_prediction = source_prediction
                best_loss = float(distortion(source_prediction, target, valid))
                best_payload, best_mode, best_name = source_payload, False, 'source'
                record = {'sample_path': row['sample_path'], 'drive_id': row['drive_id'],
                          'source': before, 'source_payload_bytes': len(source_payload), 'alternatives': {}}
                measured['source'].append(before)
                for name in ('reverse', 'replicate'):
                    mode = name == 'reverse'
                    if name == 'replicate' and h % 16 == 0:
                        prediction, payload = source_prediction, source_payload
                    else:
                        values = target.flip(2) if mode else target
                        padded = (F.pad(values, (0, (-w) % 16, 0, (-h) % 16)) if mode else pad_spatial_replication(values))
                        z = model.encoder(padded)
                        indices = nearest(z, model.quantizer.embedding.weight)
                        assert tuple(z.shape[-2:]) == tuple(latent)
                        prediction = model.decoder(vectors(indices, model.quantizer.embedding.weight, latent))[..., :h, :w]
                        if mode:
                            prediction = prediction.flip(2)
                        payload = stream.encode(indices, raw.shape, latent, side)
                    metric = finite_json(score(row, raw, packet, prediction[0, 0].cpu().numpy()))
                    eligible = len(payload) <= len(source_payload) and guarded_admit(before, metric)
                    measured[name].append(metric)
                    measured[name + '_guarded'].append(metric if eligible else before)
                    value = float(distortion(prediction, target, valid))
                    record['alternatives'][name] = {'metrics': metric, 'admitted': bool(eligible),
                                                     'payload_bytes': len(payload), 'distortion': value}
                    framed = pack_frame(payload if eligible else source_payload, mode if eligible else False)
                    (folder / f'{index:05d}.{name}.fsc').write_bytes(framed)
                    replay = decode_frame(framed, stream, model, reverse, 'cuda')
                    expected = decode((prediction if eligible else source_prediction)[0, 0].cpu().numpy(), side)
                    np.testing.assert_array_equal(replay, expected)
                    if eligible and value < best_loss:
                        best_loss, best_prediction = value, prediction
                        best_payload, best_mode, best_name = payload, mode, name
                record['best_name'] = best_name
                record['best'] = finite_json(score(row, raw, packet, best_prediction[0, 0].cpu().numpy()))
                record['best_frame_bytes'] = len(pack_frame(best_payload, best_mode))
                measured['best_guarded'].append(record['best'])
                (folder / f'{index:05d}.best.fsc').write_bytes(pack_frame(best_payload, best_mode))
                (folder / f'{index:05d}.source.fsc').write_bytes(pack_frame(source_payload, False))
                records.append(record)
            write_json(folder / 'records.json', records)
            arms = {}
            for name, metrics in measured.items():
                macro, groups = aggregate_metrics(metrics)
                arms[name] = {'macro': macro, 'groups': groups}
                suffix = {'source': 'source', 'reverse_guarded': 'reverse', 'replicate_guarded': 'replicate', 'best_guarded': 'best'}.get(name)
                if suffix:
                    arms[name]['frame_bytes'] = sum(p.stat().st_size for p in folder.glob(f'*.{suffix}.fsc'))
            for name in arms:
                arms[name]['rmse_gain_vs_source'] = 1 - arms[name]['macro']['macro_relative_rmse'] / arms['source']['macro']['macro_relative_rmse']
            results['splits'][split] = {'windows': len(rows), 'arms': arms,
                'chosen_modes': {name: sum(r['best_name'] == name for r in records) for name in ('source', 'reverse', 'replicate')}}
            print(json.dumps({'split': split, 'gain': arms['best_guarded']['rmse_gain_vs_source'],
                'bytes': arms['best_guarded']['frame_bytes'], 'source_bytes': arms['source']['frame_bytes']}), flush=True)
    np.save(out / 'counts.npy', np.load(args.calibration_dir / 'counts.npy', allow_pickle=False))
    maps = [json.loads(line) for line in (args.samples_dir / 'channel_maps.jsonl').read_text().splitlines()]
    ids = {row['channel_map_id'] for row in selected['val']}
    write_json(out / 'maps.json', [m for m in maps if m['channel_map_id'] in ids])
    write_json(out / 'transport.json', [{'frame_index': i, 'sample_path': r['sample_path'], 'drive_id': r['drive_id'],
        'block_index': int(r['block_index']), 'time_start': int(r['time_start']), 'channel_map_id': r['channel_map_id']}
        for i, r in enumerate(selected['val'])])
    write_json(out / 'registry.json', {'format': 'FSC1', 'mode_zero': 'source_decoder', 'mode_one': 'reverse_spatial_output',
        'source_sha256': source_hash, 'builtin_transform_code_sha256': digest(Path(__file__)), 'learned_transform_weights_bytes': 0})
    common = sum((out / p).stat().st_size for p in ('counts.npy', 'maps.json', 'transport.json', 'registry.json'))
    for arm in ('source', 'reverse_guarded', 'replicate_guarded', 'best_guarded'):
        results['splits']['val']['arms'][arm]['stream_bytes'] = results['splits']['val']['arms'][arm]['frame_bytes'] + common
    results['common_auxiliary_bytes'] = common
    results['base_model_weights_shared_excluded'] = True
    results['no_new_model_weights'] = True
    assert digest(args.source) == source_hash
    results['investment_gate_met'] = (results['splits']['val']['arms']['best_guarded']['rmse_gain_vs_source'] >= .05)
    write_json(out / 'results.json', finite_json(results))
    (out / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
