#!/usr/bin/env python3
"""Frozen neural RD/physical stress on one explicitly external-development SMF group."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.affine_restoration import candidates, restore
from compression.numeric_restoration import encode_numeric
from finetuning.reversible_normalization import HEADER
from finetuning.affine_codec_study import score_restored, temporal_distortion, decode_affine_frame
from finetuning.affine_index_search import paired_search
from finetuning.diagnose_reconstruction import spectral_metrics
from finetuning.evaluate_checkpoints import metrics
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.physical_validation import ordinary_fidelity
from finetuning.representation_study import digest, load_packet, read_manifest, write_json
from finetuning.robust_escape import encode_robust
from finetuning.run_index_search import FrozenIndexStream, nearest, vectors
from finetuning.run_physical_guard_search import physical_admit


def pooled_score(raw, decoded, valid, window=1000):
    if raw.shape != decoded.shape or raw.shape != valid.shape:
        raise ValueError('Matching continuous waveform shapes required')
    assert raw[~valid].tobytes() == decoded[~valid].tobytes()
    mean, scale = float(raw[valid].astype(np.float64).mean()), float(raw[valid].astype(np.float64).std()) + 1e-6
    x, y = np.zeros_like(raw), np.zeros_like(decoded)
    x[valid] = ((raw[valid].astype(np.float64) - mean) / scale).astype(np.float32)
    y[valid] = ((decoded[valid].astype(np.float64) - mean) / scale).astype(np.float32)
    pairs = valid[:, 1:] & valid[:, :-1]
    difference_error = np.zeros(pairs.shape, dtype=np.float64)
    difference_error[pairs] = (decoded[:, 1:][pairs].astype(np.float64) - decoded[:, :-1][pairs].astype(np.float64)
                              - raw[:, 1:][pairs].astype(np.float64) + raw[:, :-1][pairs].astype(np.float64))
    seam = (np.arange(1, raw.shape[1]) % window) == 0
    result = {**metrics(x[valid], y[valid]), **spectral_metrics(x, y), **ordinary_fidelity(raw, decoded, valid),
        'ordinary_cells': int(valid.sum()), 'escaped_cells': int((~valid).sum())}
    for label, mask in [('seam', pairs & seam[None, :]), ('interior', pairs & ~seam[None, :])]:
        result[label + '_difference_error_rmse_raw'] = float(np.sqrt(np.mean(difference_error[mask] ** 2))) if mask.any() else None
        result[label + '_pairs'] = int(mask.sum())
    return finite_json(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--external-dir', type=Path, required=True)
    parser.add_argument('--train-samples-dir', type=Path, required=True)
    parser.add_argument('--compact', type=Path, required=True)
    parser.add_argument('--dense', type=Path, required=True)
    parser.add_argument('--dense-reference-protocol', type=Path, required=True)
    parser.add_argument('--train-calibration', type=Path, required=True)
    parser.add_argument('--search-calibration', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the local RTX 4060')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(2088)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert (args.external_dir / 'COMPLETE').exists()
    external_protocol = json.loads((args.external_dir / 'protocol.json').read_text())
    samples = json.loads((args.external_dir / 'samples.json').read_text())
    calibration = json.loads(args.train_calibration.read_text())
    hashes = {arm: digest(getattr(args, arm)) for arm in ('compact', 'dense')}
    assert hashes['compact'] == calibration['checkpoint_sha256']
    assert hashes['dense'] == json.loads(args.dense_reference_protocol.read_text())['candidates']['high_A']
    assert digest(args.train_samples_dir / 'samples_manifest.csv') == calibration['manifest_sha256']
    train_lookup = {r['sample_path']: r for r in read_manifest(args.train_samples_dir)['train']}
    train = [train_lookup[p] for p in calibration['train_table_windows']]
    lr = json.loads(args.search_calibration.read_text())['selected_lr']
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_external_smf_stress.py', 'finetuning/affine_codec_study.py', 'finetuning/affine_index_search.py',
            'compression/affine_restoration.py', 'compression/numeric_restoration.py',
            'finetuning/run_physical_guard_search.py', 'finetuning/run_index_search.py']
    write_json(out / 'protocol.json', {'kind': 'external_development_smf_frozen_v1', 'parameter_updates': 0,
        'gpu': torch.cuda.get_device_name(), 'source_checkpoint_sha256': hashes, 'train_table_windows': calibration['train_table_windows'],
        'external_protocol_sha256': digest(args.external_dir / 'protocol.json'),
        'sample_sha256': {r['path']: r['sha256'] for r in samples}, 'code_sha256': {p: digest(ROOT / p) for p in code},
        'coverage': '6 x continuous 1s x all 551 rows; 20 adjacent temporal frames x spatial 512+39; no amplitude removal',
        'calibration': 'both symbol tables regenerated only on identical existing 135 train windows, TF32 disabled; no external adaptation of tables',
        'arms': 'frozen compact/dense and strictly guarded per-frame affine restoration for each, no checkpoint selection',
        'search_subset': 'compact only, temporal index 10 and both spatial tiles per file; shared 48-step trajectory strict vs affine, fixed historical LR',
        'search_lr': lr, 'pooled_mask': 'exact union of transmitted per-frame ordinary/escape masks',
        'role': external_protocol['group_role'], 'reserved_EDAS_groups': external_protocol['signal_unseen_reserved_groups'],
        'numeric_contract': 'FNX1 raw uint32 escapes or FNZ1 zlib-6 delta-index + raw-bit escapes; FAX1/FAZ1 affine equivalents; no clipping, imputation or row removal; legacy finite input remains FNE1/FAS1',
        'limitations': ['one acquisition-attribute group, not final or four-family external validation',
            'full supplied zone differs from historically filtered coverage; no pure domain-shift attribution',
            'pooled 1s metrics differ from averaged 50ms-window metrics; both reported',
            'different rates compact/dense; no dominance claim at equal bytes']})
    for p in code:
        dest = out / 'code_snapshot' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / p, dest)
    maps = [{'region': 0, 'source_channel_indices': list(range(512))}, {'region': 1, 'source_channel_indices': list(range(512, 551))}]
    write_json(out / 'maps.json', maps)
    file_results, summaries, search_records, all_transport = [], {}, [], []
    for arm in ('compact', 'dense'):
        saved = torch.load(getattr(args, arm), map_location='cpu', weights_only=False)
        model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
        tensor_bytes = sum(v.numel() * v.element_size() for v in saved['model_state_dict'].values())
        del saved
        forward = model.decoder.forward
        model.decoder.forward = lambda x, forward=forward: checkpoint(forward, x, use_reentrant=False) if torch.is_grad_enabled() and x.requires_grad else forward(x)
        counts = np.zeros(2048, dtype=np.int64)
        with torch.no_grad():
            for row in train:
                _, packet = load_packet(row)
                target = torch.from_numpy(packet.values)[None, None].cuda()
                h, w = packet.values.shape
                z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
                indices = nearest(z, model.quantizer.embedding.weight)
                counts += np.bincount(indices.cpu().numpy(), minlength=2048)
        stream = FrozenIndexStream(counts, hashes[arm])
        folder = out / arm
        folder.mkdir()
        np.save(folder / 'counts.npy', counts)
        write_json(folder / 'descriptor.json', {'source_checkpoint_sha256': hashes[arm], 'counts_sha256': digest(folder / 'counts.npy'),
            'FNE1': 'original finite input-normalization restoration', 'FAS1': 'finite adaptive output offset/scale',
            'FNX1/FNZ1': 'exact uint32 nonfinite/extreme escapes, raw or delta+zlib-6',
            'FAX1/FAZ1': 'same exact escapes with adaptive output offset/scale', 'new_waveform_weight_bytes': 0})
        measured = {'direct': [], 'affine': []}
        frames = {'direct': 0, 'affine': 0}
        records = []
        for file_index, sample in enumerate(samples):
            assert digest(args.external_dir / sample['path']) == sample['sha256']
            with np.load(args.external_dir / sample['path'], allow_pickle=False) as data:
                raw_full = data['data']
            restored_full = {name: np.empty_like(raw_full) for name in measured}
            valid_full = np.zeros(raw_full.shape, dtype=bool)
            for window in range(20):
                for region, begin in enumerate((0, 512)):
                    end = min(begin + 512, 551)
                    raw = raw_full[begin:end, window * 1000:(window + 1) * 1000].copy()
                    packet = encode_numeric(raw)
                    row = {'family': 'single_mode', 'configuration': 'External SMF/' + sample['orientation'],
                           'sample_path': f'{sample["drive_id"]}/w{window:02d}/r{region}'}
                    target = torch.from_numpy(packet.values)[None, None].cuda()
                    h, w = raw.shape
                    with torch.no_grad():
                        z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
                        indices = nearest(z, model.quantizer.embedding.weight)
                        latent = tuple(z.shape[-2:])
                        pred = model.decoder(vectors(indices, model.quantizer.embedding.weight, latent))[..., :h, :w][0, 0].cpu().numpy()
                    source_frame = stream.encode(indices, raw.shape, latent, packet.side_stream)
                    direct = restore(pred, packet.side_stream)
                    before = score_restored(row, raw, packet, direct)
                    best = (temporal_distortion(direct, packet), source_frame, direct, before, 'original')
                    for fit in candidates(packet.values, pred, packet.valid, packet.side_stream):
                        decoded = restore(pred, fit['side'])
                        score = score_restored(row, raw, packet, decoded)
                        loss = temporal_distortion(decoded, packet)
                        if loss < best[0] and physical_admit(before, score):
                            framed = stream.encode(indices, raw.shape, latent, fit['side'])
                            assert len(framed) == len(source_frame)
                            best = (loss, framed, decoded, score, fit['kind'])
                    key = f'f{file_index:02d}_w{window:02d}_r{region}'
                    for policy, frame, decoded, score in [('direct', source_frame, direct, before), ('affine', best[1], best[2], best[3])]:
                        (folder / f'{key}.{policy}.frame').write_bytes(frame)
                        replay = decode_affine_frame(frame, stream, model, 'cuda')
                        np.testing.assert_array_equal(replay.view(np.uint32), decoded.view(np.uint32))
                        restored_full[policy][begin:end, window * 1000:(window + 1) * 1000] = decoded
                        measured[policy].append(score)
                        frames[policy] += len(frame)
                    valid_full[begin:end, window * 1000:(window + 1) * 1000] = packet.valid
                    records.append({'sample': row['sample_path'], 'file_index': file_index, 'window': window, 'region': region,
                        'direct': before, 'affine': best[3], 'restoration': best[4], 'direct_frame_bytes': len(source_frame), 'affine_frame_bytes': len(best[1]),
                        'nonfinite_cells': int((~np.isfinite(raw)).sum()), 'escaped_cells': int((~packet.valid).sum()),
                        'side_bytes': len(packet.side_stream), 'uncompressed_escape_side_bytes': HEADER.size + 8 * int((~packet.valid).sum())})
                    if arm == 'compact':
                        all_transport.append({'frame': key, 'drive_id': sample['drive_id'], 'window': window, 'region': region,
                            'clock_start_s': sample['clock_interval_s'][0] + window * .05})
                        if window == 10:
                            a, source_score, best_search, evaluated = paired_search(model, target, torch.from_numpy(packet.valid)[None, None].cuda(), row, raw, packet, stream, lr)
                            assert a == source_frame
                            search_record = {'sample': row['sample_path'], 'source': source_score, 'source_frame_bytes': len(a), 'evaluated': evaluated}
                            for policy, item in best_search.items():
                                frame = item.pop('frame')
                                (folder / f'{key}.search_{policy}.frame').write_bytes(frame)
                                search_record[policy] = {**item, 'frame_bytes': len(frame)}
                            search_records.append(search_record)
            for policy, decoded in restored_full.items():
                value = pooled_score(raw_full, decoded, valid_full)
                file_results.append({'arm': arm, 'policy': policy, 'drive_id': sample['drive_id'], 'orientation': sample['orientation'],
                    'stimulus_label_hz': sample['stimulus_label_hz'], 'cells': raw_full.size, 'pooled_1s': value})
                np.savez_compressed(folder / f'f{file_index:02d}.{policy}.reconstruction.npz', data=decoded)
            print(json.dumps({'arm': arm, 'files_completed': file_index + 1, 'total': len(samples)}), flush=True)
        write_json(folder / 'records.json', records)
        write_json(folder / 'file_pooled_results.json', [r for r in file_results if r['arm'] == arm])
        summaries[arm] = {'serialized_input_checkpoint_bytes': getattr(args, arm).stat().st_size, 'waveform_tensor_bytes': tensor_bytes, 'policies': {}}
        summaries[arm]['numeric_side_audit'] = {key: sum(r[key] for r in records) for key in
                                              ('nonfinite_cells', 'escaped_cells', 'side_bytes', 'uncompressed_escape_side_bytes')}
        for policy, values in measured.items():
            macro, groups = aggregate_metrics(values)
            summaries[arm]['policies'][policy] = {'window_macro': macro, 'window_groups': groups,
                'frame_bytes': frames[policy], 'affine_selected_windows': sum(r['restoration'] != 'original' for r in records) if policy == 'affine' else 0}
        del model, stream
        torch.cuda.empty_cache()
    write_json(out / 'transport.json', all_transport)
    write_json(out / 'search_subset_records.json', search_records)
    result = {'parameter_updates': 0, 'files': len(samples), 'window_tiles_per_model': 240, 'input_cells_per_model': 6 * 551 * 20000,
        'arms': summaries, 'pooled_1s_files': file_results, 'search_subset': {},
        'limitations': 'external DEVELOPMENT, one linked SMF group; no final holdout or four-family validation; no matched-rate superiority claim; pooled metrics use actual frame mask union'}
    common = (out / 'maps.json').stat().st_size + (out / 'transport.json').stat().st_size
    for arm, summary in summaries.items():
        auxiliary = common + (out / arm / 'counts.npy').stat().st_size + (out / arm / 'descriptor.json').stat().st_size
        for value in summary['policies'].values():
            value.update(stream_bytes=value['frame_bytes'] + auxiliary,
                bpc=8 * (value['frame_bytes'] + auxiliary) / result['input_cells_per_model'],
                ratio_float32=4 * result['input_cells_per_model'] / (value['frame_bytes'] + auxiliary))
    for policy in ('source', 'strict', 'affine'):
        values = [r['source'] if policy == 'source' else r[policy]['metrics'] for r in search_records]
        macro, groups = aggregate_metrics(values)
        result['search_subset'][policy] = {'macro': macro, 'groups': groups,
            'frame_bytes': sum(r['source_frame_bytes'] if policy == 'source' else r[policy]['frame_bytes'] for r in search_records)}
    for arm in ('compact', 'dense'):
        assert digest(getattr(args, arm)) == hashes[arm]
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()
    print(json.dumps({'external_development_files': len(samples), 'parameter_updates': 0,
        'window_rmse': {arm: {p: v['window_macro']['macro_relative_rmse'] for p, v in data['policies'].items()} for arm, data in summaries.items()}}), flush=True)


if __name__ == '__main__':
    main()
