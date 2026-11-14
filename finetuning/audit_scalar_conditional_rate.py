#!/usr/bin/env python3
"""Train-file-only scalar entropy audit, frozen transforms and exact conditional streams."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.local_scale_stream import LocalScaleStream, symbol_codes, TILE, SCALE_BANK
from compression.scalar_stream import ScalarStream
from compression.study_bundle import write_bundle, read_bundle, metadata_archive, HEADER, LENGTH
from finetuning.representation_study import digest, read_manifest, pilot_rows, load_packet, write_json
from finetuning.run_spatial_output_study import split_train_files
from finetuning.scalar_rd import ScalarRDCodec
from finetuning.scalar_source_lineage import audit_scalar_history

SOURCES = ('scalar_rd_c4_lambda005_fold1_20261031', 'scalar_rd_warm_c16_lambda02_fold1_20261031')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090; no local model execution')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    out = args.output_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    splits = read_manifest(args.samples_dir)
    rows = pilot_rows(splits['train'], seed=2104)
    fit_indices, check_indices = split_train_files(rows, seed=2104)
    fit, checks = [rows[i] for i in fit_indices], [rows[i] for i in check_indices]
    if {r['drive_id'] for r in fit} & {r['drive_id'] for r in checks}:
        raise ValueError('Calibration and checks must have disjoint training file IDs')
    sources = {name: ROOT / 'results/experiments' / name / 'best.pt' for name in SOURCES}
    code = ['finetuning/audit_scalar_conditional_rate.py', 'compression/local_scale_stream.py',
            'compression/scalar_stream.py', 'compression/study_bundle.py', 'finetuning/scalar_rd.py',
            'finetuning/representation_study.py', 'finetuning/run_spatial_output_study.py',
            'finetuning/latent_geometry.py', 'finetuning/robust_escape.py', 'finetuning/reversible_normalization.py',
            'finetuning/scalar_source_lineage.py']
    history_audits = {name: audit_scalar_history(source.parent, ROOT) for name, source in sources.items()}
    protocol = {'kind': 'frozen_scalar_conditional_rate_audit_v2', 'seed': 2104,
        'historical_implementation_review': history_audits,
        'sources_sha256': {name: digest(path) for name, path in sources.items()},
        'calibration_train_only': [r['sample_path'] for r in fit],
        'check_train_only': [r['sample_path'] for r in checks],
        'input_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in fit + checks},
        'metadata_sha256': {name: digest(args.samples_dir / name) for name in ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')},
        'code_sha256': {name: digest(ROOT / name) for name in code},
        'vendor_sha256': {name: digest(args.repo_root / 'vqvae' / name) for name in ('train_vqvae.py', 'evaluate_vqvae.py')},
        'variants': ['learned_logistic', 'empirical_marginal', 'local_scale'],
        'tile': TILE, 'fixed_scale_bank': SCALE_BANK.tolist(), 'empirical_smoothing': 1,
        'local_side': 'One uncompressed byte per channel/tile, selects empirical channel prior or one of64 fixed logistic scales; side, all probability tables and framing charged',
        'comparison': 'Same rounded scalar coefficients, same one-stream range coder, same tile traversal and frame/bundle schema; no neural updates',
        'scope': 'Entropy coding diagnostic of the existing neural codec, not a new reconstruction architecture or hyperprior training',
        'advancement': 'Consider full development inference only if local-scale frames save>=10% vs empirical marginal and projected complete cohort saving>=10%; projections are not measured validation gains. No automatic hyperprior training.',
        'validation_signal_windows_read': 0, 'parameter_updates': 0, 'final_holdout_accessed': False,
        'limitations': ['Check file IDs are disjoint from calibration but historical neural weights saw this train corpus',
                       'Directory/file-ID differences do not prove acquisition independence',
                       'Conditional scale selection uses source coefficients at encoder; explicit scale IDs are sent, not a free oracle']}
    out.mkdir(exist_ok=False)
    write_json(out / 'protocol.json', protocol)
    for name in code:
        dest = out / 'code_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    all_results = {}
    validation_count = len(splits['val'])  # Metadata only; no validation NPZ opened.
    for name, source in sources.items():
        old_protocol = json.loads((source.parent / 'protocol.json').read_text())
        historical_result = json.loads((source.parent / 'results.json').read_text())
        if digest(source) != historical_result['best_checkpoint_sha256']:
            raise ValueError('Scalar checkpoint differs from retained historical result')
        if old_protocol['manifest_sha256'] != protocol['metadata_sha256']['samples_manifest.csv']:
            raise ValueError('Historical neural source trained on another corpus')
        if audit_scalar_history(source.parent, ROOT) != history_audits[name]:
            raise ValueError('Historical inference compatibility changed')
        saved = torch.load(source, map_location='cpu', weights_only=False)
        config = saved['config']
        model = ScalarRDCodec(base_channels=config['base_channels'], latent_channels=config['latent_channels'], geometry=config['geometry']).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model_state_dict']); del saved
        channels = config['latent_channels']
        counts = np.zeros((channels, 512), dtype=np.int64)
        with torch.no_grad():
            for row in fit:
                if row['split'] != 'train' or digest(ROOT / row['sample_path']) != protocol['input_sha256'][row['sample_path']]:
                    raise ValueError('Train-only calibration input changed')
                _, packet = load_packet(row)
                symbols = torch.round(model.analysis(torch.from_numpy(packet.values)[None, None].cuda()))[0].cpu().numpy()
                _, codes = symbol_codes(symbols)
                for channel in range(channels):
                    counts[channel] += np.bincount(codes[channel].reshape(-1), minlength=512)
        streams = {'learned_logistic': LocalScaleStream(ScalarStream.from_scales(model.scales.detach().cpu().numpy()).probabilities, channels),
                   'empirical_marginal': LocalScaleStream.from_empirical_counts(counts),
                   'local_scale': LocalScaleStream.from_empirical_counts(counts, adaptive=True)}
        directory = out / name; directory.mkdir()
        frame_paths, records, totals = {}, [], {variant: 0 for variant in streams}
        for variant in streams:
            (directory / variant).mkdir(); frame_paths[variant] = []
        symbol_hashes = []
        with torch.no_grad():
            for index, row in enumerate(checks):
                if row['split'] != 'train' or digest(ROOT / row['sample_path']) != protocol['input_sha256'][row['sample_path']]:
                    raise ValueError('Only unchanged training waveforms may enter checks')
                raw, packet = load_packet(row)
                target = torch.from_numpy(packet.values)[None, None].cuda()
                predicted, _, quantized = model(target)
                symbols, _ = symbol_codes(quantized[0].cpu().numpy())
                symbol_hashes.append(hashlib.sha256(symbols.astype('<i4').tobytes()).hexdigest())
                record = {'sample_path': row['sample_path'], 'drive_id': row['drive_id'],
                          'channel_map_id': row['channel_map_id'], 'cells': raw.size}
                for variant, stream in streams.items():
                    frame = stream.encode(symbols, raw.shape, packet.side_stream)
                    decoded, shape, side = stream.decode(frame)
                    np.testing.assert_array_equal(decoded, symbols)
                    if side != packet.side_stream or shape != raw.shape:
                        raise AssertionError('Shape/normalization side changed')
                    replay = model.synthesis(torch.from_numpy(decoded.copy())[None].cuda().float(), shape)
                    torch.testing.assert_close(replay, predicted, rtol=0, atol=0)
                    path = directory / variant / f'{index:05d}.lss'; path.write_bytes(frame)
                    frame_paths[variant].append(path); totals[variant] += len(frame)
                    record[variant + '_bytes'] = len(frame)
                records.append(record)
        results, projected = {}, {}
        common_records = [{key: value for key, value in record.items() if not key.endswith('_bytes')} for record in records]
        for variant, stream in streams.items():
            assets = {name: (args.samples_dir / name).read_bytes() for name in ('samples_manifest.csv', 'channel_maps.jsonl', 'preprocessing.json')}
            assets['prior.f32'] = stream.serialized
            assets['registry.json'] = json.dumps({'model_sha256': protocol['sources_sha256'][name],
                'channels': channels, 'adaptive': stream.adaptive, 'frames': common_records}, sort_keys=True).encode()
            bundle = directory / variant / 'checks.fsb'
            write_bundle(bundle, assets, frame_paths[variant])
            reader = read_bundle(bundle); recovered_assets = next(reader)
            prior = np.frombuffer(recovered_assets['prior.f32'], dtype='<f4').reshape(-1, 512)
            receiver = LocalScaleStream(prior, channels, stream.adaptive)
            recovered_hashes = []
            for frame in reader:
                recovered, _, _ = receiver.decode(frame)
                recovered_hashes.append(hashlib.sha256(recovered.astype('<i4').tobytes()).hexdigest())
            if recovered_hashes != symbol_hashes:
                raise AssertionError('Independent bundle replay differs')
            results[variant] = {'frame_bytes': totals[variant], 'prior_table_bytes': len(stream.serialized),
                'complete_check_bytes': bundle.stat().st_size, 'check_windows': len(checks),
                'shared_neural_tensor_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                'scalar_and_neural_replay_bit_exact': True, 'independent_bundle_replay': True}
            projected_assets = dict(assets)
            registry = json.loads(assets['registry.json'])
            registry['frames'] = [{'sample_path': r['sample_path'], 'drive_id': r['drive_id'],
                'channel_map_id': r['channel_map_id'], 'cells': int(r['channels']) * int(r['time_samples'])}
                for r in sorted(splits['val'], key=lambda row: row['sample_path'])]
            projected_assets['registry.json'] = json.dumps(registry, sort_keys=True).encode()
            # Exact metadata/length cost for the proposed cohort, only frame payloads extrapolated.
            projected[variant] = (HEADER.size + LENGTH.size * validation_count
                + len(metadata_archive(projected_assets)) + totals[variant] * validation_count / len(checks))
        baseline, local = results['empirical_marginal'], results['local_scale']
        payload_gain = 100 * (1 - local['frame_bytes'] / baseline['frame_bytes'])
        # A prespecified screening extrapolation, not actual validation transport.
        projected_gain = 100 * (1 - projected['local_scale'] / projected['empirical_marginal'])
        all_results[name] = {'variants': results, 'frame_gain_vs_empirical_pct': payload_gain,
            'complete_check_gain_vs_empirical_pct': 100 * (1 - local['complete_check_bytes'] / baseline['complete_check_bytes']),
            'projected_complete_validation_gain_pct_NOT_MEASURED': projected_gain,
            'full_development_inference_eligible': payload_gain >= 10 and projected_gain >= 10,
            'validation_signal_windows_read': 0, 'parameter_updates': 0, 'model_promoted': False}
        write_json(directory / 'records.json', records)
        write_json(directory / 'results.json', all_results[name])
        print(name, json.dumps(all_results[name]), flush=True)
        if digest(source) != protocol['sources_sha256'][name]:
            raise ValueError('Frozen scalar source changed')
        del model
        torch.cuda.empty_cache()
    write_json(out / 'results.json', all_results)
    if any(digest(ROOT / name) != expected for name, expected in protocol['code_sha256'].items()):
        raise ValueError('Frozen audit code changed')
    (out / 'COMPLETE').touch()
    lock.close()


if __name__ == '__main__':
    main()
