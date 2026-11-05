#!/usr/bin/env python3
"""Train-only frozen-codec PQ probe; advancement is gated before validation access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.alphabet_stream import AlphabetStream
from compression.product_quantization import quantize_product, replay_product, validate_transform
from finetuning.model_factory import build_model
from finetuning.paired_search_diagnostics import aggregate_metrics, finite_json
from finetuning.representation_study import digest, read_manifest, load_packet, score, distortion, write_json
from finetuning.run_alphabet_factorization import lloyd
from finetuning.run_index_search import nearest, vectors
from finetuning.run_physical_guard_search import physical_admit
from finetuning.run_spatial_output_study import macro_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rotation-probe', type=Path)
    parser.add_argument('--external-codebooks', type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one remote RTX 4090')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(2091)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = args.baseline_dir
    assert (base / 'COMPLETE').exists()
    previous = json.loads((base / 'protocol.json').read_text())
    assert not previous['smoke_only']
    assert digest(args.samples_dir / 'samples_manifest.csv') == previous['manifest_sha256']
    # Read only train arrays; validation entries in the manifest are not signals.
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    fit = [train[p] for p in previous['calibration_train']]
    check = [train[p] for p in previous['internal_train_check']]
    assert not {r['drive_id'] for r in fit} & {r['drive_id'] for r in check}
    source_dir = base / 'compact2048'
    source_path = source_dir / 'selected.pt'
    source_hash = digest(source_path)
    assert source_hash == json.loads((source_dir / 'descriptor.json').read_text())['checkpoint_sha256']
    saved = torch.load(source_path, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    assert saved['config']['embedding_dim'] == 128
    weight_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    del saved
    source_stream = AlphabetStream(np.load(source_dir / 'counts.npy', allow_pickle=False), source_hash)
    basis = center = covariance = None
    representation = 'product-vq-2-v1'
    if args.rotation_probe is not None:
        covariance = json.loads((args.rotation_probe / 'covariance_diagnostic.json').read_text())
        earlier = json.loads((args.rotation_probe / 'protocol.json').read_text())
        assert covariance['rotation_probe_allowed'] and covariance['source_sha256'] == source_hash
        assert earlier['fit_train'] == [r['sample_path'] for r in fit]
        assert earlier['check_train'] == [r['sample_path'] for r in check]
        assert digest(args.rotation_probe / 'balanced_pca_basis.npy') == covariance['basis_sha256']
        assert digest(args.rotation_probe / 'balanced_pca_center.npy') == covariance['center_sha256']
        basis = torch.from_numpy(np.load(args.rotation_probe / 'balanced_pca_basis.npy', allow_pickle=False)).cuda()
        center = torch.from_numpy(np.load(args.rotation_probe / 'balanced_pca_center.npy', allow_pickle=False)).cuda()
        validate_transform(basis, center, 128, model.quantizer.embedding.weight.device, model.quantizer.embedding.weight.dtype)
        representation = 'pca-product-vq-2-v1'
    external_selection = None
    arms = {'pq2x32': [2, 32, 64], 'pq2x64': [2, 64, 64]}
    if args.external_codebooks is not None:
        assert args.rotation_probe is not None
        assert (args.external_codebooks / 'CODEBOOKS_READY').exists()
        training = json.loads((args.external_codebooks / 'protocol.json').read_text())
        assert training['source_checkpoint_sha256'] == source_hash
        assert training['fit_train'] == [r['sample_path'] for r in fit]
        assert training['check_train'] == [r['sample_path'] for r in check]
        external_selection = json.loads((args.external_codebooks / 'selection.json').read_text())
        arms = {'latent': [2, 32, 64], 'waveform': [2, 32, 64]}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_product_quantization_probe.py', 'compression/product_quantization.py',
            'compression/alphabet_stream.py', 'finetuning/run_alphabet_factorization.py',
            'finetuning/model_factory.py', 'finetuning/run_physical_guard_search.py',
            'finetuning/representation_study.py', 'finetuning/run_spatial_output_study.py',
            'finetuning/analyze_product_covariance.py']
    write_json(out / 'protocol.json', {'kind': 'frozen_product_vq_train_probe_v2' if covariance else 'frozen_product_vq_train_probe_v1', 'seed': 2091,
        'neural_parameter_updates': 0, 'gpu': torch.cuda.get_device_name(), 'source_checkpoint_sha256': source_hash,
        'baseline_results_sha256': digest(base / 'results.json'), 'manifest_sha256': previous['manifest_sha256'],
        'code_sha256': {p: digest(ROOT / p) for p in code},
        'input_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in [*fit, *check]},
        'fit_train': [r['sample_path'] for r in fit], 'check_train': [r['sample_path'] for r in check],
        'arms': arms,
        'external_selection_sha256': digest(args.external_codebooks / 'selection.json') if external_selection else None,
        'external_training_protocol_sha256': digest(args.external_codebooks / 'protocol.json') if external_selection else None,
        'initialization': 'balanced-PCA halves' if covariance else 'fixed contiguous halves',
        'centroid_rule': 'externally selected paired centroid-only optimization' if external_selection else '<=256 vectors per fit tile; farthest-first +12 Lloyd steps each subspace',
        'rotation': covariance,
        'rotation_rule': 'only the previously gated orthogonal PCA basis; no OPQ, whitening, rotation/LR sweep or extra samples',
        'tables': 'one pooled marginal per alphabet fitted on fit files; same fixed/range fallback as baseline',
        'wire': 'FAK1 symbol-grid width = latent-grid width * groups; symbols interleaved per position; descriptor binds source/codebooks; one route byte in both source and fallback',
        'guard': 'same per-window bytes, RMSE/correlation/p90/FFT magnitude/phase/RMS-distance plus strictly lower normalized D; no relaxation',
        'advance_gate': '>=1% lower train-check guarded macro normalized D AND nonincreasing routed total stream including source+candidate dictionaries/descriptor; source is always fallback',
        'codebook_cost': 'explicit additional shared float32 codebooks, NOT free; source weights retained for fallback; report raw bytes and amortization separately from stream',
        'extension': 'validation not accessed by this runner; eligible positive requires a separately fixed confirmation, no automatic tuning',
        'limitations': ['all checks are train files seen in historical source training, not independent acquisitions',
                        'constant tiles make relative RMSE macro ill-conditioned; advancement uses normalized D plus unchanged physical guards',
                        'centroid fitting is training on fit data, but encoder/decoder/codebook of source remain unchanged',
                        'no EDAS/SMF/test access or rate-equivalent continuous claim']})
    for f in code:
        dest = out / 'code_snapshot' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)
    source_folder = out / 'source'
    source_folder.mkdir()
    shutil.copy2(source_dir / 'counts.npy', source_folder / 'counts.npy')
    shutil.copy2(source_dir / 'descriptor.json', source_folder / 'descriptor.json')
    transport = [{key: row[key] for key in ('sample_path', 'drive_id', 'family', 'configuration', 'spatial_region_index')} for row in check]
    write_json(out / 'transport.json', transport)

    def encode_latent(row):
        raw, packet = load_packet(row)
        target = torch.from_numpy(packet.values)[None, None].cuda()
        valid = torch.from_numpy(packet.valid)[None, None].cuda()
        h, w = raw.shape
        z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
        return raw, packet, target, valid, z

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = {'neural_parameter_updates': 0, 'validation_signal_windows_read': 0,
              'source_waveform_tensor_bytes': weight_bytes, 'arms': {},
              'external_centroid_parameter_updates': sum(v['updates'] for v in external_selection.values()) if external_selection else 0,
              'neural_updates_note': 'zero applies to the frozen SOURCE network; externally trained product centroids are counted separately'}
    with torch.no_grad():
        pool = []
        for row in fit:
            _, _, _, _, z = encode_latent(row)
            flat = z.permute(0, 2, 3, 1).reshape(-1, 128)
            positions = torch.linspace(0, len(flat) - 1, min(256, len(flat)), device='cuda').long()
            pool.append(flat[positions].clone())
        pool = torch.cat(pool)
        fit_pool = (pool - center) @ basis if basis is not None else pool
        cached, source_losses = [], []
        source_bytes = 0
        for i, row in enumerate(check):
            raw, packet, target, valid, z = encode_latent(row)
            indices = nearest(z, model.quantizer.embedding.weight)
            latent = tuple(z.shape[-2:])
            prediction = model.decoder(vectors(indices, model.quantizer.embedding.weight, latent))[..., :raw.shape[0], :raw.shape[1]]
            metric = finite_json(score(row, raw, packet, prediction[0, 0].cpu().numpy()))
            loss = float(distortion(prediction, target, valid))
            frame = source_stream.encode(indices.cpu().numpy(), raw.shape, latent, packet.side_stream)
            source_folder.joinpath(f'{i:05d}.fak').write_bytes(frame)
            source_folder.joinpath(f'{i:05d}.route').write_bytes(b'\x00' + frame)
            cached.append({'metrics': metric, 'loss': loss, 'frame': frame})
            source_losses.append((row['family'], row['configuration'], loss))
            source_bytes += len(frame) + 1
        source_distortion = macro_loss(source_losses)
        source_aux = sum((source_folder / f).stat().st_size for f in ('counts.npy', 'descriptor.json'))
        source_total = source_bytes + source_aux + (out / 'transport.json').stat().st_size
        result['source'] = {'macro_normalized_distortion': source_distortion, 'routed_frame_bytes': source_bytes,
                            'stream_bytes': source_total, 'check_windows': len(check)}
        for arm, (_, alphabet, _) in arms.items():
            folder = out / arm
            folder.mkdir()
            if external_selection:
                selected_path = args.external_codebooks / arm / 'selected.npy'
                assert digest(selected_path) == external_selection[arm]['selected_sha256']
                books = torch.from_numpy(np.load(selected_path, allow_pickle=False)).cuda()
                assert list(books.shape) == [2, alphabet, 64]
            else:
                books = torch.stack([lloyd(fit_pool[:, g * 64:(g + 1) * 64], alphabet, seed=2091) for g in range(2)])
            np.save(folder / 'codebooks.npy', books.cpu().numpy())
            book_hash = digest(folder / 'codebooks.npy')
            asset_fingerprint = bytes.fromhex(source_hash) + bytes.fromhex(book_hash) + representation.encode()
            if covariance:
                shutil.copy2(args.rotation_probe / 'balanced_pca_basis.npy', folder / 'basis.npy')
                shutil.copy2(args.rotation_probe / 'balanced_pca_center.npy', folder / 'center.npy')
                asset_fingerprint += bytes.fromhex(covariance['basis_sha256']) + bytes.fromhex(covariance['center_sha256'])
            binding = hashlib.sha256(asset_fingerprint).hexdigest()
            counts = np.zeros(alphabet, dtype=np.int64)
            for row in fit:
                _, _, _, _, z = encode_latent(row)
                symbols, _ = quantize_product(z, books, basis, center)
                counts += np.bincount(symbols.cpu().numpy().reshape(-1), minlength=alphabet)
            np.save(folder / 'counts.npy', counts)
            stream = AlphabetStream(counts, binding)
            transform_bytes = basis.numel() * basis.element_size() + center.numel() * center.element_size() if covariance else 0
            descriptor = {'format': 'FAK1', 'representation': representation, 'groups': 2,
                'source_checkpoint_sha256': source_hash, 'codebooks_sha256': book_hash,
                'counts_sha256': digest(folder / 'counts.npy'), 'binding_sha256': binding,
                'identifier': stream.identifier.hex(), 'alphabet': alphabet,
                'symbol_layout': 'position-major interleaved subspaces, symbol-grid width = latent width * groups',
                'additional_shared_codebook_tensor_bytes': books.numel() * books.element_size(),
                'additional_shared_transform_tensor_bytes': transform_bytes}
            if covariance:
                descriptor.update({'basis_sha256': covariance['basis_sha256'], 'center_sha256': covariance['center_sha256']})
            write_json(folder / 'descriptor.json', descriptor)
            losses, guarded_losses, records, measured = [], [], [], []
            candidate_bytes = guarded_bytes = accepted = 0
            quant_error = latent_power = 0.
            for i, row in enumerate(check):
                raw, packet, target, valid, z = encode_latent(row)
                symbols, q = quantize_product(z, books, basis, center)
                prediction = model.decoder(q)[..., :raw.shape[0], :raw.shape[1]]
                loss = float(distortion(prediction, target, valid))
                metric = finite_json(score(row, raw, packet, prediction[0, 0].cpu().numpy()))
                grid = tuple(z.shape[-2:])
                frame = stream.encode(symbols.cpu().numpy().reshape(-1), raw.shape, (grid[0], grid[1] * 2), packet.side_stream)
                recovered, shape, symbol_grid, side = stream.decode(frame)
                assert shape == raw.shape and side == packet.side_stream and symbol_grid == (grid[0], grid[1] * 2)
                replay = replay_product(torch.from_numpy(recovered.astype(np.int64)).cuda().view(-1, 2), grid, books, basis, center)
                torch.testing.assert_close(replay, q, rtol=0, atol=0)
                replayed_prediction = model.decoder(replay)[..., :raw.shape[0], :raw.shape[1]]
                torch.testing.assert_close(prediction, replayed_prediction, rtol=0, atol=0)
                before = cached[i]
                admitted = len(frame) <= len(before['frame']) and physical_admit(before['metrics'], metric) and loss < before['loss']
                chosen = frame if admitted else before['frame']
                losses.append((row['family'], row['configuration'], loss))
                guarded_losses.append((row['family'], row['configuration'], loss if admitted else before['loss']))
                candidate_bytes += len(frame)
                guarded_bytes += len(chosen) + 1
                accepted += admitted
                quant_error += float((q.double() - z.double()).square().sum())
                latent_power += float(z.double().square().sum())
                folder.joinpath(f'{i:05d}.candidate.fak').write_bytes(frame)
                folder.joinpath(f'{i:05d}.guarded.route').write_bytes(bytes([int(admitted)]) + chosen)
                records.append({'sample_path': row['sample_path'], 'source': before['metrics'], 'candidate': metric,
                    'source_normalized_distortion': before['loss'], 'candidate_normalized_distortion': loss,
                    'guarded_normalized_distortion': loss if admitted else before['loss'], 'admitted': bool(admitted),
                    'source_frame_bytes': len(before['frame']), 'candidate_frame_bytes': len(frame),
                    'latent': grid, 'symbols': int(symbols.numel()), 'nominal_symbol_bits': int(symbols.numel()) * stream.bits})
                measured.append(metric)
            macro, groups = aggregate_metrics(measured)
            aux = sum((folder / f).stat().st_size for f in ('counts.npy', 'descriptor.json'))
            total = guarded_bytes + aux + source_aux + (out / 'transport.json').stat().st_size
            guarded_distortion = macro_loss(guarded_losses)
            gain = 1 - guarded_distortion / source_distortion
            result['arms'][arm] = {'macro_normalized_distortion': macro_loss(losses),
                'guarded_macro_normalized_distortion': guarded_distortion, 'guarded_distortion_gain': gain,
                'admitted_windows': int(accepted), 'candidate_frame_bytes': candidate_bytes,
                'candidate_stream_bytes': candidate_bytes + aux + (out / 'transport.json').stat().st_size,
                'guarded_routed_frame_bytes': guarded_bytes, 'guarded_stream_bytes': total,
                'shared_codebook_tensor_bytes': books.numel() * books.element_size(),
                'shared_codebook_file_bytes': (folder / 'codebooks.npy').stat().st_size,
                'shared_transform_tensor_bytes': transform_bytes,
                'shared_transform_file_bytes': sum((folder / f).stat().st_size for f in ('basis.npy', 'center.npy')) if covariance else 0,
                'source_model_retained_for_fallback': True, 'relative_latent_quantization_mse': quant_error / max(latent_power, 1e-30),
                'macro_diagnostic_only': macro, 'groups_diagnostic_only': groups,
                'advance_gate_met': gain >= .01 and total <= source_total}
            write_json(folder / 'records.json', records)
            print(json.dumps({arm: {k: result['arms'][arm][k] for k in ('macro_normalized_distortion', 'guarded_distortion_gain', 'admitted_windows', 'candidate_frame_bytes', 'guarded_stream_bytes', 'advance_gate_met')}}), flush=True)
    assert digest(source_path) == source_hash
    torch.cuda.synchronize()
    result['combined_seconds'] = time.perf_counter() - start
    write_json(out / 'results.json', finite_json(result))
    (out / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
