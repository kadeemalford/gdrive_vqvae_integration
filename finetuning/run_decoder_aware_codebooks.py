#!/usr/bin/env python3
"""Bounded paired PQ-centroid optimization, with all source network weights frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.product_quantization import quantize_product, validate_transform
from finetuning.model_factory import build_model
from finetuning.representation_study import digest, read_manifest, load_packet, distortion, write_json
from finetuning.run_spatial_output_study import macro_loss


def source_state_hash(model):
    checksum = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        checksum.update(key.encode())
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def centroid_objective(z, books, basis, center, decoder, target, valid, mode):
    _, q = quantize_product(z.detach(), books, basis, center)
    if mode == 'latent':
        return F.mse_loss(q, z.detach())
    if mode != 'waveform':
        raise ValueError('Unknown centroid objective')
    predicted = decoder(q)[..., :target.shape[-2], :target.shape[-1]]
    return distortion(predicted, target, valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--pca-probe', type=Path, required=True)
    parser.add_argument('--rotation-probe', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one remote RTX 4090')
    sys.path.insert(0, str(args.repo_root.resolve()))
    torch.set_num_threads(4)
    torch.manual_seed(2092)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert (args.pca_probe / 'COMPLETE').exists()
    prior = json.loads((args.pca_probe / 'protocol.json').read_text())
    assert prior['rotation']['rotation_probe_allowed']
    assert digest(args.samples_dir / 'samples_manifest.csv') == prior['manifest_sha256']
    train = {r['sample_path']: r for r in read_manifest(args.samples_dir)['train']}
    fit = [train[p] for p in prior['fit_train']]
    checks = [train[p] for p in prior['check_train']]
    assert not {r['drive_id'] for r in fit} & {r['drive_id'] for r in checks}
    source = args.baseline_dir / 'compact2048/selected.pt'
    source_hash = digest(source)
    assert source_hash == prior['source_checkpoint_sha256']
    saved = torch.load(source, map_location='cpu', weights_only=False)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    del saved
    before_state_hash = source_state_hash(model)
    initial_dir = args.pca_probe / 'pq2x32'
    descriptor = json.loads((initial_dir / 'descriptor.json').read_text())
    for f, key in [('codebooks.npy', 'codebooks_sha256'), ('basis.npy', 'basis_sha256'), ('center.npy', 'center_sha256')]:
        assert digest(initial_dir / f) == descriptor[key]
    initial = torch.from_numpy(np.load(initial_dir / 'codebooks.npy', allow_pickle=False)).cuda()
    assert tuple(initial.shape) == (2, 32, 64)
    basis = torch.from_numpy(np.load(initial_dir / 'basis.npy', allow_pickle=False)).cuda()
    center = torch.from_numpy(np.load(initial_dir / 'center.npy', allow_pickle=False)).cuda()
    validate_transform(basis, center, 128, initial.device, initial.dtype)
    budget = 2 if args.smoke else 256
    rng = random.Random(2092)
    sequence = [fit[rng.randrange(len(fit))] for _ in range(budget)]
    used_checks = checks
    if args.smoke:
        sequence[0] = max(fit, key=lambda r: int(r['channels']))
        used_checks = [max(checks, key=lambda r: int(r['channels']))]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    code = ['finetuning/run_decoder_aware_codebooks.py', 'finetuning/run_product_quantization_probe.py',
            'compression/product_quantization.py', 'compression/alphabet_stream.py',
            'finetuning/model_factory.py', 'finetuning/representation_study.py',
            'finetuning/run_physical_guard_search.py', 'finetuning/run_spatial_output_study.py']
    write_json(out / 'protocol.json', {'kind': 'paired_decoder_aware_centroids_v1', 'seed': 2092, 'smoke_only': args.smoke,
        'source_checkpoint_sha256': source_hash, 'source_state_before_sha256': before_state_hash,
        'initial_codebooks_sha256': descriptor['codebooks_sha256'], 'basis_sha256': descriptor['basis_sha256'],
        'center_sha256': descriptor['center_sha256'], 'pca_probe_results_sha256': digest(args.pca_probe / 'results.json'),
        'manifest_sha256': prior['manifest_sha256'], 'code_sha256': {f: digest(ROOT / f) for f in code},
        'input_sha256': {r['sample_path']: digest(ROOT / r['sample_path']) for r in [*fit, *checks]},
        'fit_train': [r['sample_path'] for r in fit], 'check_train': [r['sample_path'] for r in checks],
        'selection_checks': [r['sample_path'] for r in used_checks], 'sequence': [r['sample_path'] for r in sequence],
        'budget_per_arm': budget, 'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
        'trainable_scope': 'only 2x32x64 float32 centroid values; source encoder/decoder/codebook and PCA basis/center frozen',
        'optimizer': 'fresh Adam lr=1e-3, no weight decay, clip=1, batch=1, FP32, TF32=false, same draws in both arms',
        'contrast': 'latent arm minimizes mean squared latent vector error; waveform arm minimizes 0.5 normalized MSE +0.5 L1 through frozen decoder; hard nearest assignments recomputed each step',
        'selection': 'lowest macro normalized reconstruction D on separate train files at 0/64/128/256; step zero always eligible, no validation selection',
        'wire_evaluation': 'same FAK1/physical guards and all shared PCA/codebooks counted explicitly; tables refit on fit files only',
        'advance_rule': 'waveform >=1% guarded normalized D gain vs original and >=1% vs latent arm, at nonincreasing total routed stream vs original; only then consider separate development confirmation',
        'extension': 'none; no LR, objective weight, iteration or quantizer sweep; no validation signal access',
        'limitations': ['source pretrained on train-check files', 'short centroid-only adaptation, not optimum or family impossibility',
                        'same observed PCA opportunity, no new examples or rotation tuning', 'arm compute differs because only waveform backpropagates through decoder']})
    for f in code:
        dest = out / 'code_snapshot' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest)

    def encode(row):
        with torch.no_grad():
            _, packet = load_packet(row)
            target = torch.from_numpy(packet.values)[None, None].cuda()
            valid = torch.from_numpy(packet.valid)[None, None].cuda()
            h, w = target.shape[-2:]
            z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
        return z, target, valid

    def evaluate(books):
        values = []
        with torch.no_grad():
            for row in used_checks:
                z, target, valid = encode(row)
                loss = centroid_objective(z, books, basis, center, model.decoder, target, valid, 'waveform')
                values.append((row['family'], row['configuration'], float(loss)))
        return macro_loss(values)

    selection = {}
    for mode in ('latent', 'waveform'):
        folder = out / mode
        folder.mkdir()
        books = torch.nn.Parameter(initial.clone())
        assert books.requires_grad and not any(p.requires_grad for p in model.parameters())
        optimizer = torch.optim.Adam([books], lr=1e-3)
        best_loss, best_step = evaluate(books), 0
        best = books.detach().cpu().clone()
        history = [{'step': 0, 'macro_check_reconstruction_distortion': best_loss}]
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for step, row in enumerate(sequence, 1):
            z, target, valid = encode(row)
            optimizer.zero_grad(set_to_none=True)
            loss = centroid_objective(z, books, basis, center, model.decoder, target, valid, mode)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite centroid objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([books], 1., error_if_nonfinite=True)
            assert books.grad is not None and not any(p.grad is not None for p in model.parameters())
            optimizer.step()
            del loss, z, target, valid
            if step in (64, 128, 256, budget):
                value = evaluate(books)
                history.append({'step': step, 'macro_check_reconstruction_distortion': value, 'last_gradient_norm': float(norm)})
                if value < best_loss:
                    best_loss, best_step = value, step
                    best = books.detach().cpu().clone()
                temporary = folder / 'last.tmp'
                torch.save({'centroids': books.detach().cpu(), 'step': step, 'optimizer': optimizer.state_dict(),
                            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, temporary)
                temporary.replace(folder / 'last.pt')
                write_json(folder / 'history.json', history)
                print(json.dumps({'arm': mode, 'step': step, 'train_check_D': value, 'best_step': best_step}), flush=True)
        torch.cuda.synchronize()
        np.save(folder / 'selected.npy', best.numpy())
        selection[mode] = {'selected_sha256': digest(folder / 'selected.npy'), 'selected_step': best_step,
            'selected_check_distortion': best_loss, 'updates': budget, 'seconds': time.perf_counter() - start,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
        del optimizer, books, best
    assert source_state_hash(model) == before_state_hash
    assert digest(source) == source_hash
    assert digest(initial_dir / 'basis.npy') == descriptor['basis_sha256']
    assert digest(initial_dir / 'center.npy') == descriptor['center_sha256']
    write_json(out / 'selection.json', selection)
    (out / 'CODEBOOKS_READY').touch()
    del model
    torch.cuda.empty_cache()
    command = [sys.executable, str(ROOT / 'finetuning/run_product_quantization_probe.py'),
        '--baseline-dir', str(args.baseline_dir), '--samples-dir', str(args.samples_dir),
        '--repo-root', str(args.repo_root), '--rotation-probe', str(args.rotation_probe),
        '--external-codebooks', str(out), '--output-dir', str(out / 'evaluation')]
    subprocess.run(command, check=True)
    evaluation = json.loads((out / 'evaluation/results.json').read_text())
    waveform, control = evaluation['arms']['waveform'], evaluation['arms']['latent']
    result = {'source_network_parameter_updates': 0, 'centroid_parameter_updates': 2 * budget,
        'validation_signal_windows_read': 0, 'smoke_only': args.smoke, 'selection': selection,
        'evaluation_results_sha256': digest(out / 'evaluation/results.json'),
        'source_state_after_sha256': before_state_hash,
        'waveform_guarded_gain_vs_latent': 1 - waveform['guarded_macro_normalized_distortion'] / control['guarded_macro_normalized_distortion'],
        'advance_gate_met': bool(waveform['advance_gate_met'] and
            waveform['guarded_macro_normalized_distortion'] <= .99 * control['guarded_macro_normalized_distortion'])}
    write_json(out / 'results.json', result)
    (out / 'COMPLETE').touch()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
