#!/usr/bin/env python3
"""Train-only context-before-quantization probe with a capacity-matched local control."""
import argparse
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.analysis_context import analysis_context_forward
from finetuning.finetune_vqvae import save_checkpoint
from finetuning.latent_context_adapter import LatentContextAdapter
from finetuning.model_factory import build_model
from finetuning.representation_study import digest, read_manifest, pilot_rows, load_packet, distortion, write_json
from finetuning.run_decoder_aware_codebooks import source_state_hash
from finetuning.run_spatial_output_study import split_train_files
from finetuning.run_untied_synthesis_probe import weighted_macro_distortion
from finetuning.update_schedule import load_independent_optimizer_state

ARMS = ('local', 'axial')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-protocol', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one remote RTX4090 required')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.backends.mha.set_fastpath_enabled(False)
    sys.path.insert(0, str(args.repo_root.resolve()))
    out = args.output_dir; out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / '.lr_schedule_gpu.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    reference = json.loads(args.source_protocol.read_text())
    if (source.get('epoch') != 24 or source.get('initialization', {}).get('arm') != 'A'
            or source['config']['latent_geometry'] != 'spatial-4x'
            or source['protocol_sha256'] != digest(args.source_protocol)
            or digest(args.samples_dir / 'samples_manifest.csv') != reference['metadata_sha256']['samples_manifest.csv']):
        raise ValueError('Original transferred A source/corpus differs')
    seed = 2106; torch.manual_seed(seed)
    model = build_model({**source['config'], 'exact_eval_quantization': True}, source, 'cuda').eval().requires_grad_(False)
    del source
    frozen_hash = source_state_hash(model)
    rows = read_manifest(args.samples_dir)['train']
    selected = pilot_rows(rows, seed=2100)
    _, held = split_train_files(selected, seed=2100)
    checks = [selected[i] for i in held]; held_files = {r['drive_id'] for r in checks}
    fit = [r for r in rows if r['drive_id'] not in held_files]
    rng = random.Random(seed)
    initial, maximum, interval = (2, 2, 1) if args.smoke else (512, 2048, 128)
    sequence = [rng.choice(fit) for _ in range(maximum)]
    if args.smoke:
        sequence[0] = max(fit, key=lambda row: int(row['channels']))
        checks = [max(checks, key=lambda row: int(row['channels']))]
    used = {r['sample_path']: r for r in sequence + checks}
    code = ['finetuning/run_analysis_context_probe.py', 'finetuning/analysis_context.py',
            'finetuning/latent_context_adapter.py', 'finetuning/run_index_search.py',
            'finetuning/run_decoder_aware_codebooks.py', 'finetuning/run_untied_synthesis_probe.py',
            'compression/untied_synthesis.py', 'finetuning/run_spatial_output_study.py',
            'finetuning/representation_study.py', 'finetuning/model_factory.py',
            'finetuning/latent_geometry.py', 'finetuning/robust_escape.py',
            'finetuning/reversible_normalization.py', 'finetuning/update_schedule.py']
    counts = {kind: sum(p.numel() for p in LatentContextAdapter(kind).parameters()) for kind in ARMS}
    if max(counts.values()) / min(counts.values()) >= 1.01:
        raise ValueError('Analysis control capacities differ by >=1%')
    protocol = {'kind': 'context_before_quantization_probe_v1', 'seed': seed, 'smoke_only': args.smoke,
        'source_sha256': digest(args.source), 'source_state_sha256': frozen_hash,
        'source_protocol_sha256': digest(args.source_protocol),
        'manifest_sha256': digest(args.samples_dir / 'samples_manifest.csv'),
        'arms': list(ARMS), 'trainable_parameters': counts,
        'initial_updates_per_arm': initial, 'maximum_updates_per_arm': maximum, 'check_interval': interval,
        'sequence': [r['sample_path'] for r in sequence], 'checks': [r['sample_path'] for r in checks],
        'input_sha256': {name: digest(ROOT / name) for name in sorted(used)},
        'code_sha256': {name: digest(ROOT / name) for name in code},
        'vendor_sha256': {name: digest(args.repo_root / 'vqvae' / name) for name in ('train_vqvae.py', 'evaluate_vqvae.py')},
        'freeze': 'Original encoder, codebook and decoder frozen; identity residual adapter acts on continuous encoder features BEFORE nearest quantization',
        'optimizer': 'Fresh Adam lr=1e-3 clip1 no wd, same batches and seeds, FP32 TF32=false deterministic math attention',
        'objective': '0.5 MSE+0.5 L1+native VQ loss on ordinary cells; commitment gradient reaches analysis adapter, codebook stays frozen',
        'selection': 'Minimum normalized D on separate train-file checks, original identity eligible; no validation signals',
        'extension': 'Once all arms to2048 if winner>=0.5% vs source, >=0.2% recent, and either contextual winner>=0.25% over local or local winner>=1% over source',
        'advance': '>=1% normalized D gain vs source and >=0.5% vs local for attention claim; later codec gate>=5% physical RMSE at nonincreasing complete bytes/unchanged guards',
        'wire_note': 'Same grid/K and decoder; analysis IDs can change. Nominal bits equal, complete bytes not assumed. Adapter is encoder-only, not a receiver weight.',
        'limitations': ['Internal checks were seen by historical base training; not independent acquisitions',
                       'This isolated frozen-backbone probe is not a full jointly trained Transformer codec',
                       'Normalized loss improvements are not physical RMSE percentages'],
        'parameter_updates_to_base': 0, 'validation_signal_windows_read': 0, 'final_holdout_accessed': False,
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'torch': str(torch.__version__)}
    if args.resume:
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Analysis probe resume changed')
        if (out / 'COMPLETE').exists():
            return
    else:
        out.mkdir(exist_ok=False); write_json(out / 'protocol.json', protocol)
        for name in code:
            path = out / 'code_snapshot' / name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, path)

    @lru_cache(maxsize=8)
    def entry(name):
        row = used[name]
        if row['split'] != 'train' or digest(ROOT / name) != protocol['input_sha256'][name]:
            raise ValueError('Only unchanged training inputs allowed')
        raw, packet = load_packet(row)
        target = torch.from_numpy(packet.values)[None, None]
        h, w = raw.shape
        with torch.no_grad():
            features = model.encoder(F.pad(target.cuda(), (0, (-w) % 16, 0, (-h) % 16))).cpu()
        return features, target, torch.from_numpy(packet.valid)[None, None], raw.size

    def evaluate(adapter):
        adapter.eval(); values = []
        with torch.no_grad():
            for row in checks:
                features, target, valid, cells = entry(row['sample_path'])
                predicted, _, _, _ = analysis_context_forward(model, adapter, features.cuda(), target.shape[-2:])
                value = float(distortion(predicted, target.cuda(), valid.cuda()))
                if not np.isfinite(value):
                    raise RuntimeError('Nonfinite analysis selection score')
                values.append((row['family'], row['configuration'], value, cells))
        return weighted_macro_distortion(values)

    identity = LatentContextAdapter('local').cuda(); baseline = evaluate(identity); del identity
    if baseline <= 0:
        raise ValueError('No positive baseline distortion')

    def train(kind, budget):
        folder = out / kind; folder.mkdir(exist_ok=True); torch.manual_seed(seed)
        adapter = LatentContextAdapter(kind).cuda()
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        best, best_step, completed = baseline, 0, 0
        best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
        history = [{'step': 0, 'check_D': baseline}]; last = folder / 'last.pt'
        if last.exists():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            if saved['protocol_sha256'] != digest(out / 'protocol.json'):
                raise ValueError('Analysis adapter lineage changed')
            adapter.load_state_dict(saved['adapter']); load_independent_optimizer_state(optimizer, saved['optimizer'])
            completed, best, best_step = saved['step'], saved['best_D'], saved['best_step']
            best_state, history = saved['best_state'], saved['history']
            if {float(state['step']) for state in optimizer.state.values()} != {float(completed)}:
                raise ValueError('Restored analysis optimizer counter differs')
            del saved
        nonzero_gradient_in_block = False
        for step in range(completed + 1, budget + 1):
            if (step - 1) % interval == 0 and any(digest(ROOT / name) != expected for name, expected in protocol['code_sha256'].items()):
                raise ValueError('Frozen analysis code changed')
            features, target, valid, _ = entry(sequence[step - 1]['sample_path'])
            adapter.train(); optimizer.zero_grad(set_to_none=True)
            predicted, vq, _, _ = analysis_context_forward(model, adapter, features.cuda(), target.shape[-2:], training=True)
            loss = distortion(predicted, target.cuda(), valid.cuda()) + vq
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite analysis objective')
            loss.backward()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise AssertionError('Base weights received gradients')
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
            nonzero_gradient_in_block |= bool(norm > 0)
            optimizer.step()
            if {float(state['step']) for state in optimizer.state.values()} != {float(step)}:
                raise ValueError('Analysis optimizer dose mismatch')
            if step % interval == 0 or step == budget:
                if not nonzero_gradient_in_block:
                    raise RuntimeError('Analysis adapter received no gradient throughout a block')
                nonzero_gradient_in_block = False
                value = evaluate(adapter); history.append({'step': step, 'check_D': value})
                if value < best:
                    best, best_step = value, step
                    best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
                save_checkpoint({'adapter': adapter.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                    'best_D': best, 'best_step': best_step, 'best_state': best_state, 'history': history,
                    'protocol_sha256': digest(out / 'protocol.json')}, last)
                write_json(folder / 'history.json', history)
                print(f'analysis-context {kind} {step}/{budget} D={value:.8f} best={best:.8f}', flush=True)
        if source_state_hash(model) != frozen_hash:
            raise AssertionError('Base model changed')
        save_checkpoint({'kind': kind, 'adapter_state': best_state, 'source_sha256': protocol['source_sha256']}, folder / 'selected.pt')
        return {'best_D': best, 'best_step': best_step, 'updates': max(budget, completed), 'history': history}

    selections = {kind: train(kind, initial) for kind in ARMS}
    extension = out / 'EXTENSION.json'
    if extension.exists():
        decision = json.loads(extension.read_text())
    else:
        winner = min(ARMS, key=lambda kind: selections[kind]['best_D'])
        scores = {row['step']: row['check_D'] for row in selections[winner]['history']}
        recent = 1 - scores[512] / scores[256] if not args.smoke and scores[256] > 0 else 0.
        best = selections[winner]['best_D']
        structural = best <= .9975 * selections['local']['best_D'] if winner != 'local' else best <= .99 * baseline
        decision = {'extend_both': not args.smoke and best <= .995 * baseline and recent >= .002 and structural,
                    'initial_selection': selections, 'winner': winner, 'recent_gain': recent}
        write_json(extension, decision)
    if decision['extend_both']:
        selections = {kind: train(kind, maximum) for kind in ARMS}
    winner = min(ARMS, key=lambda kind: selections[kind]['best_D'])
    source_gain = 100 * (1 - selections[winner]['best_D'] / baseline)
    local_gain = 100 * (1 - selections[winner]['best_D'] / selections['local']['best_D'])
    if source_state_hash(model) != frozen_hash or digest(args.source) != protocol['source_sha256']:
        raise AssertionError('Frozen source changed')
    if any(digest(args.repo_root / 'vqvae' / name) != expected for name, expected in protocol['vendor_sha256'].items()):
        raise ValueError('Frozen vendor implementation changed')
    write_json(out / 'results.json', {'baseline_check_D': baseline, 'selection': selections, 'best_arm': winner,
        'best_gain_vs_source_pct': source_gain, 'best_gain_vs_local_pct': local_gain,
        'advance_gate_met': not args.smoke and source_gain >= 1,
        'context_advantage_supported': not args.smoke and winner == 'axial' and local_gain >= .5,
        'source_weights_unchanged': True, 'analysis_assignments_fixed': False,
        'validation_signal_windows_read': 0, 'smoke_only': args.smoke, 'model_promoted': False})
    (out / 'COMPLETE').touch(); lock.close()


if __name__ == '__main__':
    main()
