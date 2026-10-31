#!/usr/bin/env python3
"""Single-GPU, bounded native-width comparison at the frozen compact rate."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.finetune_vqvae import load_sample, save_checkpoint
from finetuning.model_factory import new_model
from finetuning.latent_geometry import latent_positions
from finetuning.run_architecture_initialization_experiment import initialize_codebook
from finetuning.run_sampling_experiment import check_splits, run
from finetuning.run_spatial_abc_study import extension_decision, status
from compression.encode_decode_samples import encode_and_decode


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def capacity_config(width):
    return dict(architecture='native-v1', base_channels=width, embedding_dim=128,
                num_embeddings=2048, downsample_levels=4, commitment_cost=.25,
                latent_geometry='spatial-4x', exact_eval_quantization=True,
                activation_checkpointing=True)


def smoke(config, sample):
    """Disposable maximum-shape model; no smoke updates enter training."""
    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark
    # Compare code-only replay without nondeterministic CUDA convolution noise.
    # Trainer/evaluator subprocesses retain their existing numerical policy.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2033)
    model = new_model(config, 'cuda').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    tensor = torch.from_numpy(load_sample(sample, 'robust-escape-v1'))[None, None].cuda()
    assert tuple(tensor.shape[-2:]) == (512, 1000)
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(5):
        torch.cuda.synchronize(); started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        recon, vq, indices, _ = model(tensor)
        loss = .5 * F.mse_loss(recon, tensor) + .5 * F.l1_loss(recon, tensor) + vq
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize(); durations.append(time.monotonic() - started)
        assert indices.numel() == 8064
    model.eval()
    with torch.no_grad():
        for height in (1, 111, 512):
            x = tensor[..., :height, :]
            recon, _, indices, _ = model(x)
            replay, codes = encode_and_decode(model, x)
            torch.testing.assert_close(recon, replay, atol=0, rtol=0)
            torch.testing.assert_close(indices, codes)
            assert recon.shape == x.shape
            assert indices.numel() == latent_positions(model, height, 1000)
    result = {'parameters': sum(p.numel() for p in model.parameters()),
              'max_shape': [512, 1000], 'indices_at_max_shape': 8064,
              'step_seconds': durations, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'exact_discrete_replay': True, 'smoke_updates_discarded': True}
    result['deterministic_cudnn_for_replay_check'] = True
    del model, optimizer, tensor, recon, replay, loss, vq, x
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = original_deterministic
    torch.backends.cudnn.benchmark = original_benchmark
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly physical GPU 0 must be visible; refusing multi-GPU execution')
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.repo_root.resolve()))
    output, samples = args.output_dir.resolve(), args.samples_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent / f'.{output.name}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_splits(samples, 1)
    rows = read_csv(samples / 'samples_manifest.csv')
    for split, expected in (('train', 14620), ('val', 6940)):
        selected = [r for r in rows if r['split'] == split]
        assert len(selected) == expected
        assert len(list((samples / split).glob('*.npz'))) == expected
        assert {r['spatial_band_policy'] for r in selected} == {'persistent-amplitude-v1'}
        assert {r['family'] for r in selected} == {'doped', 'multicore', 'multimode', 'single_mode'}
        assert all((ROOT / r['sample_path']).is_file() for r in selected)
    assert {int(r['fold']) for r in rows if r['split'] == 'val'} == {2}
    prep = json.loads((samples / 'preprocessing.json').read_text())
    assert prep['spatial_band_policy'] == 'persistent-amplitude-v1'
    transfer = ROOT / 'results/remote_setup/initial_TRANSFER_SHA256.json'
    if not transfer.is_file():
        raise ValueError('Verified remote transfer manifest required')
    transferred = json.loads(transfer.read_text())
    for row in rows:
        if row['split'] in ('train', 'val'):
            relative = row['sample_path']
            assert relative in transferred, relative
            assert digest(ROOT / relative) == transferred[relative], relative
    maps = {m['channel_map_id']: m for m in map(json.loads, (samples / 'channel_maps.jsonl').read_text().splitlines())}
    for row in rows:
        if row['split'] not in ('train', 'val'):
            continue
        mapping = maps[row['channel_map_id']]
        decision = prep['file_decisions'][row['drive_id']]
        assert mapping['spatial_filter_id'] == row['spatial_filter_id'] == decision['filter_id']
        assert not set(mapping['source_channel_indices']).intersection(decision['removed_source_channel_indices'])
        assert mapping['selected_rows_contiguous']
    seed = 2033
    rng = random.Random(seed)
    calibration = []
    for family in ('doped', 'multicore', 'multimode', 'single_mode'):
        choices = sorted(ROOT / r['sample_path'] for r in rows if r['split'] == 'train' and r['family'] == family)
        calibration.extend(rng.sample(choices, 8))
    calibration.sort()
    maximum = next(ROOT / r['sample_path'] for r in rows if r['split'] == 'train'
                   and int(r['channels']) == 512 and int(r['time_samples']) == 1000)
    files = sorted((ROOT / 'finetuning').glob('*.py')) + sorted((ROOT / 'compression').glob('*.py'))
    files += sorted((args.repo_root / 'vqvae').glob('*.py'))
    metadata = ('samples_manifest.csv', 'preprocessing.json', 'channel_maps.jsonl')
    protocol = {'kind': 'native_capacity_compact_v1', 'seed': seed, 'test_fold': 1, 'validation_fold': 2,
                'arms': {'large96M': capacity_config(120), 'control28M': capacity_config(64)},
                'order': ['large96M', 'control28M'], 'initial_epochs': 16, 'max_epochs': 24,
                'steps_per_epoch': 4096, 'extension': '8 epochs once if last-four-block RMSE gain >=0.2%',
                'learning_rate': 1e-4, 'weight_decay': 1e-5, 'batch_size': 1,
                'objective': '0.5*MSE + 0.5*L1 + VQ; spectral/channel weights zero',
                'normalization': 'robust-escape-v1', 'preprocessing': prep['parameters'],
                'calibration': [str(p) for p in calibration], 'samples_dir': str(samples),
                'metadata_sha256': {n: digest(samples / n) for n in metadata},
                'transfer_manifest_sha256': digest(transfer),
                'code_sha256': {str(p): digest(p) for p in files},
                'software': {'torch': str(torch.__version__), 'cuda': torch.version.cuda},
                'gpu': torch.cuda.get_device_name(0), 'visible_device': '0',
                'initialization': 'random encoder/decoder; train-only codebook reservoir, no offshore tensors',
                'selection': 'minimum fixed pilot macro RMSE; 2 windows per source file/region; full validation at phase endpoints',
                'rate_note': 'Same per-shape fixed indices; nominal rate includes normalization/escapes, not every container/map/model cost',
                'no_test_inference': True, 'final_model_promotion': False}
    if args.resume:
        assert json.loads((output / 'protocol.json').read_text()) == protocol, 'Protocol changed'
        if (output / 'COMPLETE').exists():
            return
    else:
        output.mkdir(exist_ok=False)
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        for source in files:
            relative = source.relative_to(ROOT)
            dest = output / 'code_snapshot' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    def verify():
        if shutil.disk_usage(output).free < 20 * 2**30:
            raise RuntimeError('Less than 20 GiB disk headroom; stopping without deleting files')
        for path, expected in protocol['code_sha256'].items():
            assert digest(path) == expected, path
        for name, expected in protocol['metadata_sha256'].items():
            assert digest(samples / name) == expected, name
    def evaluate(label, phase, path):
        directory = output / f'full_phase{phase}_{label}'
        if (directory / 'COMPLETE').exists():
            return
        command = ['finetuning/evaluate_spatial_study.py', '--samples-dir', str(samples),
                   '--repo-root', str(args.repo_root), '--test-fold', '1', '--output-dir', str(directory),
                   '--candidate', f'{label}={path}']
        if directory.exists():
            command += ['--resume']
        run(command, output / f'full_phase{phase}_{label}.log', quiet=True)
    def train(label, target):
        verify()
        directory = output / label
        last = directory / 'last.pt'
        completed = 0
        if last.exists():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            completed = int(saved['epoch']); del saved
            for filename in ('metrics.csv', 'validation_metrics.csv'):
                history = read_csv(directory / filename)
                if history:
                    assert int(history[-1]['epoch']) == completed, 'CSV/checkpoint epoch mismatch'
        retained = output / f'retained_phase{target}_{label}.pt'
        if completed < target:
            status(output, 'training', arm=label, completed_epochs=completed, target_epochs=target)
            command = ['finetuning/finetune_vqvae.py', '--repo-root', str(args.repo_root),
                       '--checkpoint', str(output / 'initializations' / f'{label}.pt'),
                       '--samples-dir', str(samples), '--output-dir', str(directory),
                       '--epochs', str(target-completed), '--steps-per-epoch', '4096',
                       '--learning-rate', '0.0001', '--weight-decay', '0.00001', '--seed', str(seed),
                       '--device', 'cuda', '--sampling', 'window', '--normalization', 'robust-escape-v1',
                       '--latent-geometry', 'spatial-4x', '--physical-validation', '--validation-details',
                       '--evaluate-initial', '--val-samples-per-file', '2', '--codebook-diagnostics',
                       '--channel-diagnostics', '--spatial-weighted-validation',
                       '--early-stopping-metric', 'macro-rmse', '--early-stopping-patience', '25',
                       '--early-stopping-min-delta', '0.0001']
            if completed:
                command += ['--resume', str(last)]
            run(command, output / f'{label}.log', resume=bool(completed), quiet=True)
        if not retained.exists():
            saved = torch.load(last, map_location='cpu', weights_only=False)
            assert int(saved['epoch']) == target; del saved
            shutil.copy2(directory / 'best_macro.pt', retained)
        evaluate(label, target, retained)
    try:
        verify()
        smoke_path = output / 'smoke.json'
        if not smoke_path.exists():
            status(output, 'maximum_shape_smoke')
            results = {label: smoke(config, maximum) for label, config in protocol['arms'].items()}
            smoke_path.write_text(json.dumps(results, indent=2) + '\n')
        initial = output / 'initializations'; initial.mkdir(exist_ok=True)
        for label, config in protocol['arms'].items():
            path = initial / f'{label}.pt'
            if path.exists():
                continue
            status(output, 'initialization', arm=label)
            torch.manual_seed(seed)
            model = new_model(config, 'cuda')
            vectors = initialize_codebook(model, calibration, seed, normalization='robust-escape-v1')
            save_checkpoint({'config': config, 'model_state_dict': model.cpu().state_dict(), 'epoch': 0,
                             'initialization': {'random': True, 'reservoir_vectors': vectors},
                             'protocol_sha256': digest(output / 'protocol.json')}, path)
            del model; torch.cuda.empty_cache()
        for label in protocol['order']:
            train(label, 16)
        ledger = output / 'extension_decisions.json'
        decisions = {label: extension_decision(read_csv(output / label / 'validation_metrics.csv'))
                     for label in protocol['order']}
        if ledger.exists():
            decisions = json.loads(ledger.read_text())
        else:
            ledger.write_text(json.dumps(decisions, indent=2) + '\n')
        for label in protocol['order']:
            if decisions[label]['extend']:
                train(label, 24)
        summaries = {}
        for label in protocol['order']:
            phase = 24 if decisions[label]['extend'] else 16
            summaries[label] = json.loads((output / f'full_phase{phase}_{label}' / f'{label}.json').read_text())
        (output / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
        # Full per-configuration guards are descriptive, never automatic promotion.
        groups = {}
        for label in protocol['order']:
            phase = 24 if decisions[label]['extend'] else 16
            groups[label] = {(r['family'], r['configuration']): r for r in read_csv(
                output / f'full_phase{phase}_{label}' / f'{label}.groups.csv')}
        regressions = []
        for key, candidate in groups['large96M'].items():
            reference = groups['control28M'][key]
            for metric, direction in (('relative_rmse', 1), ('correlation', -1), ('ordinary_channel_p90_relative_rmse', 1)):
                delta = float(candidate[metric]) - float(reference[metric])
                if direction * delta > 1e-8:
                    regressions.append({'family': key[0], 'configuration': key[1], 'metric': metric, 'delta': delta})
        (output / 'guards.json').write_text(json.dumps({'regressions': regressions,
            'final_model_promoted': False, 'note': 'Compare phase16 separately if selective compute differs'}, indent=2)+'\n')
        (output / 'COMPLETE').write_text('Capacity comparison and full validation complete; no final promotion.\n')
        status(output, 'complete', final_model_promoted=False)
    except BaseException as error:
        status(output, 'failed', error=repr(error))
        (output / 'FAILED').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
