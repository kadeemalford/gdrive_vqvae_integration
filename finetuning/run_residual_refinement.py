#!/usr/bin/env python3
"""Bounded signal-residual neural refinement of an immutable compact DAS codec.

The refinement decoder receives its own discrete indices only. Its output is
added to the base waveform decoded from transmitted base indices. No residual
normalization side channel, continuous bypass, or parameter updates to the base.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'smartgrids'))
from compression.symbol_stream import SymbolCodec
from compression.study_bundle import write_bundle, read_bundle
from finetuning.model_factory import new_model, build_model
from finetuning.latent_geometry import latent_positions
from finetuning.robust_escape import encode_robust
from finetuning.reversible_normalization import decode, parse_side_stream
from finetuning.physical_validation import ordinary_fidelity
from finetuning.evaluate_checkpoints import metrics
from finetuning.diagnose_reconstruction import spectral_metrics
from finetuning.reconstruction_objective import aggregate_validation, stratified_validation, validation_metadata
from finetuning.finetune_vqvae import save_checkpoint
from finetuning.run_spatial_abc_study import extension_decision, status
from finetuning.run_sampling_experiment import check_splits
from finetuning.run_frozen_entropy_roi import encoder_indices, decoder_waveform


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def append_csv(path, row):
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def json_file(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n');temporary.replace(path)


def measured(raw, packet, predicted):
    restored = decode(predicted, packet.side_stream)
    assert raw[~packet.valid].tobytes() == restored[~packet.valid].tobytes()
    _, mean, scale, _, _ = parse_side_stream(packet.side_stream)
    normalized = ((restored.astype(np.float64)-mean)/scale).astype(np.float32)
    normalized[~packet.valid] = 0
    physical = ordinary_fidelity(raw, restored, packet.valid)
    return {**metrics(packet.values[packet.valid], normalized[packet.valid]),
            **spectral_metrics(packet.values, normalized), **physical}


def admissible(base, refined):
    return (refined['relative_rmse'] <= base['relative_rmse']
            and refined['correlation'] >= base['correlation']
            and refined['ordinary_channel_p90_relative_rmse'] <= base['ordinary_channel_p90_relative_rmse'])


def read_window(path):
    with np.load(path, allow_pickle=False) as sample:
        raw = np.asarray(sample['data'], dtype=np.float32)
    return raw, encode_robust(raw)


def new_refiner(config, device):
    model = new_model(config, device)
    if config.get('quantizer', 'vq') == 'fsq-8x8x32':
        from finetuning.fsq_quantizer import FiniteScalarQuantizer
        if hasattr(model, '_exact_eval_hook'):
            model._exact_eval_hook.remove()
            del model._exact_eval_hook
        model.quantizer = FiniteScalarQuantizer(model.embedding_dim).to(device)
    elif config.get('quantizer', 'vq') != 'vq':
        raise ValueError('Unknown residual quantizer')
    return model


def load_refiner(saved, device):
    model = new_refiner(saved['config'], device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.eval()


def residual_input(base, packet):
    tensor = torch.from_numpy(packet.values)[None, None].cuda()
    mask = torch.from_numpy(packet.valid)[None, None].cuda()
    with torch.no_grad():
        prediction = base(tensor)[0]
        residual = (tensor-prediction).masked_fill(~mask, 0.)
    return tensor, mask, prediction, residual


def validation(base, refiner, paths, meta):
    rows = {'base': [], 'all_refined': [], 'guarded': []}
    accepted = 0
    base.eval();refiner.eval()
    with torch.no_grad():
        for path in paths:
            raw, packet = read_window(path)
            _, _, basic, residual = residual_input(base, packet)
            refined = basic + refiner(residual)[0]
            a = measured(raw, packet, basic[0, 0].cpu().numpy())
            b = measured(raw, packet, refined[0, 0].cpu().numpy())
            keep = admissible(a, b);accepted += keep
            common = {'family': meta[path]['family'], 'configuration': meta[path]['configuration'],
                      '_aggregation_weight': float(raw.size)}
            for key, values in (('base', a), ('all_refined', b), ('guarded', b if keep else a)):
                rows[key].append({**common, **values})
    result = {key: aggregate_validation(values)[0] for key, values in rows.items()}
    result['accepted_windows'] = accepted
    return result


def allocate_refinements(values, factor, retained=None):
    """Nested budget expansion: never drop a previously selected refinement."""
    selected = set(retained or ())
    base_bytes = sum(r['base_bytes'] for r in values)
    used = base_bytes + sum(r['refinement_bytes'] for r in values if r['sample_id'] in selected)
    budget = math.floor(base_bytes*factor)
    assert used <= budget
    candidates = []
    for row in values:
        if not row['admitted'] or row['sample_id'] in selected:
            continue
        a, b = row['base'], row['refined']
        error_gain = np.clip(1-b['relative_rmse']**2/max(a['relative_rmse']**2, 1e-12), 0, 1)
        tail_gain = np.clip(1-b['ordinary_channel_p90_relative_rmse']/max(a['ordinary_channel_p90_relative_rmse'], 1e-6), 0, 1)
        gain = .5*(error_gain+tail_gain)*row['cells']
        if np.isfinite(gain) and gain > 0:
            candidates.append((-gain/row['refinement_bytes'], row['sample_id'], row['refinement_bytes']))
    for _, name, extra in sorted(candidates):
        if used+extra <= budget:
            selected.add(name);used += extra
    return selected


def full_validation(base, checkpoint, source, samples, metadata, calibration, destination):
    if (destination/'COMPLETE').exists():
        return
    destination.mkdir(exist_ok=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['pipeline']['role'] == 'normalized-signal-residual-v1'
    base_codec = SymbolCodec.from_bytes((source/'compact.tables.json').read_bytes())
    assert saved['pipeline']['base_sha256'] == base_codec.spec['checkpoint_sha256']
    refiner = load_refiner(saved, 'cuda');del saved
    counts = np.zeros(2048, dtype=np.int64)
    conditional = np.zeros((17, 2048), dtype=np.int64)
    with torch.no_grad():
        for path in calibration:
            _, packet = read_window(path)
            _, _, _, residual = residual_input(base, packet)
            indices, _ = encoder_indices(refiner, residual[0, 0].cpu().numpy())
            counts += np.bincount(indices, minlength=2048)
            contexts = np.concatenate(([16], indices[:-1]//128))
            conditional += np.bincount(contexts*2048+indices, minlength=17*2048).reshape(17, 2048)
    codec = SymbolCodec.from_counts(counts, conditional, digest(checkpoint))
    (destination/'refinement.tables.json').write_bytes(codec.serialized)
    records = []
    paths = sorted((samples/'val').glob('*.npz'))
    for number, path in enumerate(paths):
        name = path.stem
        if number % 250 == 0:
            print(f'[full residual] {destination.name}: {number}/{len(paths)}', flush=True)
        raw, packet = read_window(path)
        base_path = source/'frames/compact'/f'{name}.fsc'
        base_indices, base_shape, base_grid, base_side = base_codec.decode(base_path.read_bytes())
        assert base_shape == raw.shape and base_side == packet.side_stream
        with torch.no_grad():
            basic_values = decoder_waveform(base, base_indices, base_shape, base_grid)
            basic = torch.from_numpy(basic_values)[None, None].cuda()
            target = torch.from_numpy(packet.values)[None, None].cuda()
            mask = torch.from_numpy(packet.valid)[None, None].cuda()
            residual = (target-basic).masked_fill(~mask, 0.)
            indices, latent = encoder_indices(refiner, residual[0, 0].cpu().numpy())
            frame, _ = codec.encode(indices, raw.shape, latent, b'')
            recovered, shape, grid, side = codec.decode(frame)
            np.testing.assert_array_equal(indices, recovered);assert side == b''
            prediction = decoder_waveform(refiner, recovered, shape, grid)
        a = measured(raw, packet, basic[0, 0].cpu().numpy())
        b = measured(raw, packet, basic[0, 0].cpu().numpy()+prediction)
        frame_path = destination/'refinements'/f'{name}.fsc'
        frame_path.parent.mkdir(exist_ok=True);frame_path.write_bytes(frame)
        assert base_path.is_file()
        records.append({'sample_id': name, 'drive_id': metadata[path]['drive_id'],
                        'family': metadata[path]['family'], 'configuration': metadata[path]['configuration'],
                        'cells': int(raw.size), 'base': a, 'refined': b,
                        'admitted': admissible(a, b), 'base_bytes': base_path.stat().st_size,
                        'refinement_bytes': len(frame), 'refinement_indices': len(indices)})
    json_file(destination/'window_metrics.json', records)
    by_file = defaultdict(list)
    for row in records:
        by_file[row['drive_id']].append(row)
    modes = {'base': set(), 'all_refined': {r['sample_id'] for r in records},
             'guarded_all': {r['sample_id'] for r in records if r['admitted']}}
    previous = {}
    for factor in (1.25, 1.5):
        selected = set()
        for file, values in by_file.items():
            previous[file] = allocate_refinements(values, factor, previous.get(file))
            selected.update(previous[file])
        modes[f'guarded_{str(factor).replace(".", "p")}'] = selected
    assets = {name: (source/name).read_bytes() for name in
              ('channel_maps.jsonl', 'preprocessing_policy.json', 'compact.tables.json')}
    with (destination/'validation_metadata.csv').open('w') as stream:
        columns = list(metadata[paths[0]])
        writer = csv.DictWriter(stream, fieldnames=columns);writer.writeheader()
        writer.writerows(metadata[path] for path in paths)
    assets['validation_metadata.csv'] = (destination/'validation_metadata.csv').read_bytes()
    summaries, grouped = {}, {}
    cells = sum(r['cells'] for r in records)
    for label, selected in modes.items():
        rows, frames, sequence = [], [], []
        for row in records:
            refine = row['sample_id'] in selected
            rows.append({'family': row['family'], 'configuration': row['configuration'],
                         '_aggregation_weight': float(row['cells']), **row['refined' if refine else 'base']})
            frames.append(source/'frames/compact'/f'{row["sample_id"]}.fsc')
            if refine:
                frames.append(destination/'refinements'/f'{row["sample_id"]}.fsc')
            sequence.append([row['sample_id'], refine])
        shared = {**assets, 'sequence.json': json.dumps(sequence, separators=(',', ':')).encode(),
                  'composition.json': json.dumps({'kind': 'base_plus_normalized_signal_residual_v1',
                    'normalization_side': 'base_frame_only', 'refinement_side': 'empty',
                    'source_window_required_by_decoder': False}).encode()}
        if selected:
            shared['refinement.tables.json'] = codec.serialized
        bundle = destination/f'{label}.fsb'
        if bundle.exists():
            raise FileExistsError('Partial full evaluation contains existing bundles; preserve and inspect before rerun')
        accounting = write_bundle(bundle, shared, frames)
        reader = read_bundle(bundle);assert next(reader) == shared
        for encoded, frame_path in zip(reader, frames, strict=True):
            assert encoded == frame_path.read_bytes()
        macro, groups = aggregate_validation(rows)
        summaries[label] = {**macro, **accounting, 'stream_bytes': bundle.stat().st_size,
                            'actual_ratio_shared_weights': 4*cells/bundle.stat().st_size,
                            'refined_windows': len(selected), 'windows': len(records),
                            'refined_cell_fraction': sum(r['cells'] for r in records if r['sample_id'] in selected)/cells,
                            'all_transport_frames_roundtripped': True, 'no_model_promotion': True}
        weight_bytes = 4*sum(p.numel() for p in base.parameters())
        if selected:
            weight_bytes += 4*sum(p.numel() for p in refiner.parameters())
        summaries[label]['shared_float32_weight_bytes'] = weight_bytes
        summaries[label]['ratio_including_weights_once'] = 4*cells/(bundle.stat().st_size+weight_bytes)
        grouped[label] = groups
        with (destination/f'{label}.groups.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    json_file(destination/'results.json', summaries)
    reference = {(r['family'], r['configuration']): r for r in grouped['base']}
    guards = {}
    for label, groups in grouped.items():
        bad = []
        for row in groups:
            original = reference[row['family'], row['configuration']]
            for metric, direction in (('relative_rmse', 1), ('correlation', -1), ('ordinary_channel_p90_relative_rmse', 1)):
                delta = row[metric]-original[metric]
                if direction*delta > 1e-8:
                    bad.append({'family': row['family'], 'configuration': row['configuration'], 'metric': metric, 'delta': delta})
        guards[label] = {'regressions': bad, 'final_model_promoted': False}
    json_file(destination/'guards.json', guards)
    report = ['# Frozen-base neural residual refinement', '',
              'Selected on a fixed pilot; complete development validation, not external acceptance.', '',
              '| Mode | RMSE | Correlation | Channel p90 | Actual bundle ratio | Refined cells |',
              '|---|---:|---:|---:|---:|---:|']
    for label, values in summaries.items():
        report.append(f'| {label} | {values["macro_relative_rmse"]:.6f} | {values["macro_correlation"]:.6f} | {values["macro_ordinary_channel_p90_relative_rmse"]:.6f} | {values["actual_ratio_shared_weights"]:.3f} | {values["refined_cell_fraction"]:.3f} |')
    report += ['', 'Base weights immutable. Refinement decoder receives discrete residual codes only.',
               'Original normalization/escapes transmitted once in the base frame. Residual has no additional scale side channel.',
               'Actual bytes include losslessly packaged geometry/metadata, tables, boundaries and refinements.',
               'Nested allocation preserves lower-budget choices. Guards are encoder rules, not independent acceptance.']
    (destination/'REPORT.md').write_text('\n'.join(report)+'\n')
    (destination/'COMPLETE').write_text('Residual frames and transport verified; no promotion.\n')
    del refiner;gc.collect();torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--quantizer', choices=('vq', 'fsq-8x8x32'), default='vq')
    parser.add_argument('--samples-dir',type=Path,default=ROOT/'data/finetune/spatial_band_v1_fold0_20261024')
    parser.add_argument('--base-checkpoint',type=Path,default=ROOT/'results/experiments/spatial_abc_20261024_v2/compact_A/best_macro.pt')
    parser.add_argument('--base-frames-source',type=Path,default=ROOT/'results/experiments/frozen_entropy_roi_fold0_20261030')
    parser.add_argument('--test-fold',type=int,choices=range(4),default=0)
    parser.add_argument('--seed',type=int,default=2035)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or '4060' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Local RTX4060 required')
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    output = args.output_dir.resolve();output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent/f'.{output.name}.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    source = args.base_frames_source.resolve()
    assert (source/'COMPLETE').exists()
    samples = args.samples_dir.resolve();check_splits(samples, args.test_fold)
    manifest = read_rows(samples/'samples_manifest.csv')
    train = sorted(ROOT/r['sample_path'] for r in manifest if r['split']=='train')
    paths = sorted((samples/'val').glob('*.npz'))
    assert train and paths
    for split,selected in (('train',train),('val',paths)):
        rows=[r for r in manifest if r['split']==split]
        assert len(rows)==len(selected)
        assert {r['family'] for r in rows}=={'doped','multicore','multimode','single_mode'}
        assert {r['spatial_band_policy'] for r in rows}=={'persistent-amplitude-v1'}
    validation_folds={int(r['fold']) for r in manifest if r['split']=='val'}
    assert len(validation_folds)==1 and args.test_fold not in validation_folds
    meta = validation_metadata(paths, samples/'samples_manifest.csv')
    pilot = stratified_validation(paths, meta, 2, args.seed)
    calibration = []
    rng = random.Random(args.seed)
    for family in ('doped', 'multicore', 'multimode', 'single_mode'):
        candidates = sorted(ROOT/r['sample_path'] for r in manifest if r['split']=='train' and r['family']==family)
        calibration.extend(rng.sample(candidates, 16))
    calibration.sort()
    base_path = args.base_checkpoint.resolve()
    source_protocol=json.loads((source/'protocol.json').read_text())
    assert source_protocol['test_fold']==args.test_fold
    assert source_protocol['checkpoints']['compact']['sha256']==digest(base_path)
    config = {'architecture': 'native-v1', 'base_channels': 32, 'embedding_dim': 128,
              'num_embeddings': 2048, 'downsample_levels': 4, 'commitment_cost': .25,
              'latent_geometry': 'standard', 'activation_checkpointing': True, 'exact_eval_quantization': True,
              'input_representation': 'normalized-signal-residual-v1', 'quantizer': args.quantizer}
    files = sorted((ROOT/'finetuning').glob('*.py'))+sorted((ROOT/'compression').glob('*.py'))
    files += sorted((ROOT.parent/'smartgrids/vqvae').glob('*.py'))
    protocol = {'kind': 'frozen_base_neural_signal_residual_v1', 'seed': args.seed, 'base_checkpoint': str(base_path),
                'base_sha256': digest(base_path), 'config': config, 'test_fold': args.test_fold, 'validation_fold':next(iter(validation_folds)),
                'initial_epochs': 16, 'max_epochs': 24, 'steps_per_epoch': 1024,
                'lr': 1e-4, 'weight_decay': 1e-5, 'precision': 'FP32 deterministic cuDNN',
                'loss': '0.5 MSE + 0.5 L1 on valid normalized residual cells + VQ',
                'residual_normalization': 'identity in original global-normalized coordinates; zero at escaped cells',
                'quantization_note': 'VQ has commitment/codebook loss; FSQ has fixed scalar bins and zero auxiliary VQ loss. Projection parameters replace the learned embedding table; not a pure optimizer-only change.',
                'selection': 'minimum guarded pilot macro RMSE; 2 windows per file/region',
                'extension': 'one 8-epoch extension if recent gain >=0.2% and guarded RMSE >=1% better than base',
                'calibration': [str(p) for p in calibration], 'pilot': [str(p) for p in pilot],
                'code_sha256': {str(p): digest(p) for p in files},
                'metadata_sha256': {n: digest(samples/n) for n in ('samples_manifest.csv', 'preprocessing.json', 'channel_maps.jsonl')},
                'source_frames_protocol_sha256': digest(source/'protocol.json'),
                'no_test_access': True, 'final_model_promotion': False,
                'torch': str(torch.__version__), 'cuda': torch.version.cuda}
    if args.resume:
        assert json.loads((output/'protocol.json').read_text())==protocol, 'Protocol changed'
        if (output/'COMPLETE').exists():return
    else:
        output.mkdir(exist_ok=False);json_file(output/'protocol.json', protocol)
        for path in files:
            relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('external_vqvae')/path.name
            dest = output/'code_snapshot'/relative;dest.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, dest)
    def verify():
        if shutil.disk_usage(output).free < 10*2**30:raise RuntimeError('Less than 10 GiB free; no automatic deletion')
        for path, expected in protocol['code_sha256'].items():assert digest(path)==expected, path
        for name, expected in protocol['metadata_sha256'].items():assert digest(samples/name)==expected, name
    try:
        verify()
        saved = torch.load(base_path, map_location='cpu', weights_only=False)
        base = build_model({**saved['config'], 'exact_eval_quantization': True}, saved, 'cuda').eval();del saved
        for p in base.parameters():p.requires_grad_(False)
        initial = output/'initial.pt'
        if not initial.exists():
            status(output, 'max_shape_smoke')
            torch.manual_seed(args.seed);refiner=new_refiner(config, 'cuda').train()
            smoke_optimizer=torch.optim.AdamW(refiner.parameters(), lr=1e-4)
            maximum=next(ROOT/r['sample_path'] for r in manifest if r['split']=='train' and int(r['channels'])==512)
            _, packet=read_window(maximum);_,mask,_,residual=residual_input(base,packet)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                smoke_optimizer.zero_grad(set_to_none=True)
                predicted,vq,indices,_=refiner(residual)
                loss=.5*F.mse_loss(predicted[mask],residual[mask])+.5*F.l1_loss(predicted[mask],residual[mask])+vq
                assert torch.isfinite(loss);loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in refiner.parameters() if p.grad is not None)
                torch.nn.utils.clip_grad_norm_(refiner.parameters(),1.);smoke_optimizer.step()
                assert indices.numel()==2016
                assert all(p.grad is None for p in base.parameters())
            json_file(output/'smoke.json', {'parameters':sum(p.numel() for p in refiner.parameters()),
                'base_indices_max':8064,'refinement_indices_max':2016,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'discarded_smoke_updates':True})
            del refiner,smoke_optimizer,predicted,vq,loss,residual,mask;gc.collect();torch.cuda.empty_cache()
            torch.manual_seed(args.seed);refiner=new_refiner(config,'cuda').eval()
            vectors=[];generator=torch.Generator().manual_seed(args.seed)
            if args.quantizer == 'vq':
                with torch.no_grad():
                    for path in calibration:
                        _,packet=read_window(path);_,_,_,residual=residual_input(base,packet)
                        z=refiner.encoder(F.pad(residual,(0,(-residual.shape[-1])%16,0,(-residual.shape[-2])%16)))
                        flat=z.permute(0,2,3,1).reshape(-1,128).cpu()
                        vectors.append(flat[torch.randperm(len(flat),generator=generator)[:256]])
                    vectors=torch.cat(vectors);assert len(vectors)>=2048
                    refiner.quantizer.embedding.weight.copy_(vectors[torch.randperm(len(vectors),generator=generator)[:2048]].cuda())
            save_checkpoint({'config':config,'model_state_dict':refiner.cpu().state_dict(),'epoch':0,
                'pipeline':{'role':'normalized-signal-residual-v1','base_sha256':protocol['base_sha256']}},initial)
            del refiner,vectors;gc.collect();torch.cuda.empty_cache()
        saved=torch.load(initial,map_location='cpu',weights_only=False)
        refiner=load_refiner(saved,'cuda');del saved
        optimizer=torch.optim.AdamW(refiner.parameters(),lr=1e-4,weight_decay=1e-5)
        sampling=random.Random(args.seed);torch.manual_seed(args.seed)
        completed=0;best=float('inf')
        last=output/'last.pt'
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False)
            refiner.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict'])
            completed=saved['epoch'];best=saved['best_guarded_rmse'];sampling.setstate(saved['sampling_rng'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
            assert int(read_rows(output/'metrics.csv')[-1]['epoch'])==completed
        def checkpoint(epoch):
            return {'config':config,'model_state_dict':refiner.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'epoch':epoch,'best_guarded_rmse':best,'sampling_rng':sampling.getstate(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'pipeline':{'role':'normalized-signal-residual-v1','base_sha256':protocol['base_sha256'],
                                'source_side':'base normalization only','residual_side':'none'}}
        targets=[16]
        while targets:
            target=targets.pop(0)
            while completed<target:
                verify();epoch=completed+1;status(output,'training',epoch=epoch,target_epochs=target)
                refiner.train();started=time.monotonic();losses=[];temporal_losses=[];vq_losses=[];codes=set()
                for _ in range(1024):
                    path=sampling.choice(train);_,packet=read_window(path)
                    _,mask,_,residual=residual_input(base,packet)
                    optimizer.zero_grad(set_to_none=True)
                    predicted,vq,indices,_=refiner(residual)
                    temporal=.5*F.mse_loss(predicted[mask],residual[mask])+.5*F.l1_loss(predicted[mask],residual[mask])
                    loss=temporal+vq
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite residual loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(refiner.parameters(),1.);optimizer.step()
                    losses.append(float(loss.detach()));codes.update(indices.detach().cpu().tolist())
                    temporal_losses.append(float(temporal.detach()));vq_losses.append(float(vq.detach()))
                optimizer.zero_grad(set_to_none=True)
                evaluated=validation(base,refiner,pilot,meta)
                improved=evaluated['guarded']['macro_relative_rmse']<best
                if improved:best=evaluated['guarded']['macro_relative_rmse']
                row={'epoch':epoch,'train_loss':float(np.mean(losses)),'train_codes':len(codes),
                     'train_temporal_loss':float(np.mean(temporal_losses)), 'train_quantizer_aux_loss':float(np.mean(vq_losses)),
                     'base_rmse':evaluated['base']['macro_relative_rmse'],
                     'all_refined_rmse':evaluated['all_refined']['macro_relative_rmse'],
                     'macro_relative_rmse':evaluated['guarded']['macro_relative_rmse'],
                     'macro_correlation':evaluated['guarded']['macro_correlation'],
                     'accepted_windows':evaluated['accepted_windows'],'elapsed_sec':time.monotonic()-started}
                append_csv(output/'metrics.csv',row)
                save_checkpoint(checkpoint(epoch),last)
                if improved:shutil.copy2(last,output/'best_macro.pt')
                completed=epoch;print(json.dumps(row),flush=True)
            retained=output/f'retained_phase{target}.pt'
            if not retained.exists():shutil.copy2(output/'best_macro.pt',retained)
            status(output,'full_validation',phase=target)
            full_validation(base,retained,source,samples,meta,calibration,output/f'full_phase{target}')
            if target==16:
                ledger=output/'extension_decision.json'
                if ledger.exists():decision=json.loads(ledger.read_text())
                else:
                    history=read_rows(output/'metrics.csv');decision=extension_decision(history)
                    improvement=1-min(float(r['macro_relative_rmse']) for r in history)/float(history[-1]['base_rmse'])
                    decision['best_guarded_gain_vs_base']=improvement
                    decision['extend']=decision['extend'] and improvement>=.01
                    json_file(ledger,decision)
                if decision['extend']:targets.append(24)
        shutil.copy2(output/f'full_phase{completed}'/'REPORT.md',output/'REPORT.md')
        (output/'COMPLETE').write_text('Bounded residual refinement and complete stream validation finished; no promotion.\n')
        status(output,'complete',final_model_promoted=False)
    except BaseException as error:
        status(output,'failed',error=repr(error));(output/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
