#!/usr/bin/env python3
"""Train-only fixed-ID synthesis lookup probe; never change analysis assignments."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression.untied_synthesis import SynthesisLookup
from finetuning.finetune_vqvae import save_checkpoint
from finetuning.model_factory import build_model
from finetuning.representation_study import digest, read_manifest, pilot_rows, load_packet, distortion, write_json
from finetuning.run_decoder_aware_codebooks import source_state_hash
from finetuning.run_index_search import nearest
from finetuning.run_spatial_output_study import split_train_files
from finetuning.update_schedule import load_independent_optimizer_state


def weighted_macro_distortion(rows):
    groups = defaultdict(list)
    for family, configuration, value, cells in rows:
        groups[family, configuration].append((value, cells))
    families = defaultdict(list)
    for (family, _), values in groups.items():
        families[family].append(float(np.average([v for v, _ in values], weights=[w for _, w in values])))
    return float(np.mean([np.mean(values) for values in families.values()]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-protocol', type=Path, required=True)
    p.add_argument('--samples-dir', type=Path, required=True)
    p.add_argument('--repo-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090')
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.output_dir.parent/'.lr_schedule_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(args.repo_root.resolve()))
    seed = 2100
    splits = read_manifest(args.samples_dir)
    pilots = pilot_rows(splits['train'], seed=seed)
    fit_ids, check_ids = split_train_files(pilots, seed=seed)
    checks = [pilots[i] for i in check_ids]
    held_files = {r['drive_id'] for r in checks}
    fit = [r for r in splits['train'] if r['drive_id'] not in held_files]
    assert not held_files & {r['drive_id'] for r in fit}
    rng = random.Random(seed)
    initial_budget, maximum, interval = (2, 2, 1) if args.smoke else (256, 1024, 64)
    sequence = [fit[rng.randrange(len(fit))] for _ in range(maximum)]
    if args.smoke:
        sequence[0] = max(fit, key=lambda r: int(r['channels']))
        checks = [max(checks, key=lambda r: int(r['channels']))]
    saved = torch.load(args.source, map_location='cpu', weights_only=False)
    if saved.get('initialization',{}).get('arm')!='A' or saved.get('epoch')!=24:
        raise ValueError('Use the retained transferred A epoch24 reference for this bounded probe')
    source_protocol=json.loads(args.source_protocol.read_text())
    if (saved['protocol_sha256']!=digest(args.source_protocol)
            or digest(args.samples_dir/'samples_manifest.csv')!=source_protocol['metadata_sha256']['samples_manifest.csv']):
        raise ValueError('Source checkpoint/corpus protocol differs')
    if (saved['config']['num_embeddings'] != 2048 or saved['config']['embedding_dim']!=128
            or saved['config']['latent_geometry'] != 'spatial-4x'):
        raise ValueError('Use the original K2048 compact VQ source, not a reduced PQ model')
    if Path(saved['finetune']['samples_dir']).name != args.samples_dir.name:
        raise ValueError('Frozen source belongs to another target cohort')
    torch.manual_seed(seed)
    model = build_model(saved['config'], saved, 'cuda').eval().requires_grad_(False)
    del saved
    base_hash = source_state_hash(model)
    used = {r['sample_path']: r for r in sequence + checks}
    code = ['finetuning/run_untied_synthesis_probe.py', 'compression/untied_synthesis.py',
            'finetuning/run_decoder_aware_codebooks.py', 'finetuning/run_index_search.py',
            'finetuning/model_factory.py', 'finetuning/representation_study.py', 'finetuning/update_schedule.py',
            'finetuning/latent_geometry.py','finetuning/robust_escape.py','finetuning/reversible_normalization.py',
            'finetuning/run_spatial_output_study.py']
    protocol = {'kind':'fixed_analysis_untied_synthesis_probe_v1','seed':seed,'smoke_only':args.smoke,
        'source_sha256':digest(args.source),'source_state_sha256':base_hash,
        'source_protocol_sha256':digest(args.source_protocol),
        'manifest_sha256':digest(args.samples_dir/'samples_manifest.csv'),
        'initial_budget_per_arm':initial_budget,'maximum_budget_per_arm':maximum,'check_interval':interval,
        'sequence':[r['sample_path'] for r in sequence],'checks':[r['sample_path'] for r in checks],
        'input_sha256':{path:digest(ROOT/path) for path in sorted(used)},
        'code_sha256':{name:digest(ROOT/name) for name in code},
        'arms':['latent','waveform'],'trainable':'Only separate 2048x128 synthesis table, initialized from original analysis book',
        'extra_shared_synthesis_tensor_bytes':2048*128*4,
        'optimizer':'Fresh Adam lr=1e-3, clip1, no weight decay, same batches, FP32 TF32=false',
        'assignment':'Original encoder and original analysis book frozen; IDs computed once and cached, never reassigned',
        'objective':'Latent control: MSE to frozen encoder features; waveform: 0.5 normalized MSE+0.5 L1 through frozen decoder',
        'selection':'Cell-weighted normalized D within config, equal configs/families, separate train-file checks, initial table eligible',
        'extension':'Once, both arms to 1024 if waveform improves >=0.5% vs source, >=0.1% vs latent control, and >=0.2% from step128 to256',
        'advance':'Only consider real-wire/development confirmation if waveform D improves >=1% against original AND latent control; full structural target remains >=5% relative RMSE with unchanged bytes/physical guards',
        'cost_note':'Fixed IDs do not prove complete-stream equality; extra synthesis table is a shared model cost and must be charged in any codec confirmation',
        'prior_difference':'Earlier decoder-aware PQ used reduced 2x32 codebooks and recomputed nearest assignments; this uses original 2048 IDs and does not move the assignment dictionary',
        'validation_signal_windows_read':0,'final_holdout_accessed':False,
        'limitations':['Source model historically saw these training groups, including internal checks',
                       'File-ID split is not proof of independent acquisitions or absence of content aliases',
                       'Normalized D gains are not physical RMSE percentages or codec acceptance'],
        'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'torch':str(torch.__version__)}
    protocol['deterministic_algorithms']=True
    protocol['cublas_workspace_config']=os.environ['CUBLAS_WORKSPACE_CONFIG']
    out = args.output_dir
    if args.resume:
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Synthesis probe resume protocol changed')
        if (out/'COMPLETE').exists():
            return
    else:
        out.mkdir(parents=True,exist_ok=False);write_json(out/'protocol.json',protocol)
        for name in code:
            dest=out/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    cache={}
    with torch.no_grad():
        for path,row in sorted(used.items()):
            if row['split']!='train' or digest(ROOT/path)!=protocol['input_sha256'][path]:
                raise ValueError('Only unchanged training inputs may enter this probe')
            raw,packet=load_packet(row)
            x=torch.from_numpy(packet.values)[None,None].cuda()
            h,w=raw.shape
            z=model.encoder(F.pad(x,(0,(-w)%16,0,(-h)%16)))
            ids=nearest(z,model.quantizer.embedding.weight)
            cache[path]=(z.cpu(),ids.cpu(),tuple(z.shape[-2:]),x.cpu(),torch.from_numpy(packet.valid)[None,None],raw.size)

    def evaluate(lookup):
        values=[]
        with torch.no_grad():
            for row in checks:
                _,ids,shape,target,valid,cells=cache[row['sample_path']]
                target,valid=target.cuda(),valid.cuda()
                q=lookup(ids.cuda(),shape)
                predicted=model.decoder(q)[...,:target.shape[-2],:target.shape[-1]]
                value=float(distortion(predicted,target,valid))
                if not np.isfinite(value):raise RuntimeError('Nonfinite synthesis selection score')
                values.append((row['family'],row['configuration'],value,cells))
        return weighted_macro_distortion(values)

    original= SynthesisLookup(model.quantizer.embedding.weight)
    baseline=evaluate(original);del original
    if baseline<=0:raise ValueError('No positive check distortion to improve')

    def train(arm,budget):
        folder=out/arm;folder.mkdir(exist_ok=True)
        lookup=SynthesisLookup(model.quantizer.embedding.weight)
        optimizer=torch.optim.Adam([lookup.weight],lr=1e-3)
        last=folder/'last.pt'
        best,best_step,completed=baseline,0,0
        history=[{'step':0,'check_D':baseline}]
        best_weight=lookup.weight.detach().cpu().clone()
        if last.exists():
            state=torch.load(last,map_location='cpu',weights_only=False)
            if state['protocol_sha256']!=digest(out/'protocol.json'):
                raise ValueError('Lookup optimizer/source lineage differs')
            lookup.weight.data.copy_(state['weight'].cuda());load_independent_optimizer_state(optimizer,state['optimizer'])
            completed,best,best_step=state['step'],state['best_D'],state['best_step']
            best_weight,history=state['best_weight'],state['history']
            if {float(value['step']) for value in optimizer.state.values()}!={float(completed)}:
                raise ValueError('Lookup optimizer counter differs')
        began=time.perf_counter()
        for step in range(completed+1,budget+1):
            row=sequence[step-1];z,ids,shape,target,valid,_=cache[row['sample_path']]
            if any(digest(ROOT/name)!=expected for name,expected in protocol['code_sha256'].items()):
                raise ValueError('Frozen synthesis probe code changed')
            optimizer.zero_grad(set_to_none=True)
            q=lookup(ids.cuda(),shape)
            if arm=='latent':loss=F.mse_loss(q,z.cuda())
            else:
                predicted=checkpoint(model.decoder,q,use_reentrant=False)[...,:target.shape[-2],:target.shape[-1]]
                loss=distortion(predicted,target.cuda(),valid.cuda())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite synthesis loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([lookup.weight],1.,error_if_nonfinite=True)
            if any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen source received gradients')
            optimizer.step()
            if {float(value['step']) for value in optimizer.state.values()}!={float(step)}:
                raise ValueError('Lookup optimizer counter mismatch')
            if step%interval==0 or step==budget:
                value=evaluate(lookup);history.append({'step':step,'check_D':value})
                if value<best:best,best_step,best_weight=value,step,lookup.weight.detach().cpu().clone()
                save_checkpoint({'weight':lookup.weight.detach().cpu().clone(),'step':step,'optimizer':optimizer.state_dict(),
                    'protocol_sha256':digest(out/'protocol.json'),
                    'best_D':best,'best_step':best_step,'best_weight':best_weight,'history':history},last)
                write_json(folder/'history.json',history)
                print(f'synthesis {arm} step={step} D={value:.8f} best={best:.8f}',flush=True)
        if source_state_hash(model)!=base_hash:raise AssertionError('Source analysis or decoder weights changed')
        np.save(folder/'selected.npy',best_weight.numpy())
        result={'best_D':best,'best_step':best_step,'updates':max(completed,budget),'history':history,
                'seconds_this_invocation':time.perf_counter()-began,'synthesis_parameters':lookup.weight.numel()}
        write_json(folder/'selection.json',result)
        del optimizer,lookup
        return result

    selections={arm:train(arm,initial_budget) for arm in ('latent','waveform')}
    extension=out/'EXTENSION.json'
    if extension.exists():
        decision=json.loads(extension.read_text())
    else:
        wave,latent=selections['waveform'],selections['latent']
        values={r['step']:r['check_D'] for r in wave['history']}
        recent=1-values[256]/values[128] if not args.smoke and values[128]>0 else 0.
        extend=(not args.smoke and wave['best_D']<=.995*baseline and wave['best_D']<=.999*latent['best_D'] and recent>=.002)
        decision={'extend_both':extend,'recent_gain':recent,'phase256_selection':selections}
        write_json(extension,decision)
    if decision['extend_both']:
        selections={arm:train(arm,maximum) for arm in ('latent','waveform')}
    wave,latent=selections['waveform'],selections['latent']
    advance=(not args.smoke and wave['best_D']<=.99*baseline and wave['best_D']<=.99*latent['best_D'])
    if source_state_hash(model)!=base_hash or digest(args.source)!=protocol['source_sha256']:
        raise AssertionError('Frozen source lineage changed')
    write_json(out/'results.json',{'baseline_check_D':baseline,'selection':selections,
        'waveform_gain_vs_source_pct':100*(1-wave['best_D']/baseline),
        'waveform_gain_vs_latent_pct':100*(1-wave['best_D']/latent['best_D']),
        'advance_gate_met':advance,'validation_signal_windows_read':0,'source_weights_unchanged':True,
        'analysis_assignments_fixed':True,'model_promoted':False,'smoke_only':args.smoke})
    (out/'COMPLETE').touch()
    lock.close()


if __name__=='__main__':main()
