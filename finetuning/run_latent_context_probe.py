#!/usr/bin/env python3
"""Paired train-only controls for known-shape and richer decoded-token context."""
import argparse
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from finetuning.finetune_vqvae import save_checkpoint
from finetuning.latent_context_adapter import KINDS, LatentContextAdapter
from finetuning.model_factory import build_model
from finetuning.representation_study import digest, read_manifest, pilot_rows, load_packet, distortion, write_json
from finetuning.run_decoder_aware_codebooks import source_state_hash
from finetuning.run_index_search import nearest, vectors
from finetuning.run_spatial_output_study import split_train_files
from finetuning.run_untied_synthesis_probe import weighted_macro_distortion
from finetuning.update_schedule import load_independent_optimizer_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-protocol',type=Path,required=True)
    p.add_argument('--samples-dir',type=Path,required=True)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expose exactly one remote RTX4090')
    args.output_dir.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.output_dir.parent/'.lr_schedule_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.backends.mha.set_fastpath_enabled(False)
    sys.path.insert(0,str(args.repo_root.resolve()))
    saved=torch.load(args.source,map_location='cpu',weights_only=False)
    source_protocol=json.loads(args.source_protocol.read_text())
    if (saved.get('epoch')!=24 or saved.get('initialization',{}).get('arm')!='A'
            or saved['config']['latent_geometry']!='spatial-4x'
            or saved['protocol_sha256']!=digest(args.source_protocol)
            or source_protocol['metadata_sha256']['samples_manifest.csv']!=digest(args.samples_dir/'samples_manifest.csv')):
        raise ValueError('Frozen original transferred source/corpus lineage differs')
    seed=2102;torch.manual_seed(seed)
    model=build_model(saved['config'],saved,'cuda').eval().requires_grad_(False);del saved
    base_hash=source_state_hash(model)
    splits=read_manifest(args.samples_dir)
    pilots=pilot_rows(splits['train'],seed=2100)
    _,held=split_train_files(pilots,seed=2100)
    checks=[pilots[i] for i in held];held_ids={r['drive_id'] for r in checks}
    fit=[r for r in splits['train'] if r['drive_id'] not in held_ids]
    rng=random.Random(seed)
    initial,maximum,interval=(2,2,1) if args.smoke else (512,2048,128)
    sequence=[fit[rng.randrange(len(fit))] for _ in range(maximum)]
    if args.smoke:
        sequence[0]=max(fit,key=lambda r:int(r['channels']))
        checks=[max(checks,key=lambda r:int(r['channels']))]
    used={r['sample_path']:r for r in sequence+checks}
    code=['finetuning/run_latent_context_probe.py','finetuning/latent_context_adapter.py',
          'finetuning/run_untied_synthesis_probe.py','compression/untied_synthesis.py',
          'finetuning/run_decoder_aware_codebooks.py','finetuning/run_index_search.py',
          'finetuning/run_spatial_output_study.py','finetuning/representation_study.py',
          'finetuning/model_factory.py','finetuning/latent_geometry.py','finetuning/robust_escape.py',
          'finetuning/reversible_normalization.py','finetuning/update_schedule.py']
    counts={}
    for kind in KINDS:
        counts[kind]=sum(parameter.numel() for parameter in LatentContextAdapter(kind).parameters())
    if max(counts.values())/min(counts.values())>=1.01:raise ValueError('Control capacities differ by >=1%')
    protocol={'kind':'decoded_latent_context_probe_v1','seed':seed,'smoke_only':args.smoke,
        'source_sha256':digest(args.source),'source_state_sha256':base_hash,
        'source_protocol_sha256':digest(args.source_protocol),'manifest_sha256':digest(args.samples_dir/'samples_manifest.csv'),
        'arms':list(KINDS),'trainable_parameters':counts,'initial_budget_per_arm':initial,'maximum_budget_per_arm':maximum,
        'check_interval':interval,'sequence':[r['sample_path'] for r in sequence],'checks':[r['sample_path'] for r in checks],
        'input_sha256':{name:digest(ROOT/name) for name in sorted(used)},
        'code_sha256':{name:digest(ROOT/name) for name in code},
        'optimizer':'Fresh Adam lr=1e-3 clip1 no weight decay, same batches, FP32 TF32=false, deterministic math attention',
        'inputs':'Only fixed original quantized code vectors; shape/coords derived from already transmitted H/W and latent grid',
        'freeze':'Original encoder, analysis codebook, all base decoder parameters and analysis IDs frozen',
        'identity':'All residual output projections zero-initialized; all arms initially reproduce source',
        'control':'Local pointwise MLP versus shape-aware pointwise versus pooled per-channel moments versus axial content attention; parameter counts within 1%',
        'groupnorm_note':'Base GroupNorm already carries global mean/variance across space/time. This is richer shape/content conditioning, not the first global dependency.',
        'selection':'Cell-weight normalized D within config, equal configs/families; separate train-file checks; initial source eligible',
        'extension':'Once, all arms to2048 if best arm gains >=0.5% vs source, >=0.2% recent; plus >=0.25% over local for contextual winner OR >=1% over source if local winner',
        'advance':'Best normalized D must improve >=1% vs source; context advantage claimed only if >=0.5% over local. Full future codec gate remains >=5% physical relative RMSE with unchanged complete bytes/guards',
        'extra_shared_model_bytes':{k:4*v for k,v in counts.items()},
        'limitations':['No original signal, continuous encoder feature, family/file ID or residual enters decoder adapter',
                       'File-ID checks are not acquisition-independent and were seen by the historical source pretraining',
                       'Normalized D improvements are not physical RMSE percentages; fixed IDs alone do not prove complete stream equality'],
        'validation_signal_windows_read':0,'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'torch':str(torch.__version__)}
    out=args.output_dir
    if args.resume:
        if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Context probe resume changed')
        if (out/'COMPLETE').exists():return
    else:
        out.mkdir(exist_ok=False);write_json(out/'protocol.json',protocol)
        for name in code:
            path=out/'code_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    cache={}
    with torch.no_grad():
        for name,row in sorted(used.items()):
            if row['split']!='train' or digest(ROOT/name)!=protocol['input_sha256'][name]:raise ValueError('Train-only input changed')
            raw,packet=load_packet(row);target=torch.from_numpy(packet.values)[None,None].cuda()
            h,w=raw.shape;z=model.encoder(F.pad(target,(0,(-w)%16,0,(-h)%16)))
            ids=nearest(z,model.quantizer.embedding.weight)
            cache[name]=(ids.cpu(),tuple(z.shape[-2:]),target.cpu(),torch.from_numpy(packet.valid)[None,None],raw.size)

    def prediction(entry,adapter=None,training=False):
        ids,shape,target,_,_=entry
        q=vectors(ids.cuda(),model.quantizer.embedding.weight,shape)
        if adapter is not None:q=adapter(q,tuple(target.shape[-2:]))
        output=checkpoint(model.decoder,q,use_reentrant=False) if training else model.decoder(q)
        return output[...,:target.shape[-2],:target.shape[-1]]

    def evaluate(adapter=None):
        if adapter is not None:adapter.eval()
        values=[]
        with torch.no_grad():
            for row in checks:
                entry=cache[row['sample_path']];_,_,target,valid,cells=entry
                value=float(distortion(prediction(entry,adapter),target.cuda(),valid.cuda()))
                if not np.isfinite(value):raise RuntimeError('Nonfinite context selection')
                values.append((row['family'],row['configuration'],value,cells))
        return weighted_macro_distortion(values)

    baseline=evaluate()
    if baseline<=0:raise ValueError('No positive baseline check distortion')

    def train(kind,budget):
        folder=out/kind;folder.mkdir(exist_ok=True);torch.manual_seed(seed)
        adapter=LatentContextAdapter(kind).cuda();optimizer=torch.optim.Adam(adapter.parameters(),lr=1e-3)
        completed,best,best_step=0,baseline,0
        best_state={name:value.detach().cpu().clone() for name,value in adapter.state_dict().items()}
        history=[{'step':0,'check_D':baseline}];last=folder/'last.pt'
        if last.exists():
            state=torch.load(last,map_location='cpu',weights_only=False)
            if state['protocol_sha256']!=digest(out/'protocol.json'):raise ValueError('Adapter lineage changed')
            adapter.load_state_dict(state['adapter']);load_independent_optimizer_state(optimizer,state['optimizer'])
            completed,best,best_step=state['step'],state['best_D'],state['best_step']
            best_state,history=state['best_state'],state['history']
            if {float(value['step']) for value in optimizer.state.values()}!={float(completed)}:
                raise ValueError('Restored adapter optimizer counter mismatch')
        began=time.perf_counter()
        for step in range(completed+1,budget+1):
            if (step-1)%interval==0 and any(digest(ROOT/name)!=value for name,value in protocol['code_sha256'].items()):
                raise ValueError('Frozen context code changed')
            adapter.train();entry=cache[sequence[step-1]['sample_path']]
            optimizer.zero_grad(set_to_none=True)
            loss=distortion(prediction(entry,adapter,training=True),entry[2].cuda(),entry[3].cuda())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite context training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
            if any(parameter.grad is not None for parameter in model.parameters()):raise AssertionError('Source received adapter gradients')
            optimizer.step()
            if {float(value['step']) for value in optimizer.state.values()}!={float(step)}:raise ValueError('Adapter Adam counter mismatch')
            if step%interval==0 or step==budget:
                value=evaluate(adapter);history.append({'step':step,'check_D':value})
                if value<best:
                    best,best_step=value,step;best_state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
                save_checkpoint({'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                    'best_D':best,'best_step':best_step,'best_state':best_state,'history':history,
                    'protocol_sha256':digest(out/'protocol.json')},last)
                write_json(folder/'history.json',history)
                print(f'context {kind} step={step} D={value:.8f} best={best:.8f}',flush=True)
        if source_state_hash(model)!=base_hash:raise AssertionError('Source weights changed')
        save_checkpoint({'kind':kind,'adapter_state':best_state,'source_sha256':protocol['source_sha256']},folder/'selected.pt')
        result={'best_D':best,'best_step':best_step,'updates':max(completed,budget),'history':history,
                'seconds_this_invocation':time.perf_counter()-began}
        write_json(folder/'selection.json',result);del adapter,optimizer
        return result

    selections={kind:train(kind,initial) for kind in KINDS}
    extension=out/'EXTENSION.json'
    if extension.exists():decision=json.loads(extension.read_text())
    else:
        winner=min(KINDS,key=lambda k:selections[k]['best_D']);item=selections[winner]
        values={r['step']:r['check_D'] for r in item['history']}
        recent=(1-values[512]/values[256]) if not args.smoke and values[256]>0 else 0.
        structural=(item['best_D']<=.9975*selections['local']['best_D'] if winner!='local' else item['best_D']<=.99*baseline)
        decision={'extend_all':not args.smoke and item['best_D']<=.995*baseline and recent>=.002 and structural,
                  'winner':winner,'recent_gain':recent,'initial_selection':selections}
        write_json(extension,decision)
    if decision['extend_all']:selections={kind:train(kind,maximum) for kind in KINDS}
    winner=min(KINDS,key=lambda k:selections[k]['best_D']);best=selections[winner]['best_D']
    source_gain=100*(1-best/baseline);local_gain=100*(1-best/selections['local']['best_D'])
    write_json(out/'results.json',{'baseline_check_D':baseline,'selection':selections,'best_arm':winner,
        'best_gain_vs_source_pct':source_gain,'best_gain_vs_local_pct':local_gain,
        'advance_gate_met':not args.smoke and source_gain>=1.,
        'context_advantage_supported':not args.smoke and winner!='local' and local_gain>=.5,
        'source_weights_unchanged':source_state_hash(model)==base_hash,'analysis_assignments_fixed':True,
        'validation_signal_windows_read':0,'model_promoted':False,'smoke_only':args.smoke})
    if source_state_hash(model)!=base_hash or digest(args.source)!=protocol['source_sha256']:raise AssertionError('Source lineage changed')
    (out/'COMPLETE').touch();lock.close()


if __name__=='__main__':main()
