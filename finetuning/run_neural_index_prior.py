#!/usr/bin/env python3
"""Small two-pass entropy prior for frozen VQ indices; no waveform updates."""
import argparse
from collections import defaultdict
import csv
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'smartgrids'))
from finetuning.checkerboard_prior import CheckerboardPrior,anchor_mask
from compression.neural_symbol_stream import NeuralSymbolCodec
from compression.symbol_stream import SymbolCodec,HEADER
from compression.study_bundle import write_bundle,read_bundle
from finetuning.model_factory import build_model
from finetuning.robust_escape import encode_robust
from finetuning.run_frozen_entropy_roi import encoder_indices
from finetuning.reconstruction_objective import stratified_validation,validation_metadata
from finetuning.run_sampling_experiment import check_splits
from finetuning.run_spatial_abc_study import status
from finetuning.finetune_vqvae import save_checkpoint


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def json_file(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def load_grid(path):
    with np.load(path,allow_pickle=False) as packet:return packet['indices'].astype(np.int64)


def conditional_bits(model,paths):
    model.eval();bits=total=0
    with torch.no_grad():
        for path in paths:
            grid=load_grid(path);tensor=torch.from_numpy(grid)[None].cuda()
            mask=~anchor_mask(*grid.shape,'cuda')
            if not mask.any():continue
            logits=model(tensor)[0,:,mask].T
            bits+=float(F.cross_entropy(logits,tensor[0][mask],reduction='sum'))/math.log(2)
            total+=int(mask.sum())
    return bits/max(total,1)


def run_cohort(label,source,samples,output,seed):
    if (output/'COMPLETE').exists():return json.loads((output/'results.json').read_text())
    output.mkdir(exist_ok=True)
    source_protocol=json.loads((source/'protocol.json').read_text())
    check_splits(samples,source_protocol['test_fold'])
    for name,expected_metadata in source_protocol['metadata_sha256'].items():
        assert digest(samples/name)==expected_metadata,name
    with (samples/'samples_manifest.csv').open() as stream:manifest=list(csv.DictReader(stream))
    waveform=Path(source_protocol['checkpoints']['compact']['path'])
    expected=source_protocol['checkpoints']['compact']['sha256'];assert digest(waveform)==expected
    provenance={'source':str(source),'source_protocol_sha256':digest(source/'protocol.json'),
        'source_results_sha256':digest(source/'results.json'),'samples_dir':str(samples),
        'waveform_sha256':expected,'seed':seed,'metadata_sha256':source_protocol['metadata_sha256']}
    if (output/'provenance.json').exists():assert json.loads((output/'provenance.json').read_text())==provenance
    else:json_file(output/'provenance.json',provenance)
    huffman=SymbolCodec.from_bytes((source/'compact.tables.json').read_bytes())
    assert huffman.spec['checkpoint_sha256']==expected
    saved=torch.load(waveform,map_location='cpu',weights_only=False)
    wave=build_model({**saved['config'],'exact_eval_quantization':True},saved,'cuda').eval();del saved
    waveform_weight_bytes=4*sum(p.numel() for p in wave.parameters())
    counts=np.zeros(2048,dtype=np.int64)
    train=[]
    status(output,'extract_training_indices',cohort=label)
    with torch.no_grad():
        for number,row in enumerate(r for r in manifest if r['split']=='train'):
            path=ROOT/row['sample_path']
            target=output/'tokens/train'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
            source_hash=digest(path)
            if target.exists():
                with np.load(target,allow_pickle=False) as cached:
                    assert str(cached['source_sha256'])==source_hash
                    indices=cached['indices'].reshape(-1)
            else:
                with np.load(path,allow_pickle=False) as sample:
                    normalized=encode_robust(np.asarray(sample['data'],dtype=np.float32))
                indices,latent=encoder_indices(wave,normalized.values)
                temporary=target.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,indices=indices.reshape(latent),source_sha256=source_hash)
                temporary.replace(target)
            counts+=np.bincount(indices,minlength=2048);train.append(target)
            if number%1000==0:status(output,'extract_training_indices',completed=number,cohort=label)
    del wave;gc.collect();torch.cuda.empty_cache()
    validation=[];source_val=[]
    for row in manifest:
        if row['split']!='val':continue
        original=ROOT/row['sample_path'];source_val.append(original)
        indices,_,latent,_=huffman.decode((source/'frames/compact'/f'{original.stem}.fsc').read_bytes())
        target=output/'tokens/val'/original.name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():np.savez_compressed(target,indices=indices.reshape(latent))
        else:np.testing.assert_array_equal(load_grid(target).reshape(-1),indices)
        validation.append(target)
    meta=validation_metadata(source_val,samples/'samples_manifest.csv')
    pilot_original=stratified_validation(source_val,meta,2,seed)
    pilot=[output/'tokens/val'/path.name for path in pilot_original]
    marginal=((counts+1)/(counts.sum()+2048)).astype(np.float32)
    np.savez_compressed(output/'marginal.npz',probabilities=marginal)
    np.savez_compressed(output/'train_counts.npz',counts=counts)
    torch.manual_seed(seed);model=CheckerboardPrior(marginal).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    rng=random.Random(seed)
    initial=conditional_bits(model,pilot);best=initial
    best_packet={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                 'epoch':0,'conditional_bits':initial}
    history=[];targets=[8];completed=0
    last=output/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict'])
        rng.setstate(saved['sampling_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history=saved['history'];completed=saved['epoch'];best=saved['best'];best_packet=saved['best_packet'];del saved
    save_checkpoint(best_packet,output/'best.pt')
    while targets:
        target=targets.pop(0)
        while completed<target:
            model.train();started=time.monotonic();losses=[]
            status(output,'training_index_prior',epoch=completed+1,target_epochs=target,cohort=label)
            for _ in range(512):
                grid=load_grid(rng.choice(train));tensor=torch.from_numpy(grid)[None].cuda()
                mask=~anchor_mask(*grid.shape,'cuda')
                optimizer.zero_grad(set_to_none=True)
                logits=model(tensor)[0,:,mask].T
                loss=F.cross_entropy(logits,tensor[0][mask])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite index-prior loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach())/math.log(2))
            value=conditional_bits(model,pilot);completed+=1
            row={'epoch':completed,'train_conditional_bits':float(np.mean(losses)),
                 'pilot_conditional_bits':value,'elapsed_sec':time.monotonic()-started}
            history.append(row);json_file(output/'history.json',history)
            if value<best:
                best=value
                best_packet={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                             'epoch':completed,'conditional_bits':value}
                save_checkpoint(best_packet,output/'best.pt')
            save_checkpoint({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'sampling_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'history':history,'epoch':completed,'best':best,'best_packet':best_packet},last)
            print(label,json.dumps(row),flush=True)
        if target==8:
            ledger=output/'extension.json'
            if ledger.exists():extend=json.loads(ledger.read_text())['extend']
            else:
                previous=min(r['pilot_conditional_bits'] for r in history[-4:-2])
                recent=min(r['pilot_conditional_bits'] for r in history[-2:])
                improvement=(previous-recent)/max(previous,1e-12)
                # Only half the symbols use this predictor; this is a proxy, not actual bytes.
                estimated_total_gain=.5*(initial-best)/max(initial,1e-12)
                extend=improvement>=.005 and estimated_total_gain>=.05
                json_file(ledger,{'extend':extend,'recent_gain':improvement,
                    'estimated_total_gain':estimated_total_gain,'extra_epochs':8 if extend else 0})
            if extend:targets.append(16)
    saved=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    del optimizer,model;gc.collect();torch.cuda.empty_cache()
    prior=CheckerboardPrior(marginal);prior.load_state_dict(saved['state_dict']);prior.eval()
    inference=output/'prior_inference.pt'
    inference_packet={'state_dict':prior.state_dict(),'epoch':saved['epoch'],
                      'architecture':'checkerboard-16-48-48-2048-v1','waveform_sha256':expected}
    if inference.exists():
        existing=torch.load(inference,map_location='cpu',weights_only=False)
        assert existing['epoch']==saved['epoch']
        for key,value in prior.state_dict().items():torch.testing.assert_close(existing['state_dict'][key],value,atol=0,rtol=0)
    else:save_checkpoint(inference_packet,inference)
    descriptor={'kind':'checkerboard_range_cpu_v1','huffman_table_sha256':huffman.identifier.hex(),
        'waveform_sha256':expected,'prior_sha256':digest(inference),'torch':str(torch.__version__),
        'constriction':importlib.metadata.version('constriction'),'probability_reference':'CPU FP32, one thread, deterministic',
        'platform':platform.machine(),'marginal_sha256':hashlib.sha256(marginal.tobytes()).hexdigest()}
    codec=NeuralSymbolCodec(huffman,marginal,descriptor,prior)
    (output/'entropy_model.json').write_bytes(codec.serialized)
    torch.set_num_threads(1)
    common={name:(source/name).read_bytes() for name in
            ('validation_metadata.csv','channel_maps.jsonl','preprocessing_policy.json')}
    common['huffman.tables.json']=huffman.serialized
    common['marginal.npz']=(output/'marginal.npz').read_bytes()
    common['entropy_model.json']=codec.serialized
    common['sequence.json']=json.dumps([p.stem for p in validation],separators=(',',':')).encode()
    frames={'neural':[],'nonneural':[],'original':[]};details=[]
    status(output,'range_encode_decode_full_validation',cohort=label,total_windows=len(validation))
    started=time.monotonic()
    for number,path in enumerate(validation):
        original=source/'frames/compact'/f'{path.stem}.fsc'
        reference,shape,latent,side=huffman.decode(original.read_bytes())
        chosen,nonneural,info=codec.encode(original.read_bytes())
        for mode,data in (('neural',chosen),('nonneural',nonneural)):
            actual,decoded_shape,decoded_latent,decoded_side=codec.decode(data)
            np.testing.assert_array_equal(actual,reference)
            assert (shape,latent,side)==(decoded_shape,decoded_latent,decoded_side)
            destination=output/'frames'/mode/f'{path.stem}.fnp';destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(data);frames[mode].append(destination)
        frames['original'].append(original)
        details.append({'sample_id':path.stem,**info,'chosen_bytes':len(chosen),'nonneural_bytes':len(nonneural)})
        if number%250==0:status(output,'range_encode_decode_full_validation',completed=number,
            total_windows=len(validation),elapsed_sec=time.monotonic()-started,cohort=label)
    torch.set_num_threads(4)
    json_file(output/'window_bytes.json',details)
    with (source/'validation_metadata.csv').open() as stream:
        cells=sum(int(r['channels'])*int(r['time_samples']) for r in csv.DictReader(stream))
    summaries={}
    neural_used=any(r['chosen_method']==1 for r in details)
    for mode,paths in frames.items():
        if mode=='original':
            assets={name:(source/name).read_bytes() for name in
                    ('validation_metadata.csv','channel_maps.jsonl','preprocessing_policy.json','compact.tables.json')}
            assets['sequence.json']=common['sequence.json']
        else:
            assets=dict(common)
            if mode=='neural' and neural_used:assets['prior_inference.pt']=inference.read_bytes()
        bundle=output/f'{mode}.fsb';accounting=write_bundle(bundle,assets,paths)
        reader=read_bundle(bundle);assert next(reader)==assets
        for actual,path in zip(reader,paths,strict=True):assert actual==path.read_bytes()
        summaries[mode]={**accounting,'total_stream_bytes':bundle.stat().st_size,
                         'actual_ratio_shared_waveform_model':4*cells/bundle.stat().st_size,
                         'ratio_including_waveform_weights_once':4*cells/(bundle.stat().st_size+waveform_weight_bytes),
                         'all_indices_side_metadata_roundtripped':True,
                         'waveform_reconstruction_unchanged':True}
    non=summaries['nonneural']['total_stream_bytes'];neural=summaries['neural']['total_stream_bytes']
    result={'streams':summaries,'neural_savings_vs_best_nonneural_fraction':1-neural/non,
            'original_savings_fraction':1-neural/summaries['original']['total_stream_bytes'],
            'prior_parameters':sum(p.numel() for p in prior.parameters()),'prior_file_bytes':inference.stat().st_size,
            'selected_epoch':saved['epoch'],'trained_epochs':completed,
            'method_counts':{str(method):sum(r['chosen_method']==method for r in details) for method in (0,1,2)},
            'new_waveform_training_updates':0,'full_validation_windows':len(validation),
            'total_validation_coding_seconds':time.monotonic()-started,'final_model_promoted':False,
            'probability_portability':'Only reference CPU/software tested; decoded-index CRC rejects inconsistencies'}
    json_file(output/'results.json',result)
    (output/'COMPLETE').write_text('Learned index prior and complete lossless stream controls verified; no waveform changes or promotion.\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();output=args.output_dir.resolve()
    lock=(output.parent/f'.{output.name}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if torch.cuda.device_count()!=1 or '4060' not in torch.cuda.get_device_name(0):raise RuntimeError('Local RTX4060 required')
    if shutil.disk_usage(output.parent).free<8*2**30:raise RuntimeError('Insufficient disk headroom')
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    code=sorted((ROOT/'finetuning').glob('*.py'))+sorted((ROOT/'compression').glob('*.py'))
    code+=sorted((ROOT.parent/'smartgrids/vqvae').glob('*.py'))
    protocol={'kind':'frozen_index_checkerboard_prior_v1','seed':2037,'initial_epochs':8,'max_epochs':16,'steps_per_epoch':512,
              'optimizer':'AdamW lr1e-3 wd1e-5 clip1','selection':'minimum fixed pilot conditional bits per non-anchor symbol',
              'extension':'recent two-epoch-block gain >=0.5% and estimated total payload gain >=5%',
              'conditional_confirmation':'fold1 only if full measured stream saves >=2% versus best non-neural range/Huffman control',
              'waveform_models_frozen':True,'CDF_reference':'CPU FP32 one thread; range backend perfect=False',
              'prior_weights_transmitted_once_in_metadata':True,'no_test_inference':True,
              'code_sha256':{str(p):digest(p) for p in code},'torch':str(torch.__version__),
              'constriction':importlib.metadata.version('constriction')}
    protocol['requirements_entropy_sha256']=digest(ROOT/'requirements-entropy.txt')
    if args.resume:
        assert json.loads((output/'protocol.json').read_text())==protocol,'Prior-study protocol changed'
        if (output/'COMPLETE').exists():return
    else:
        output.mkdir(exist_ok=False);json_file(output/'protocol.json',protocol)
        for path in code:
            relative=path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('external_vqvae')/path.name
            target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    try:
        status(output,'fold0_index_prior')
        first=run_cohort('fold0',ROOT/'results/experiments/frozen_entropy_roi_fold0_20261030',
                         ROOT/'data/finetune/spatial_band_v1_fold0_20261024',output/'fold0',2037)
        results={'fold0':first}
        if first['neural_savings_vs_best_nonneural_fraction']>=.02:
            for path,sha in protocol['code_sha256'].items():assert digest(path)==sha,path
            status(output,'fold1_index_prior')
            results['fold1']=run_cohort('fold1',ROOT/'results/experiments/residual_confirmation_fold1_20261030/frozen_frames',
                         ROOT/'data/finetune/spatial_band_v1_fold1_20261026',output/'fold1',2038)
        json_file(output/'results.json',results)
        report=['# Learned lossless index prior','','Waveform models unchanged; probabilities trained on training indices only.','',
                '| Cohort | Original bundle ratio | Non-neural range/Huffman ratio | Neural ratio incl prior | Neural saving vs non-neural |',
                '|---|---:|---:|---:|---:|']
        for label,value in results.items():
            streams=value['streams']
            report.append(f'| {label} | {streams["original"]["actual_ratio_shared_waveform_model"]:.3f} | {streams["nonneural"]["actual_ratio_shared_waveform_model"]:.3f} | {streams["neural"]["actual_ratio_shared_waveform_model"]:.3f} | {100*value["neural_savings_vs_best_nonneural_fraction"]:.2f}% |')
        report+=['','Every selected index sequence, normalization side and metadata asset roundtripped.',
                 'Neural prior weights are included once. Original waveform model weights remain shared/predeployed.',
                 'CPU probability reproducibility across other hardware/software is not yet guaranteed; index CRC fails closed.',
                 'No final promotion or external validation.']
        (output/'REPORT.md').write_text('\n'.join(report)+'\n')
        (output/'COMPLETE').write_text('Bounded lossless index-prior study complete.\n');status(output,'complete',final_model_promoted=False)
    except BaseException as error:
        status(output,'failed',error=repr(error));(output/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
