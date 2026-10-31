#!/usr/bin/env python3
"""Four frozen waveform points, small static repeat models, no neural updates."""
import argparse
import csv
import fcntl
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'smartgrids'))
from compression.symbol_stream import SymbolCodec
from compression.transition_symbol_stream import TransitionSymbolCodec,flags_and_literals
from compression.study_bundle import write_bundle,read_bundle
from finetuning.model_factory import build_model
from finetuning.robust_escape import encode_robust
from finetuning.run_frozen_entropy_roi import encoder_indices
from finetuning.run_sampling_experiment import check_splits
from finetuning.run_spatial_abc_study import status


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def json_file(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def run_case(label,source,samples,mode,output,cache=None):
    output.mkdir(exist_ok=True)
    protocol=json.loads((source/'protocol.json').read_text())
    check_splits(samples,protocol['test_fold'])
    for name,sha in protocol['metadata_sha256'].items():assert digest(samples/name)==sha
    checkpoint=Path(protocol['checkpoints'][mode]['path']);expected=protocol['checkpoints'][mode]['sha256']
    assert digest(checkpoint)==expected
    huffman=SymbolCodec.from_bytes((source/f'{mode}.tables.json').read_bytes())
    assert huffman.spec['checkpoint_sha256']==expected
    with (samples/'samples_manifest.csv').open() as stream:manifest=list(csv.DictReader(stream))
    if cache is not None:
        provenance=json.loads((cache.parent.parent/'provenance.json').read_text())
        assert provenance['waveform_sha256']==expected
    provenance={'mode':mode,'source_protocol_sha256':digest(source/'protocol.json'),
                'waveform_sha256':expected,'samples_dir':str(samples),'no_waveform_updates':True,
                'external_token_cache':str(cache) if cache is not None else None}
    if (output/'provenance.json').exists():assert json.loads((output/'provenance.json').read_text())==provenance
    else:json_file(output/'provenance.json',provenance)
    if (output/'COMPLETE').exists():return json.loads((output/'results.json').read_text())
    statistics={'marginal_counts':np.zeros(2048,dtype=np.int64),'orders':{name:{'repeats':0,'transitions':0,
                'literal_counts':np.zeros(2048,dtype=np.int64)} for name in ('temporal','spatial')}}
    wave=None
    with torch.no_grad():
        for number,row in enumerate(r for r in manifest if r['split']=='train'):
            source_path=ROOT/row['sample_path'];source_sha=digest(source_path)
            cached=cache/source_path.name if cache is not None else output/'tokens/train'/source_path.name
            if cached is not None and cached.exists():
                with np.load(cached,allow_pickle=False) as packet:
                    assert str(packet['source_sha256'])==source_sha
                    grid=packet['indices']
            else:
                if wave is None:
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    wave=build_model({**saved['config'],'exact_eval_quantization':True},saved,'cuda').eval();del saved
                with np.load(source_path,allow_pickle=False) as packet:
                    raw=np.asarray(packet['data'],dtype=np.float32)
                normalized=encode_robust(raw);indices,latent=encoder_indices(wave,normalized.values);grid=indices.reshape(latent)
                target=output/'tokens/train'/source_path.name;target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:np.savez_compressed(stream,indices=grid,source_sha256=source_sha)
                temporary.replace(target)
            statistics['marginal_counts']+=np.bincount(grid.reshape(-1),minlength=2048)
            for order in statistics['orders']:
                flags,literals=flags_and_literals(grid,order);value=statistics['orders'][order]
                value['repeats']+=int(flags.sum());value['transitions']+=int(flags.size)
                value['literal_counts']+=np.bincount(literals,minlength=2048)
            if number%1000==0:status(output,'fit_training_symbol_counts',case=label,completed=number)
    del wave;gc.collect();torch.cuda.empty_cache()
    statistics['marginal_counts']=statistics['marginal_counts'].tolist()
    for row in statistics['orders'].values():row['literal_counts']=row['literal_counts'].tolist()
    full=TransitionSymbolCodec(huffman,statistics)
    control=TransitionSymbolCodec(huffman,{'marginal_counts':statistics['marginal_counts'],'orders':{}})
    (output/'transition.tables.json').write_bytes(full.serialized)
    paths={'original':[],'control':[],'predicted':[]};details=[]
    selected=[r for r in manifest if r['split']=='val']
    for number,row in enumerate(selected):
        name=Path(row['sample_path']).stem
        original=source/'frames'/mode/f'{name}.fsc';data=original.read_bytes()
        symbols,shape,latent,side=huffman.decode(data)
        for label_mode,codec in (('control',control),('predicted',full)):
            encoded=codec.encode(data)
            recovered,decoded_shape,decoded_latent,decoded_side=codec.decode(encoded)
            np.testing.assert_array_equal(symbols,recovered)
            assert (shape,latent,side)==(decoded_shape,decoded_latent,decoded_side)
            target=output/'frames'/label_mode/f'{name}.ftr';target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==encoded,'Existing transition frame differs'
            else:target.write_bytes(encoded)
            paths[label_mode].append(target)
            if label_mode=='predicted':details.append({'sample_id':name,'method':encoded[5],
                'old_bytes':len(data),'predicted_bytes':len(encoded),'family':row['family'],'configuration':row['configuration']})
        paths['original'].append(original)
        if number%500==0:status(output,'encode_decode_validation',case=label,completed=number,total_windows=len(selected))
    json_file(output/'window_bytes.json',details)
    common={name:(source/name).read_bytes() for name in
            ('validation_metadata.csv','channel_maps.jsonl','preprocessing_policy.json')}
    sequence=json.dumps([Path(r['sample_path']).stem for r in selected],separators=(',',':')).encode()
    cells=sum(int(r['channels'])*int(r['time_samples']) for r in selected)
    waveform_metrics=json.loads((source/'results.json').read_text())[mode]
    summaries={}
    for kind,frames in paths.items():
        if kind=='original':assets={**common,'sequence.json':sequence,f'{mode}.tables.json':huffman.serialized}
        else:
            codec=control if kind=='control' else full
            assets={**common,'sequence.json':sequence,'huffman.tables.json':huffman.serialized,
                    'transition.tables.json':codec.serialized}
        bundle=output/f'{kind}.fsb'
        if bundle.exists():
            from compression.study_bundle import metadata_archive,HEADER as BUNDLE_HEADER
            accounting={'metadata_archive_bytes':len(metadata_archive(assets)),
                        'transport_header_and_lengths_bytes':BUNDLE_HEADER.size+4*len(frames),'frames':len(frames)}
        else:accounting=write_bundle(bundle,assets,frames)
        reader=read_bundle(bundle);assert next(reader)==assets
        for actual,path in zip(reader,frames,strict=True):assert actual==path.read_bytes()
        size=bundle.stat().st_size
        summaries[kind]={**accounting,'stream_bytes':size,'ratio_shared_waveform_model':4*cells/size,
            'ratio_including_waveform_weights_once':4*cells/(size+waveform_metrics['shared_float32_weight_bytes']),
            'indices_side_and_metadata_identical':True}
    result={'streams':summaries,'gain_vs_control_fraction':1-summaries['predicted']['stream_bytes']/summaries['control']['stream_bytes'],
            'gain_vs_original_fraction':1-summaries['predicted']['stream_bytes']/summaries['original']['stream_bytes'],
            'repeat_training_fractions':{order:values['repeats']/max(values['transitions'],1) for order,values in statistics['orders'].items()},
            'method_counts':{str(i):sum(r['method']==i for r in details) for i in range(4)},
            'macro_relative_rmse_unchanged':waveform_metrics['macro_relative_rmse'],
            'macro_correlation_unchanged':waveform_metrics['macro_correlation'],
            'no_waveform_training':True,'final_model_promoted':False}
    json_file(output/'results.json',result)
    (output/'COMPLETE').write_text('Static transition coding roundtripped; no waveform changes.\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();output=args.output_dir.resolve()
    lock=(output.parent/f'.{output.name}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if torch.cuda.device_count()!=1 or '4060' not in torch.cuda.get_device_name(0):raise RuntimeError('Local RTX4060 required')
    if importlib.metadata.version('constriction')!='0.5.0':raise RuntimeError('Pinned range backend required')
    if shutil.disk_usage(output.parent).free<8*2**30:raise RuntimeError('Insufficient disk headroom')
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    code=sorted((ROOT/'finetuning').glob('*.py'))+sorted((ROOT/'compression').glob('*.py'))
    code+=sorted((ROOT.parent/'smartgrids/vqvae').glob('*.py'))
    protocol={'kind':'frozen_neural_indices_static_transition_v1','directions':['temporal','spatial'],
              'reset':'literal forced at each row start in selected traversal',
              'training':'global flags and literal counts on train only; +1 literal and Beta(1,1) flag smoothing',
              'control':'minimum old Huffman and full-train marginal range, without repeat model',
              'selection':'minimum actual bytes with explicit method field',
              'neural_updates':0,'no_test_inference':True,'code_sha256':{str(p):digest(p) for p in code},
              'constriction':'0.5.0','torch':str(torch.__version__),'numpy':str(np.__version__),'final_model_promoted':False}
    if args.resume:
        assert json.loads((output/'protocol.json').read_text())==protocol,'Transition protocol changed'
        if (output/'COMPLETE').exists():return
    else:
        output.mkdir(exist_ok=False);json_file(output/'protocol.json',protocol)
        for path in code:
            relative=path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('external_vqvae')/path.name
            target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    try:
        results={}
        for fold in (0,1):
            source=(ROOT/'results/experiments/frozen_entropy_roi_fold0_20261030' if fold==0 else
                    ROOT/'results/experiments/residual_confirmation_fold1_20261030/frozen_frames')
            samples=ROOT/f'data/finetune/spatial_band_v1_fold{fold}_202610{24 if fold==0 else 26}'
            for mode in ('compact','dense'):
                label=f'fold{fold}_{mode}';status(output,'case',case=label)
                cache=ROOT/'results/experiments/neural_index_prior_20261030/fold0/tokens/train' if fold==0 and mode=='compact' else None
                results[label]=run_case(label,source,samples,mode,output/label,cache)
        json_file(output/'results.json',results)
        report=['# Static transition coding of frozen neural indices','','No new waveform training; all code sequences and side streams preserved.','',
                '| Case | Original ratio | Marginal/range control | Repeat-coded ratio | Gain vs control |',
                '|---|---:|---:|---:|---:|']
        for label,value in results.items():
            streams=value['streams']
            report.append(f'| {label} | {streams["original"]["ratio_shared_waveform_model"]:.3f} | {streams["control"]["ratio_shared_waveform_model"]:.3f} | {streams["predicted"]["ratio_shared_waveform_model"]:.3f} | {100*value["gain_vs_control_fraction"]:.2f}% |')
        report+=['','Rates include geometry metadata, model tables and transport boundaries; waveform weights are shared.',
                 'This is an entropy layer on a neural DAS codec, not a replacement waveform compressor.',
                 'No final acceptance or external validation.']
        (output/'REPORT.md').write_text('\n'.join(report)+'\n');(output/'COMPLETE').write_text('Four frozen points and all roundtrips complete.\n')
        status(output,'complete',final_model_promoted=False)
    except BaseException as error:
        status(output,'failed',error=repr(error));(output/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
