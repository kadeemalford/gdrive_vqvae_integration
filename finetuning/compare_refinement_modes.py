#!/usr/bin/env python3
"""Compare frozen dense substitution and FSQ residual at common byte ceilings."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compression.study_bundle import write_bundle,read_bundle
from finetuning.reconstruction_objective import aggregate_validation


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dominates(a,b):
    return (b['relative_rmse']<a['relative_rmse'] and b['correlation']>=a['correlation']
            and b['ordinary_channel_p90_relative_rmse']<=a['ordinary_channel_p90_relative_rmse'])


def allocate(records,allowed,factor,previous=None):
    selected=dict(previous or {r['sample_id']:'base' for r in records})
    base_bytes=sum(r['base']['bytes'] for r in records)
    used=sum(r[selected[r['sample_id']]]['bytes'] for r in records)
    ceiling=math.floor(base_bytes*factor)
    assert used<=ceiling
    while True:
        options=[]
        for row in records:
            current=selected[row['sample_id']]
            for mode in allowed:
                if mode==current:continue
                a,b=row[current],row[mode]
                if not dominates(a['metrics'],b['metrics']):continue
                extra=b['bytes']-a['bytes']
                if used+extra>ceiling:continue
                gain=(a['metrics']['relative_rmse']**2-b['metrics']['relative_rmse']**2)*row['cells']
                options.append((-gain/max(extra,1),row['sample_id'],mode,extra))
        if not options:break
        _,sample,mode,extra=min(options)
        selected[sample]=mode;used+=extra
    assert used<=ceiling
    return selected,{'base_frame_bytes':base_bytes,'frame_bytes':used,'ceiling_bytes':ceiling}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'results/experiments/frozen_entropy_roi_fold0_20261030')
    parser.add_argument('--residual',type=Path,default=ROOT/'results/experiments/residual_refinement_fsq_fold0_20261030/full_phase24')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if shutil.disk_usage(output.parent).free<4*2**30:raise RuntimeError('Insufficient disk headroom')
    source=args.source.resolve()
    residual=args.residual.resolve()
    assert (source/'COMPLETE').exists() and (residual/'COMPLETE').exists()
    output.mkdir(exist_ok=False)
    protocol={'kind':'matched_byte_frozen_dense_vs_residual_v1','budgets':[1.1,1.25,1.5,2.0],
        'score':'positive squared relative RMSE reduction * cells / extra framed bytes',
        'guard':'candidate must improve RMSE and not worsen correlation or ordinary-channel-p90 against the current mode',
        'nested':'retain lower-budget choices; upgrades allowed only if all guard metrics preserved',
        'candidates':['dense_only','residual_only','hybrid'],
        'source_sha256':digest(source/'results.json'),'residual_sha256':digest(residual/'results.json'),
        'code_sha256':{str(Path(__file__)):digest(__file__),str(ROOT/'compression/study_bundle.py'):digest(ROOT/'compression/study_bundle.py')},
        'no_training':True,'no_new_inference':True,'retrospective_development':True,'final_model_promoted':False}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for name in protocol['code_sha256']:
        path=Path(name);dest=output/'code_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    old={r['sample_id']:r for r in map(lambda p:json.loads(p.read_text()),(source/'window_cache').glob('*.json'))}
    refinement={r['sample_id']:r for r in json.loads((residual/'window_metrics.json').read_text())}
    with (source/'validation_metadata.csv').open() as stream:metadata=list(csv.DictReader(stream))
    records=[]
    for meta in metadata:
        sample=Path(meta['sample_path']).stem;a,b=old[sample],refinement[sample]
        assert a['input_cells']==b['cells']
        for metric in ('relative_rmse','correlation','ordinary_channel_p90_relative_rmse'):
            np.testing.assert_allclose(a['compact']['metrics'][metric],b['base'][metric],rtol=1e-6,atol=1e-7)
        base_path=source/'frames/compact'/f'{sample}.fsc'
        dense_path=source/'frames/dense'/f'{sample}.fsc'
        residual_path=residual/'refinements'/f'{sample}.fsc'
        common=set(a['compact']['metrics'])&set(a['dense']['metrics'])&set(b['refined'])
        records.append({'sample_id':sample,'drive_id':meta['drive_id'],'family':meta['family'],
            'configuration':meta['configuration'],'cells':b['cells'],
            'base':{'paths':[base_path],'bytes':base_path.stat().st_size,
                    'metrics':{k:a['compact']['metrics'][k] for k in common}},
            'dense':{'paths':[dense_path],'bytes':dense_path.stat().st_size,
                     'metrics':{k:a['dense']['metrics'][k] for k in common}},
            'residual':{'paths':[base_path,residual_path],'bytes':base_path.stat().st_size+residual_path.stat().st_size,
                        'metrics':{k:b['refined'][k] for k in common}}})
    assert records and len(records)==len(metadata)
    by_file=defaultdict(list)
    for row in records:by_file[row['drive_id']].append(row)
    cases={'base':{r['sample_id']:'base' for r in records}}
    accounting={}
    for label,allowed in (('dense_only',['dense']),('residual_only',['residual']),('hybrid',['dense','residual'])):
        previous={}
        for factor in protocol['budgets']:
            selections={};budgets={}
            for file,values in by_file.items():
                previous[file],budgets[file]=allocate(values,allowed,factor,previous.get(file))
                selections.update(previous[file])
            name=f'{label}_{str(factor).replace(".","p")}'
            cases[name]=selections;accounting[name]=budgets
    shared={name:(source/name).read_bytes() for name in
            ('validation_metadata.csv','channel_maps.jsonl','preprocessing_policy.json')}
    table_paths={'base':source/'compact.tables.json','dense':source/'dense.tables.json',
                 'residual':residual/'refinement.tables.json'}
    weights={'base':27958081*4,'dense':27958081*4,'residual':7094628*4}
    summaries={};grouped={}
    total_cells=sum(r['cells'] for r in records)
    for label,choices in cases.items():
        used=set(choices.values());registry={'base'}|used
        assets={**shared,**{f'{mode}.tables.json':table_paths[mode].read_bytes() for mode in registry}}
        assets['composition.json']=json.dumps({'base':'one compact frame with normalization side',
            'dense':'one dense frame with normalization side',
            'residual':'compact frame then FSQ refinement frame; sum in normalized coordinates; base side only',
            'source_window_required_by_decoder':False},sort_keys=True).encode()
        assets['sequence.json']=json.dumps([[r['sample_id'],choices[r['sample_id']]] for r in records],separators=(',',':')).encode()
        paths=[];rows=[]
        for row in records:
            mode=choices[row['sample_id']];paths.extend(row[mode]['paths'])
            rows.append({'family':row['family'],'configuration':row['configuration'],
                         '_aggregation_weight':float(row['cells']),**row[mode]['metrics']})
        bundle=output/f'{label}.fsb';wire=write_bundle(bundle,assets,paths)
        reader=read_bundle(bundle);assert next(reader)==assets
        for frame,path in zip(reader,paths,strict=True):assert frame==path.read_bytes()
        macro,groups=aggregate_validation(rows);grouped[label]=groups
        model_bytes=sum(weights[k] for k in registry)
        summaries[label]={**macro,**wire,'stream_bytes':bundle.stat().st_size,
            'actual_ratio_shared_models':4*total_cells/bundle.stat().st_size,
            'ratio_including_used_float32_models_once':4*total_cells/(bundle.stat().st_size+model_bytes),
            'used_model_parameter_bytes':model_bytes,
            'dense_cell_fraction':sum(r['cells'] for r in records if choices[r['sample_id']]=='dense')/total_cells,
            'residual_cell_fraction':sum(r['cells'] for r in records if choices[r['sample_id']]=='residual')/total_cells,
            'all_metadata_and_frames_roundtripped':True}
        with (output/f'{label}.groups.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
        (output/f'{label}.modes.json').write_text(json.dumps(choices,sort_keys=True)+'\n')
        print(label,macro['macro_relative_rmse'],macro['macro_correlation'],summaries[label]['actual_ratio_shared_models'],flush=True)
    (output/'results.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (output/'file_budgets.json').write_text(json.dumps(accounting,indent=2)+'\n')
    base={(r['family'],r['configuration']):r for r in grouped['base']};guards={}
    for label,groups in grouped.items():
        bad=[]
        for row in groups:
            original=base[row['family'],row['configuration']]
            for metric,direction in (('relative_rmse',1),('correlation',-1),('ordinary_channel_p90_relative_rmse',1)):
                delta=row[metric]-original[metric]
                if direction*delta>1e-8:bad.append({'family':row['family'],'configuration':row['configuration'],'metric':metric,'delta':delta})
        guards[label]={'regressions_vs_base':bad,'final_model_promoted':False}
    (output/'guards.json').write_text(json.dumps(guards,indent=2)+'\n')
    report=['# Matched byte-ceiling dense and residual modes','','Retrospective frozen development comparison; no final promotion.','',
            '| Mode | RMSE | Correlation | Channel p90 | Actual ratio | Dense cells | Residual cells |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for label,row in summaries.items():
        report.append(f'| {label} | {row["macro_relative_rmse"]:.6f} | {row["macro_correlation"]:.6f} | {row["macro_ordinary_channel_p90_relative_rmse"]:.6f} | {row["actual_ratio_shared_models"]:.3f} | {row["dense_cell_fraction"]:.3f} | {row["residual_cell_fraction"]:.3f} |')
    report+=['','All strategies share the guard and gain-per-extra-byte score. Byte ceilings are not exact equal byte consumption.',
             'Larger budgets preserve or improve each previously selected window on all three guarded metrics.',
             'Hybrid may require three neural models; used parameter costs reported separately.',
             'All frames and physical/acquisition metadata recovered exactly; no new inference or test access.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    (output/'COMPLETE').write_text('Matched-budget bundles verified; no promotion.\n')


if __name__=='__main__':main()
