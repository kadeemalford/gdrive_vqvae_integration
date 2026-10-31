#!/usr/bin/env python3
"""Match measured wire bytes by rate control, never by selecting validation scores."""
import argparse
from collections import defaultdict
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from compression.study_bundle import HEADER,metadata_archive,write_bundle,read_bundle
from finetuning.reconstruction_objective import aggregate_validation
from finetuning.compare_refinement_modes import dominates


def allocate_heap(records,ceilings):
    selected={r['sample_id']:'base' for r in records}
    by_id={r['sample_id']:r for r in records}
    used=defaultdict(int)
    for row in records:used[row['drive_id']]+=row['base']['wire_bytes']
    if any(used[file]>ceilings[file] for file in used):raise ValueError('Ceiling below basic representation')
    heap=[];blocked=defaultdict(list)
    def offer(row):
        current=selected[row['sample_id']]
        for mode in ('dense','residual'):
            if mode==current:continue
            a,b=row[current],row[mode]
            if not dominates(a['metrics'],b['metrics']):continue
            extra=b['wire_bytes']-a['wire_bytes']
            gain=(a['metrics']['relative_rmse']**2-b['metrics']['relative_rmse']**2)*row['cells']
            heapq.heappush(heap,(-gain/max(extra,1),row['sample_id'],mode,current,extra))
    for row in records:offer(row)
    while heap:
        item=heapq.heappop(heap)
        _,name,mode,previous,extra=item
        if selected[name]!=previous:continue
        row=by_id[name];file=row['drive_id']
        if used[file]+extra>ceilings[file]:
            blocked[file].append(item);continue
        selected[name]=mode;used[file]+=extra;offer(row)
        if extra<0:
            for entry in blocked.pop(file,[]):heapq.heappush(heap,entry)
    assert all(used[file]<=ceilings[file] for file in used)
    return selected


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load_records(source,residual):
    old={r['sample_id']:r for r in map(lambda p:json.loads(p.read_text()),(source/'window_cache').glob('*.json'))}
    newer={r['sample_id']:r for r in json.loads((residual/'window_metrics.json').read_text())}
    with (source/'validation_metadata.csv').open() as stream:metadata=list(csv.DictReader(stream))
    records=[]
    for meta in metadata:
        name=Path(meta['sample_path']).stem;a,b=old[name],newer[name]
        assert a['input_cells']==b['cells']
        common=set(a['compact']['metrics'])&set(a['dense']['metrics'])&set(b['refined'])
        basic=source/'frames/compact'/f'{name}.fsc';dense=source/'frames/dense'/f'{name}.fsc'
        refine=residual/'refinements'/f'{name}.fsc'
        row={'sample_id':name,'drive_id':meta['drive_id'],'cells':b['cells'],
             'family':meta['family'],'configuration':meta['configuration']}
        for mode,paths,measure in (('base',[basic],a['compact']['metrics']),
                                   ('dense',[dense],a['dense']['metrics']),
                                   ('residual',[basic,refine],b['refined'])):
            row[mode]={'paths':paths,'wire_bytes':sum(p.stat().st_size+4 for p in paths),
                       'metrics':{key:measure[key] for key in common}}
        records.append(row)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();output=args.output_dir.resolve()
    if shutil.disk_usage(output.parent).free<4*2**30:raise RuntimeError('Insufficient disk headroom')
    output.mkdir(exist_ok=False)
    cohorts={'fold0':(ROOT/'results/experiments/frozen_entropy_roi_fold0_20261030',
                       ROOT/'results/experiments/residual_refinement_fsq_fold0_20261030/full_phase24',
                       ROOT/'results/experiments/matched_refinement_modes_fold0_20261030'),
             'fold1':(ROOT/'results/experiments/residual_confirmation_fold1_20261030/frozen_frames',
                       ROOT/'results/experiments/residual_confirmation_fold1_20261030/residual_fsq/full_phase24',
                       ROOT/'results/experiments/residual_confirmation_fold1_20261030/matched_modes')}
    protocol={'kind':'measured_wire_rate_control_v1','reference':'previous dense-only streams',
              'targets':[1.1,1.25,1.5,2.0], 'search_iterations':24,
              'selection':'closest feasible stream size, never best validation fidelity among searched allocations',
              'allocation':'same guarded squared-RMSE gain per extra byte, heap implementation',
              'ceiling':'base wire bytes + scalar*(reference dense-used wire bytes - base wire bytes), per file',
              'monotonic_size_not_assumed':'retain closest feasible observed size; report unspent gap',
              'code_sha256':{str(Path(__file__)):digest(__file__),str(ROOT/'compression/study_bundle.py'):digest(ROOT/'compression/study_bundle.py')},
              'inputs':{name:[{'path':str(p),'results_sha256':digest(p/'results.json')} for p in values]
                        for name,values in cohorts.items()},
              'no_training':True,'no_inference':True,'retrospective_development':True,'final_model_promoted':False}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for filename in protocol['code_sha256']:
        path=Path(filename);dest=output/'code_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    results={}
    for cohort,(source,residual,matched) in cohorts.items():
        assert all((p/'COMPLETE').exists() for p in (source,residual,matched))
        records=load_records(source,residual)
        by_id={r['sample_id']:r for r in records};cells=sum(r['cells'] for r in records)
        shared={name:(source/name).read_bytes() for name in
                ('validation_metadata.csv','channel_maps.jsonl','preprocessing_policy.json')}
        tables={'base':(source/'compact.tables.json').read_bytes(),'dense':(source/'dense.tables.json').read_bytes(),
                'residual':(residual/'refinement.tables.json').read_bytes()}
        def assets_for(choices):
            registry={'base'}|set(choices.values())
            assets={**shared,**{f'{mode}.tables.json':tables[mode] for mode in registry}}
            assets['composition.json']=json.dumps({'base':'one compact frame with normalization side',
                'dense':'one dense frame with normalization side',
                'residual':'compact frame then FSQ refinement frame; sum in normalized coordinates; base side only',
                'source_window_required_by_decoder':False},sort_keys=True).encode()
            assets['sequence.json']=json.dumps([[r['sample_id'],choices[r['sample_id']]] for r in records],separators=(',',':')).encode()
            return assets
        def size_for(choices):
            return HEADER.size+len(metadata_archive(assets_for(choices)))+sum(
                r[choices[r['sample_id']]]['wire_bytes'] for r in records)
        previous_results=json.loads((matched/'results.json').read_text())
        for multiplier in protocol['targets']:
            name='dense_only_'+str(multiplier).replace('.','p')
            reference=json.loads((matched/f'{name}.modes.json').read_text())
            target=(matched/f'{name}.fsb').stat().st_size
            assert size_for(reference)==target,'Transport accounting differs from existing reference'
            basic=defaultdict(int);reference_wire=defaultdict(int)
            for row in records:
                basic[row['drive_id']]+=row['base']['wire_bytes']
                reference_wire[row['drive_id']]+=row[reference[row['sample_id']]]['wire_bytes']
            best=None;trace=[]
            low,high=0.,2.
            for iteration in range(24):
                scalar=(low+high)/2
                ceilings={file:basic[file]+math.floor(scalar*(reference_wire[file]-basic[file])) for file in basic}
                choices=allocate_heap(records,ceilings);size=size_for(choices)
                trace.append({'scalar':scalar,'stream_bytes':size,'target_bytes':target})
                if size<=target:
                    if best is None or size>best[0]:best=(size,scalar,choices)
                    low=scalar
                else:high=scalar
            if best is None:raise RuntimeError('No feasible byte-matched allocation found')
            size,scalar,choices=best
            label=f'{cohort}_{str(multiplier).replace(".","p")}'
            assets=assets_for(choices);paths=[];rows=[]
            for row in records:
                mode=choices[row['sample_id']];paths.extend(row[mode]['paths'])
                rows.append({'family':row['family'],'configuration':row['configuration'],
                             '_aggregation_weight':float(row['cells']),**row[mode]['metrics']})
            bundle=output/f'{label}.fsb';write_bundle(bundle,assets,paths)
            assert bundle.stat().st_size==size<=target
            reader=read_bundle(bundle);assert next(reader)==assets
            for frame,path in zip(reader,paths,strict=True):assert frame==path.read_bytes()
            macro,groups=aggregate_validation(rows)
            baseline=previous_results[name]
            results[label]={**macro,'reference_dense':baseline,'stream_bytes':size,'target_bytes':target,
                'unused_bytes':target-size,'relative_size_gap':(target-size)/target,
                'actual_ratio_shared_models':4*cells/size,
                'delta_rmse':macro['macro_relative_rmse']-baseline['macro_relative_rmse'],
                'delta_correlation':macro['macro_correlation']-baseline['macro_correlation'],
                'delta_p90':macro['macro_ordinary_channel_p90_relative_rmse']-baseline['macro_ordinary_channel_p90_relative_rmse'],
                'scalar_selected_by_size_only':scalar,'final_model_promoted':False}
            with (output/f'{label}.groups.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
            (output/f'{label}.rate_search.json').write_text(json.dumps(trace,indent=2)+'\n')
            (output/f'{label}.modes.json').write_text(json.dumps(choices,sort_keys=True)+'\n')
            print(label,'gap',target-size,'delta RMSE',results[label]['delta_rmse'],flush=True)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    report=['# Measured-wire matched hybrid control','','Selection by closest feasible size only, not by fidelity.','',
            '| Case | Unused target bytes | Delta RMSE | Delta correlation | Delta channel p90 |',
            '|---|---:|---:|---:|---:|']
    for name,row in results.items():report.append(f'| {name} | {row["unused_bytes"]} | {row["delta_rmse"]:.6f} | {row["delta_correlation"]:.6f} | {row["delta_p90"]:.6f} |')
    report+=['','Delta is hybrid minus frozen dense reference. Compare actual size gaps, model registry and physical tails.',
             'No new inference, training, test access, or external validation. All transported bytes roundtripped.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    (output/'COMPLETE').write_text('Wire-matched controls verified; no final promotion.\n')


if __name__=='__main__':main()
