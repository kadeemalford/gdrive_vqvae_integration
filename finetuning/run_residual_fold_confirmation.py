#!/usr/bin/env python3
"""One local GPU: compatible compact base, frozen streams, FSQ, matched modes."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import traceback

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'smartgrids'))
from finetuning.run_spatial_abc_study import extension_decision,status
from finetuning.run_sampling_experiment import run,check_splits
from finetuning.run_capacity_study import digest,read_csv
from finetuning.model_factory import new_model
from finetuning.finetune_vqvae import save_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if torch.cuda.device_count()!=1 or '4060' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Confirmation requires the single local RTX4060')
    torch.set_num_threads(4)
    output=args.output_dir.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    lock=(output.parent/f'.{output.name}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    samples=ROOT/'data/finetune/spatial_band_v1_fold1_20261026';check_splits(samples,1)
    rows=read_csv(samples/'samples_manifest.csv')
    for split,expected in (('train',14620),('val',6940)):
        selected=[r for r in rows if r['split']==split]
        assert len(selected)==expected and len(list((samples/split).glob('*.npz')))==expected
        assert {r['spatial_band_policy'] for r in selected}=={'persistent-amplitude-v1'}
    assert {int(r['fold']) for r in rows if r['split']=='val'}=={2}
    audit=ROOT/'results/experiments/spatial_corpus_audit_fold1_20261026'
    assert (audit/'COMPLETE').exists()
    audited=json.loads((audit/'summary.json').read_text())
    for name,sha in audited['prepared_metadata_sha256'].items():assert digest(samples/name)==sha
    offshore=ROOT/'models/febus_finetune_base/best.pt'
    dense=ROOT/'results/experiments/spatial_abc_fold1_20261026/high_A/best_macro.pt'
    saved=torch.load(dense,map_location='cpu',weights_only=False)
    assert Path(saved['finetune']['samples_dir']).resolve()==samples
    del saved
    files=sorted((ROOT/'finetuning').glob('*.py'))+sorted((ROOT/'compression').glob('*.py'))
    files+=sorted((ROOT.parent/'smartgrids/vqvae').glob('*.py'))
    protocol={'kind':'compact_base_and_fsq_cross_fold_v1','test_fold':1,'validation_fold':2,'seed':2036,
              'samples_dir':str(samples),'offshore':str(offshore),'offshore_sha256':digest(offshore),
              'dense':str(dense),'dense_sha256':digest(dense),
              'base_initial_epochs':16,'base_max_epochs':24,'base_steps_per_epoch':4096,
              'refiner_initial_epochs':16,'refiner_max_epochs':24,'refiner_steps_per_epoch':1024,
              'base_extension':'one 8-epoch extension if recent four-epoch-block improvement >=0.2%',
              'refiner_extension':'one 8-epoch extension if recent improvement >=0.2% and gain vs base >=1%',
              'quantizer':'FSQ 8x8x32; 2048 states/11 nominal bits; 2016 refinement indices at maximum shape',
              'no_test_inference':True,'development_cv_not_external_validation':True,
              'sources_sha256':{str(p):digest(p) for p in files},
              'metadata_sha256':audited['prepared_metadata_sha256'],
              'audit_sha256':digest(audit/'summary.json'),
              'torch':str(torch.__version__),'cuda':torch.version.cuda}
    if args.resume:
        assert json.loads((output/'protocol.json').read_text())==protocol,'Confirmation protocol changed'
        if (output/'COMPLETE').exists():return
    else:
        output.mkdir(exist_ok=False);(output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for p in files:
            relative=p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('external_vqvae')/p.name
            target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    def verify():
        if shutil.disk_usage(output).free<12*2**30:raise RuntimeError('Less than 12 GiB free; no automatic deletion')
        for path,sha in protocol['sources_sha256'].items():assert digest(path)==sha,path
        for name,sha in protocol['metadata_sha256'].items():assert digest(samples/name)==sha,name
    try:
        verify()
        initial=output/'compact_initial.pt'
        if not initial.exists():
            saved=torch.load(offshore,map_location='cpu',weights_only=False)
            config={key:saved['config'][key] for key in
                    ('base_channels','embedding_dim','num_embeddings','downsample_levels','commitment_cost')}
            config.update(architecture='native-v1',latent_geometry='spatial-4x',
                          activation_checkpointing=True,exact_eval_quantization=True)
            torch.manual_seed(2036);model=new_model(config,'cpu')
            model.load_state_dict(saved['model_state_dict'],strict=True)
            save_checkpoint({'config':config,'model_state_dict':model.state_dict(),'epoch':0,
                             'initialization':{'source_sha256':protocol['offshore_sha256'],
                                               'source_epoch':saved['epoch'],'transferred':True}},initial)
            del saved,model
        base_dir=output/'compact_base'
        def train_base(target):
            verify();last=base_dir/'last.pt';completed=0
            if last.exists():
                state=torch.load(last,map_location='cpu',weights_only=False);completed=int(state['epoch']);del state
                for name in ('metrics.csv','validation_metrics.csv'):
                    history=read_csv(base_dir/name)
                    if history:assert int(history[-1]['epoch'])==completed
            if completed<target:
                status(output,'training_compact_base',completed_epochs=completed,target_epochs=target)
                command=['finetuning/finetune_vqvae.py','--repo-root',str(ROOT.parent/'smartgrids'),
                    '--checkpoint',str(initial),'--samples-dir',str(samples),'--output-dir',str(base_dir),
                    '--epochs',str(target-completed),'--steps-per-epoch','4096','--learning-rate','0.0001',
                    '--weight-decay','0.00001','--seed','2036','--device','cuda','--sampling','window',
                    '--normalization','robust-escape-v1','--latent-geometry','spatial-4x',
                    '--physical-validation','--validation-details','--evaluate-initial','--val-samples-per-file','2',
                    '--codebook-diagnostics','--channel-diagnostics','--spatial-weighted-validation',
                    '--early-stopping-metric','macro-rmse','--early-stopping-patience','25','--early-stopping-min-delta','0.0001']
                if completed:command+=['--resume',str(last)]
                run(command,output/'compact_base.log',resume=bool(completed),quiet=True)
            retained=output/f'compact_phase{target}.pt'
            if not retained.exists():
                state=torch.load(last,map_location='cpu',weights_only=False)
                assert state['epoch']==target;del state
                shutil.copy2(base_dir/'best_macro.pt',retained)
        train_base(16)
        ledger=output/'compact_extension.json'
        if ledger.exists():decision=json.loads(ledger.read_text())
        else:
            decision=extension_decision(read_csv(base_dir/'validation_metrics.csv'))
            ledger.write_text(json.dumps(decision,indent=2)+'\n')
        if decision['extend']:train_base(24)
        verify()
        frames=output/'frozen_frames'
        if not (frames/'COMPLETE').exists():
            status(output,'frozen_streams_and_dense_controls')
            command=['finetuning/run_frozen_entropy_roi.py','--output-dir',str(frames),
                     '--samples-dir',str(samples),'--compact-checkpoint',str(base_dir/'best_macro.pt'),
                     '--dense-checkpoint',str(dense),'--test-fold','1','--seed','2036']
            if frames.exists():command+=['--resume']
            run(command,output/'frozen_frames.log',resume=frames.exists(),quiet=True)
        verify()
        refinement=output/'residual_fsq'
        if not (refinement/'COMPLETE').exists():
            status(output,'training_fsq_refinement')
            command=['finetuning/run_residual_refinement.py','--output-dir',str(refinement),
                     '--quantizer','fsq-8x8x32','--samples-dir',str(samples),
                     '--base-checkpoint',str(base_dir/'best_macro.pt'),'--base-frames-source',str(frames),
                     '--test-fold','1','--seed','2036']
            if refinement.exists():command+=['--resume']
            run(command,output/'residual_fsq.log',resume=refinement.exists(),quiet=True)
        verify()
        extension=json.loads((refinement/'extension_decision.json').read_text())
        phase=24 if extension['extend'] else 16
        matched=output/'matched_modes'
        if not (matched/'COMPLETE').exists():
            if matched.exists():raise RuntimeError('Partial matched-mode analysis exists; inspect before restarting')
            status(output,'matched_byte_comparison')
            run(['finetuning/compare_refinement_modes.py','--source',str(frames),
                 '--residual',str(refinement/f'full_phase{phase}'),'--output-dir',str(matched)],
                output/'matched_modes.log',quiet=True)
        shutil.copy2(matched/'REPORT.md',output/'REPORT.md')
        (output/'COMPLETE').write_text('Compatible-base cross-fold FSQ and byte-control confirmation finished; no final promotion.\n')
        status(output,'complete',final_model_promoted=False)
    except BaseException as error:
        status(output,'failed',error=repr(error));(output/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
