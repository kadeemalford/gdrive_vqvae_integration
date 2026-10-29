#!/usr/bin/env python3
"""Bounded, resumable A/B/C queue on a verified frozen spatial corpus.

One GPU, two exactly index-matched rate points. No test inference or promotion.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import random
import shutil
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent/"smartgrids"))
from finetuning.audit_prepared_spatial_corpus import digest
from finetuning.finetune_vqvae import save_checkpoint
from finetuning.model_factory import new_model
from finetuning.reconstruction_objective import path_fingerprint, validation_metadata, stratified_validation
from finetuning.run_architecture_initialization_experiment import initialize_codebook
from finetuning.run_sampling_experiment import check_splits, run

RATES = {"high": "spatial4-temporal4", "compact": "spatial-4x"}
ARMS = {"C": "anisotropic-multiscale-v1", "A": "native-v1", "B": "native-v1"}


def read_csv(path):
    import csv
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def extension_decision(rows):
    """Predetermined bounded convergence rule; never rejects scratch for not winning."""
    values = [(int(r["epoch"]), float(r["macro_relative_rmse"])) for r in rows if int(r["epoch"])>0]
    if len(values)<8:
        return {"extend":False,"reason":"insufficient_completed_epochs"}
    baseline = min(v for _,v in values[-8:-4])
    recent = min(v for _,v in values[-4:])
    gain = (baseline-recent)/max(abs(baseline),1e-12)
    return {"extend":gain>=.002,"relative_gain_last_two_four_epoch_blocks":gain,
            "reason":"extend_8_epochs_if_recent_rmse_gain_at_least_0.2pct","extra_epochs":8 if gain>=.002 else 0}


def status(output, stage, **details):
    state = {"updated_at":datetime.now(timezone.utc).isoformat(),"stage":stage,**details}
    temporary = output/"STATUS.json.tmp"
    temporary.write_text(json.dumps(state,indent=2)+"\n")
    temporary.replace(output/"STATUS.json")
    (output/"STATUS.md").write_text("# Estado da comparação neural A/B/C\n\n```json\n"+json.dumps(state,indent=2,ensure_ascii=False)+"\n```\n\nSem promoção final; veja protocolos, logs e avaliações completas.\n")
    print(json.dumps(state),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--samples-dir",type=Path,required=True)
    parser.add_argument("--audit-dir",type=Path,required=True)
    parser.add_argument("--smoke-dir",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--test-fold",type=int,choices=range(4),default=0)
    parser.add_argument("--validation-fold",type=int,choices=range(4))
    parser.add_argument("--rates",choices=("both","high","compact"),default="both")
    parser.add_argument("--seed",type=int,default=2031)
    parser.add_argument("--historical-checkpoint",type=Path)
    args = parser.parse_args()
    output, samples = args.output_dir.resolve(),args.samples_dir.resolve()
    if args.resume and (output/"COMPLETE").is_file():
        print("Already complete; no training or rewriting of historical artifacts.",flush=True)
        return
    rates = RATES if args.rates=="both" else {args.rates:RATES[args.rates]}
    validation_fold = args.validation_fold if args.validation_fold is not None else (args.test_fold+1)%4
    if validation_fold==args.test_fold:
        raise ValueError("Test and validation folds must differ")
    for folder in (args.audit_dir,args.smoke_dir):
        if not (folder/"COMPLETE").is_file():
            raise ValueError(f"Completed preflight required: {folder}")
    check_splits(samples,args.test_fold)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rows = read_csv(samples/"samples_manifest.csv")
    if any(int(r["fold"])!=validation_fold for r in rows if r["split"]=="val"):
        raise ValueError("Prepared validation fold differs from protocol")
    for split in ("train","val"):
        if {r["family"] for r in rows if r["split"]==split}!={"doped","multicore","multimode","single_mode"}:
            raise ValueError("Four-family development corpus required")
    audit = json.loads((args.audit_dir/"summary.json").read_text())
    if audit.get("test_fold",0)!=args.test_fold:
        raise ValueError("Corpus audit fold differs from this protocol")
    for name, expected in audit["prepared_metadata_sha256"].items():
        if digest(samples/name)!=expected:
            raise ValueError("Prepared corpus changed after audit")
    source = ROOT/"models/febus_finetune_base/best.pt"
    references = {0:ROOT/"results/experiments/robust_rate_continuation_20261018/rate_4x/best_macro.pt",
                  1:ROOT/"results/experiments/high_rate_continuation_20261015/rate_4x/best_macro.pt"}
    historical = args.historical_checkpoint.resolve() if args.historical_checkpoint else references.get(args.test_fold)
    if historical is None:
        raise ValueError("Explicit fold-compatible historical reference required")
    historical_state = torch.load(historical,map_location="cpu",weights_only=False)
    historical_samples = Path(historical_state["finetune"]["samples_dir"])
    if historical_samples.name!=f"fold_{args.test_fold}":
        raise ValueError("Historical reference training fold is incompatible with this confirmation")
    historical_manifest = historical_samples/"samples_manifest.csv"
    if not historical_manifest.is_file():
        raise ValueError("Historical training manifest required to verify validation isolation")
    historical_train_groups = {(r["family"],r["configuration"]) for r in read_csv(historical_manifest) if r["split"]=="train"}
    current_val_groups = {(r["family"],r["configuration"]) for r in rows if r["split"]=="val"}
    if historical_train_groups & current_val_groups:
        raise ValueError("Historical reference was trained on current validation configurations")
    del historical_state
    rng = random.Random(args.seed)
    calibration = []
    for family in ("doped","multicore","multimode","single_mode"):
        candidates = sorted((ROOT/r["sample_path"]).resolve() for r in rows if r["split"]=="train" and r["family"]==family)
        calibration += rng.sample(candidates,8)
    calibration = sorted(calibration)
    val = sorted((samples/"val").glob("*.npz"))
    pilot = stratified_validation(val,validation_metadata(val,samples/"samples_manifest.csv"),2,args.seed)
    files = [ROOT/"finetuning"/(name+".py") for name in (
        "run_spatial_abc_study","anisotropic_vqvae","model_factory","latent_geometry","finetune_vqvae",
        "reconstruction_objective","physical_validation","reversible_normalization","robust_escape",
        "sampling","spatial_band_filter","evaluate_spatial_study","run_architecture_initialization_experiment",
        "evaluate_checkpoints","diagnose_reconstruction")]
    files += [Path("/home/murilo-barroso/source/smartgrids/vqvae")/(name+".py") for name in ("train_vqvae","evaluate_vqvae")]
    protocol = {"kind":"spatial_abc_rate_matched_v2","seed":args.seed,"test_fold":args.test_fold,"validation_fold":validation_fold,
        "source":str(source),"source_sha256":digest(source),"historical_4x":str(historical),"historical_sha256":digest(historical),
        "samples_dir":str(samples),"corpus_audit_sha256":digest(args.audit_dir/"summary.json"),
        "metadata_sha256":audit["prepared_metadata_sha256"],"array_sha256":audit["array_sha256"],
        "smoke_sha256":digest(args.smoke_dir/"summary.json"),"code_sha256":{str(p):digest(p) for p in files},
        "rates":rates,"arms":ARMS,"order":[f"{rate}_{arm}" for rate in rates for arm in ARMS],
        "initial_epochs":16,"steps_per_epoch":4096,"max_epochs_per_arm":24,
        "extension_rule":"after all six initial arms and full audits, extend each still improving by >=0.2% between its last two four-epoch blocks for exactly 8 additional epochs; no further extension",
        "learning_rate":1e-4,"weight_decay":1e-5,"sampling":"window","normalization":"robust-escape-v1",
        "objective":"0.5*MSE + 0.5*L1 + VQ; spectral and channel auxiliary weights zero",
        "quantizer":{"K":2048,"dimension":128,"commitment_cost":.25},
        "exact_eval_quantization":True,"activation_checkpointing":True,
        "pilot_windows":len(pilot),"pilot_hash":path_fingerprint(pilot),"pilot_cap":"2 windows per file and retained spatial-region index; seed fixed",
        "full_validation_windows":len(val),"full_validation_hash":path_fingerprint(val),
        "validation_aggregation":"Cell-weight retained tiles within each configuration, equal configurations within family, equal families; avoid giving a 1-channel remainder the weight of a 512-channel tile. Relative metrics still have per-tile normalization; channel-p90 is a weighted aggregate of per-tile quantiles, not a pooled whole-file quantile.",
        "calibration_train_windows":[str(p) for p in calibration],
        "calibration":"32 train windows, 8 per family; <=256 latent vectors/window; seed reservoir for B/C only; zero updates",
        "capacity_note":"C 26,302,273 parameters versus native 27,958,081; -5.9%, not exact parameter equality. Report CUDA/time; offshore historical compute not matched.",
        "selection":"Minimum pilot macro RMSE per arm; full validation of retained phase16/phase24 checkpoints and per-config/channel guards; no per-window oracle or final promotion",
        "guard":"Compare full-val RMSE/correlation/channel-p90 per configuration against A at same rate; high rate also against historical 4x on corrected corpus; no tolerance to systematic regression at this exploratory stage",
        "rate_note":"All arms use identical per-shape fixed-width index counts and common normalization side. Map/container/model costs not included; coverage loss not codec gain.",
        "no_test_inference":True,"software":{"torch":str(torch.__version__),"cuda":torch.version.cuda}}
    output.parent.mkdir(parents=True,exist_ok=True)
    lock = (output.parent/f".{output.name}.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.resume:
        if json.loads((output/"protocol.json").read_text())!=protocol:
            raise ValueError("Resume protocol/code/data changed")
        if (output/"COMPLETE").exists():
            print("Already complete",flush=True);return
    else:
        output.mkdir(exist_ok=False)
        (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    torch.set_num_threads(4)
    def verify_frozen():
        if shutil.disk_usage(output).free<12*1024**3:
            raise RuntimeError("Insufficient disk headroom; no automatic deletion")
        for path,expected in protocol["code_sha256"].items():
            if digest(path)!=expected:
                raise ValueError(f"Code changed during study: {path}")
        for name,expected in protocol["metadata_sha256"].items():
            if digest(samples/name)!=expected:
                raise ValueError("Frozen dataset metadata changed")
    def evaluate(specs, directory):
        if (directory/"COMPLETE").exists():
            return
        verify_frozen()
        command = ["finetuning/evaluate_spatial_study.py","--samples-dir",str(samples),"--output-dir",str(directory),"--test-fold",str(args.test_fold)]
        for label,path in specs.items():
            command += ["--candidate",f"{label}={path}"]
        if directory.exists():
            command += ["--resume"]
        run(command,directory.parent/f"{directory.name}.log",quiet=True)
    try:
        verify_frozen()
        status(output,"initialization",pilot_windows=len(pilot),full_validation_windows=len(val))
        saved = torch.load(source,map_location="cpu",weights_only=False)
        base_config = {key:saved["config"][key] for key in ("base_channels","embedding_dim","num_embeddings","downsample_levels","commitment_cost")}
        initial_dir = output/"initializations"
        initial_dir.mkdir(exist_ok=True)
        initialization_rows = {}
        for rate,geometry in rates.items():
            for arm,architecture in ARMS.items():
                label = f"{rate}_{arm}"
                path = initial_dir/f"{label}.pt"
                if not path.exists():
                    torch.manual_seed(args.seed)
                    config = {**base_config,"architecture":architecture,"latent_geometry":geometry,
                              "exact_eval_quantization":True,"activation_checkpointing":True}
                    model = new_model(config,"cuda")
                    if arm=="A":
                        model.load_state_dict(saved["model_state_dict"],strict=True)
                        vectors = 0
                    else:
                        vectors = initialize_codebook(model,calibration,args.seed,normalization="robust-escape-v1")
                    save_checkpoint({"config":config,"model_state_dict":model.cpu().state_dict(),"epoch":0,
                                     "initialization":{"arm":arm,"pretrained_tensors_loaded":arm=="A",
                                         "offshore_sha256":digest(source) if arm=="A" else None,"reservoir_vectors":vectors},
                                     "protocol_sha256":digest(output/"protocol.json")},path)
                    del model
                    torch.cuda.empty_cache()
                initial = torch.load(path,map_location="cpu",weights_only=False)
                if initial["protocol_sha256"]!=digest(output/"protocol.json"):
                    raise ValueError("Initialization provenance differs")
                initialization_rows[label] = {"path":str(path),"sha256":digest(path),"initialization":initial["initialization"]}
                del initial
        del saved
        (output/"initializations.json").write_text(json.dumps(initialization_rows,indent=2)+"\n")
        status(output,"full_validation_baselines")
        evaluate({"offshore_original":source,"historical_4x":historical,
                  **{f"A_initial_{rate}":initial_dir/f"{rate}_A.pt" for rate in rates}},output/"baselines")
        def train(label, target_epochs):
            verify_frozen()
            rate,arm = label.split("_")
            target = output/label
            last = target/"last.pt"
            completed = 0
            if last.exists():
                ckpt = torch.load(last,map_location="cpu",weights_only=False)
                completed = int(ckpt["epoch"])
                del ckpt
            if completed>=target_epochs:
                snapshot = output/f"retained_phase{target_epochs}"
                snapshot.mkdir(exist_ok=True)
                destination = snapshot/f"{label}.pt"
                if not destination.exists():
                    if completed!=target_epochs:
                        raise ValueError("Earlier phase snapshot missing after later training; cannot reconstruct selection")
                    shutil.copy2(target/"best_macro.pt",destination)
                evaluate({label:destination},output/f"full_phase{target_epochs}_{label}")
                return
            status(output,"training",arm=label,completed_epochs=completed,target_epochs=target_epochs)
            command = ["finetuning/finetune_vqvae.py","--repo-root","/home/murilo-barroso/source/smartgrids",
                "--checkpoint",str(initial_dir/f"{label}.pt"),"--samples-dir",str(samples),"--output-dir",str(target),
                "--epochs",str(target_epochs-completed),"--steps-per-epoch","4096","--learning-rate","0.0001",
                "--weight-decay","0.00001","--seed",str(args.seed),"--device","cuda","--sampling","window",
                "--normalization","robust-escape-v1","--latent-geometry",RATES[rate],"--physical-validation",
                "--validation-details","--evaluate-initial","--val-samples-per-file","2",
                "--codebook-diagnostics","--channel-diagnostics","--spatial-weighted-validation","--early-stopping-metric","macro-rmse",
                "--early-stopping-patience","25","--early-stopping-min-delta","0.0001"]
            if completed:
                command += ["--resume",str(last)]
            run(command,output/f"{label}.log",resume=bool(completed),quiet=True)
            snapshot = output/f"retained_phase{target_epochs}"
            snapshot.mkdir(exist_ok=True)
            destination = snapshot/f"{label}.pt"
            if not destination.exists():
                shutil.copy2(target/"best_macro.pt",destination)
            evaluate({label:destination},output/f"full_phase{target_epochs}_{label}")
        for label in protocol["order"]:
            train(label,16)
        status(output,"phase16_complete")
        decisions = {label:extension_decision(read_csv(output/label/"validation_metrics.csv")) for label in protocol["order"]}
        ledger = output/"extension_decisions.json"
        if ledger.exists():
            decisions = json.loads(ledger.read_text())
        else:
            ledger.write_text(json.dumps(decisions,indent=2)+"\n")
        for label in protocol["order"]:
            if decisions[label]["extend"]:
                train(label,24)
        status(output,"summarizing",extension_decisions=decisions)
        report = ["# Comparação A/B/C no corpus espacial corrigido", "",
                  "Seleção por RMSE macro na triagem; avaliações completas abaixo. Não é teste independente ou promoção final.",
                  "", "| Candidato | Fase | RMSE macro | Correlação macro | p90 canal macro | Razão nominal índices+side |", "|---|---:|---:|---:|---:|---:|"]
        results = {}
        for label in protocol["order"]:
            for phase in (16,24):
                path = output/f"full_phase{phase}_{label}"/f"{label}.json"
                if path.exists():
                    result = json.loads(path.read_text())
                    results[f"{label}_phase{phase}"] = result
                    report.append(f"| {label} | {phase} | {result['macro_relative_rmse']:.6f} | {result['macro_correlation']:.6f} | {result['macro_ordinary_channel_p90_relative_rmse']:.6f} | {result['pooled_index_plus_side_ratio']:.3f}× |")
        # Strict descriptive guards, not an automatic final promotion decision.
        guards = {}
        for name in results:
            label,phase_text = name.rsplit("_phase",1)
            rate,_ = label.split("_")
            reference_label = f"{rate}_A"
            candidate = read_csv(output/f"full_phase{phase_text}_{label}"/f"{label}.groups.csv")
            references = {reference_label+"_phase16":output/f"full_phase16_{reference_label}"/f"{reference_label}.groups.csv"}
            if rate=="high":
                references["historical_4x_corrected_corpus"] = output/"baselines/historical_4x.groups.csv"
            checks = {}
            for reference_name,reference_path in references.items():
                lookup = {(r["family"],r["configuration"]):r for r in read_csv(reference_path)}
                regressions = []
                for row in candidate:
                    base = lookup[(row["family"],row["configuration"])]
                    for metric,direction in (("relative_rmse",1),("correlation",-1),("ordinary_channel_p90_relative_rmse",1)):
                        delta = float(row[metric])-float(base[metric])
                        if direction*delta>1e-8:
                            regressions.append({"family":row["family"],"configuration":row["configuration"],"metric":metric,"delta":delta})
                checks[reference_name] = {"eligible":not regressions,"regressions":regressions}
            guards[name] = {"eligible":all(check["eligible"] for check in checks.values()),"references":checks}
        (output/"guards.json").write_text(json.dumps(guards,indent=2)+"\n")
        report += ["", "Guardas estritas por configuração em guards.json; nenhuma média isolada promove um modelo.",
                   "Pontos de fase16 têm exposição adicional pareada; fase24 é extensão seletiva e deve ser comparada com seu compute explícito.",
                   "Modelos ainda em melhora ao teto são inconclusivos quanto à convergência. Não confundir pré-treino histórico offshore com compute total pareado."]
        (output/"REPORT.md").write_text("\n".join(report)+"\n")
        (output/"results.json").write_text(json.dumps(results,indent=2)+"\n")
        (output/"COMPLETE").write_text("Bounded A/B/C queue and full validation complete; no test inference or final promotion.\n")
        status(output,"complete",candidates=len(results),final_model_promoted=False)
    except BaseException as error:
        status(output,"failed",error=repr(error),traceback=traceback.format_exc())
        (output/"FAILED").write_text(traceback.format_exc())
        raise


if __name__=="__main__":
    main()
