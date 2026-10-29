#!/usr/bin/env python3
"""Full validation on the frozen spatial corpus using exact discrete code replay."""
import argparse
import csv
from functools import partial
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.audit_prepared_spatial_corpus import digest
from finetuning.finetune_vqvae import load_sample, load_valid_mask
from finetuning.model_factory import build_model
from finetuning.latent_geometry import latent_positions
from finetuning.physical_validation import restore_for_validation
from finetuning.reconstruction_objective import evaluate_detailed, validation_metadata
from finetuning.run_sampling_experiment import check_splits


def write_csv(path, rows):
    with Path(path).open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--candidate",action="append",required=True,help="label=checkpoint")
    parser.add_argument("--repo-root",default="/home/murilo-barroso/source/smartgrids")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--test-fold",type=int,choices=range(4),default=0)
    args = parser.parse_args()
    check_splits(args.samples_dir,args.test_fold)
    paths = sorted((args.samples_dir/"val").glob("*.npz"))
    metadata = validation_metadata(paths,args.samples_dir/"samples_manifest.csv")
    if {r["family"] for r in metadata.values()} != {"doped","multicore","multimode","single_mode"}:
        raise ValueError("Four-family validation required")
    protocol = {"samples_dir":str(args.samples_dir.resolve()),"test_fold":args.test_fold,
                "manifest_sha256":digest(args.samples_dir/"samples_manifest.csv"),
                "candidates":{text.split("=",1)[0]:digest(text.split("=",1)[1]) for text in args.candidate},
                "exact_eval_quantization":True,"spatial_weighting":True,"normalization":"robust-escape-v1"}
    if args.resume:
        if json.loads((args.output_dir/"protocol.json").read_text())!=protocol:
            raise ValueError("Full validation resume protocol differs")
    else:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        (args.output_dir/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    sys.path.insert(0,args.repo_root)
    from vqvae.evaluate_vqvae import load_checkpoint_config
    device = torch.device("cuda")
    summaries = []
    for candidate in args.candidate:
        label, checkpoint_path = candidate.split("=",1)
        if not label.replace("_","").isalnum():
            raise ValueError("Unsafe candidate label")
        existing = args.output_dir/f"{label}.json"
        if args.resume and existing.exists():
            summaries.append(json.loads(existing.read_text()))
            continue
        checkpoint, config = load_checkpoint_config(Path(checkpoint_path),device)
        config = {**config,"exact_eval_quantization":True,"activation_checkpointing":True}
        model = build_model(config,checkpoint,device).eval()
        epoch = checkpoint.get("epoch")
        del checkpoint
        totals = {"input_cells":0,"fixed_index_bits":0,"side_bytes":0}
        with (args.output_dir/f"{label}.windows.csv").open("w",newline="") as stream:
            writer = None
            def on_sample(path, metrics):
                nonlocal writer
                meta = metadata[path]
                cells = int(meta["channels"])*int(meta["time_samples"])
                bits = latent_positions(model,int(meta["channels"]),int(meta["time_samples"]))*math.ceil(math.log2(model.num_embeddings))
                side = round(metrics["side_bits_per_cell"]*cells/8)
                row = {"path":str(path),"drive_id":meta["drive_id"],"input_cells":cells,
                       "fixed_index_bits":bits,"side_bytes":side,**metrics}
                if writer is None:
                    writer = csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
                for key,value in (("input_cells",cells),("fixed_index_bits",bits),("side_bytes",side)):
                    totals[key] += value
            macro, groups = evaluate_detailed(model,paths,metadata,device,
                partial(load_sample,normalization="robust-escape-v1"),
                load_valid_mask=partial(load_valid_mask,normalization="robust-escape-v1"),
                fidelity_transform=partial(restore_for_validation,policy="robust-escape-v1",model=model,detailed_channels=True),
                on_sample=on_sample,spatial_weighting=True)
        result = {"label":label,"checkpoint":str(Path(checkpoint_path).resolve()),"sha256":digest(checkpoint_path),
                  "epoch":epoch,"geometry":model.latent_geometry,"parameters":sum(p.numel() for p in model.parameters()),
                  "windows":len(paths),"exact_eval_quantization":True,**macro,**totals,
                  "pooled_index_plus_side_ratio":32*totals["input_cells"]/(totals["fixed_index_bits"]+8*totals["side_bytes"])}
        write_csv(args.output_dir/f"{label}.groups.csv",groups)
        (args.output_dir/f"{label}.json").write_text(json.dumps(result,indent=2)+"\n")
        summaries.append(result)
        print(f"[full-val] {label} RMSE={macro['macro_relative_rmse']:.6f} corr={macro['macro_correlation']:.6f}",flush=True)
        del model
        torch.cuda.empty_cache()
    write_csv(args.output_dir/"summary.csv",summaries)
    (args.output_dir/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (args.output_dir/"COMPLETE").write_text("Full validation, exact discrete replay; no test access or model promotion.\n")


if __name__=="__main__":
    main()
