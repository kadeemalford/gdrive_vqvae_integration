"""Expanded train-file checks for selected and endpoint H2, no updates."""
import argparse
import json
import sys
from pathlib import Path

import torch

from finetuning.audit_prospective_preparation import sha256, write_json
from finetuning.fit_lifting_allocation import fitting_rows
from finetuning.joint_anchor_evaluation import evaluate, load_export, practical_reference
from finetuning.joint_anchor_vq import build
from finetuning.model_factory import build_model
from finetuning.run_spatial_opportunity import select


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor.resolve()))
    if not ((args.study / "PILOT_COMPLETE").exists() or (args.study / "BUDGET_COMPLETE").exists()):
        raise ValueError("Scientific training must complete before expanded checks")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    selected = select(args.corpus)
    fitting = fitting_rows(selected)
    checks = [row for row in selected if row["role"] == "check"]
    scales = json.loads(args.calibration.read_text())["scales"]
    maps = {m["channel_map_id"]: m for m in map(json.loads, (args.corpus / "channel_maps.jsonl").open())}
    selection = json.loads((args.study / "selection.json").read_text())
    training_protocol = json.loads((args.study / "protocol.json").read_text())
    for name, expected in training_protocol["sources"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen joint H2 implementation changed: {name}")
    if training_protocol["source_sha256"] != sha256(args.source):
        raise ValueError("Original source changed")
    write_json(args.output / "protocol.json", {
        "kind": "expanded_joint_h2_train_checks_v1", "parameter_updates": 0,
        "study_protocol_sha256": sha256(args.study / "protocol.json"),
        "selected_sha256": sha256(args.study / "best.pt"), "endpoint_sha256": sha256(args.study / "last.pt"),
        "selection_frozen_before_expanded_checks": selection, "check_windows": len(checks),
        "check_files": sorted({r["drive_id"] for r in checks}), "signal_split": "train",
        "source_sha256": sha256(args.source), "script_sha256": sha256(__file__),
        "limits": "Development training configurations; no independent validation or promotion"})
    reference_source = args.reference_source or args.source
    if training_protocol.get("reference_source_sha256", sha256(reference_source)) != sha256(reference_source):
        raise ValueError("Practical reference source changed")
    source = torch.load(reference_source, map_location="cpu", weights_only=False)
    teacher = build_model({**source["config"], "exact_eval_quantization": True}, source, "cuda").eval()
    practical_reference(teacher, sha256(reference_source), fitting, checks, args.corpus, scales, maps,
                        args.output / "practical_reference")
    del teacher
    results = []
    model = load_export((args.study / "best.pt").read_bytes(), "cuda")
    saved = torch.load(args.study / "best.pt", map_location="cpu", weights_only=True)
    result, _ = evaluate(model, saved["config"], fitting, checks, args.corpus, scales, maps,
                         args.output, selection["update"])
    result.update(candidate="selected", selection_update=selection["update"])
    results.append({k: v for k, v in result.items() if k != "records"})
    del model, saved
    endpoint = torch.load(args.study / "last.pt", map_location="cpu", weights_only=False)
    if endpoint["update"] != selection["update"]:
        model = build(endpoint["config"], endpoint["model_state_dict"], "cuda")
        result, _ = evaluate(model, endpoint["config"], fitting, checks, args.corpus, scales, maps,
                             args.output, endpoint["update"])
        result.update(candidate="endpoint", selection_update=endpoint["update"])
        results.append({k: v for k, v in result.items() if k != "records"})
    write_json(args.output / "results.json", results)
    (args.output / "COMPLETE").touch()
    print(json.dumps({"output": str(args.output), "complete": True, "parameter_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
