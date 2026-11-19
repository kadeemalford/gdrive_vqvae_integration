"""Native-header conditioning with zero-initialized, capacity-matched FiLM."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from finetuning.model_factory import new_model

VERSION = "physical-film-v1"
FIELDS = ("dx_m", "dt_s", "GaugeLength", "PulseWidth", "PulseRateFreq", "SamplingRate",
          "SamplingRes", "Oversampling", "AmpliPower", "DerivationTime")
METADATA = Path("results/experiments/physical_metadata_preflight_fold0_20261115/CONSISTENCY_AND_PROVENANCE.json")
RECORDS = METADATA.with_name("results.json")


def metadata_plan(summary, records):
    groups = {}
    for row in records:
        fields = summary["numeric_fields_by_file"][row["drive_id"]]
        vector = tuple(float(fields[key]) for key in FIELDS)
        if not np.isfinite(vector).all() or min(vector) <= 0:
            raise ValueError("Positive finite native header fields required")
        group = row["group_id"]
        if group in groups and groups[group] != vector:
            raise ValueError("Conflicting metadata within a configuration")
        groups[group] = vector
    if len(groups) != 10 or {g.split("/")[0] for g in groups} != {"doped", "multicore", "multimode", "single_mode"}:
        raise ValueError("Expected ten configurations and four families")
    unique_single = sorted(g for g, v in groups.items() if g.startswith("single_mode/")
                           and sum(other == v for other in groups.values()) == 1)
    if len(unique_single) < 2:
        raise ValueError("Cannot reserve unique hardware combinations")
    heldout = [unique_single[0], unique_single[-1]]
    if heldout != ["single_mode/Configuration 11", "single_mode/Configuration 8"]:
        raise ValueError("Metadata selection differs from registered protocol")
    seen = sorted(set(groups) - set(heldout))
    values = np.log2(np.array([groups[g] for g in seen], np.float64))
    center = values.mean(0)
    scale = values.std(0)
    constant = np.ptp(values, axis=0) == 0
    center[constant] = values[0, constant]
    scale[constant] = 1
    return {"fields": list(FIELDS), "groups": {g: list(v) for g, v in groups.items()},
            "seen": seen, "heldout": heldout, "log_center": center.tolist(), "log_scale": scale.tolist()}


def load_plan():
    return metadata_plan(json.loads(METADATA.read_text()), json.loads(RECORDS.read_text())["records"])


def condition(plan, method, group, missing=False):
    if method not in ("constant", "hardware", "category"):
        raise ValueError("Unknown conditioning method")
    result = np.zeros(11, np.float32)
    if method == "constant":
        return result
    if missing:
        result[-1] = 1
        return result
    if method == "category":
        if group in plan["seen"]:
            result[plan["seen"].index(group)] = 1
    elif method == "hardware":
        result[:10] = ((np.log2(plan["groups"][group]) - np.array(plan["log_center"]))
                       / np.array(plan["log_scale"])).astype(np.float32)
    else:
        raise ValueError("Unknown conditioning method")
    return result


class ConditionalGroupNorm(nn.GroupNorm):
    def __init__(self, original):
        super().__init__(original.num_groups, original.num_channels, original.eps, original.affine,
                         device=original.weight.device, dtype=original.weight.dtype)
        self.load_state_dict(original.state_dict())
        self.modulator = nn.Sequential(nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 2 * self.num_channels)).to(
            device=self.weight.device, dtype=self.weight.dtype)
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)
        self.register_buffer("condition", torch.zeros(11, device=self.weight.device), persistent=False)

    def forward(self, x):
        gamma, beta = self.modulator(self.condition).chunk(2)
        return (super().forward(x) * (1 + .1 * gamma.tanh()[None, :, None, None])
                + .1 * beta.tanh()[None, :, None, None])


def replace_norms(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, ConditionalGroupNorm(child))
            count += 1
        else:
            count += replace_norms(child)
    return count


def set_condition(model, vector):
    vector = np.asarray(vector, np.float32)
    if vector.shape != (11,) or not np.isfinite(vector).all():
        raise ValueError("Invalid conditioning vector")
    for module in model.modules():
        if isinstance(module, ConditionalGroupNorm):
            module.condition.copy_(torch.from_numpy(vector.copy()).to(module.condition.device))


def initialize(source, method, plan, device):
    config = {**source["config"], "latent_geometry": "spatial4-temporal4",
              "exact_eval_quantization": True, "activation_checkpointing": True}
    model = new_model(config, device)
    model.load_state_dict(source["model_state_dict"], strict=True)
    count = replace_norms(model)
    if not count:
        raise ValueError("No GroupNorm layers found")
    return model, {"architecture": VERSION, "base_config": config, "method": method,
                   "plan": plan, "norm_layers": count}


def build(config, state, device):
    if config.get("architecture") != VERSION:
        raise ValueError("Unknown conditioned model schema")
    model = new_model(config["base_config"], device)
    if replace_norms(model) != config["norm_layers"]:
        raise ValueError("Conditioned topology mismatch")
    model.load_state_dict(state, strict=True)
    return model.eval()
