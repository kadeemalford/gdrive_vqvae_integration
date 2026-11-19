"""Encoder-only cross-phase observations; decoder and paid index grids unchanged."""
import torch
from torch import nn
import torch.nn.functional as F

from finetuning.joint_anchor_vq import JointAnchorVQ, build as build_source
from finetuning.model_factory import new_model

VERSION = "joint-encoder-observation-vq-v1"


class ContextStem(nn.Module):
    def __init__(self, base, enabled):
        super().__init__()
        if not isinstance(base, nn.Conv2d) or base.in_channels != 1 or base.groups != 1:
            raise ValueError("Expected one-channel native observation stem")
        self.base = base
        self.extra = nn.Conv2d(1, base.out_channels, base.kernel_size, stride=base.stride,
                               padding=base.padding, dilation=base.dilation, bias=False,
                               padding_mode=base.padding_mode).to(base.weight.device, base.weight.dtype)
        nn.init.zeros_(self.extra.weight)
        self.enabled = enabled
        self.current = None

    def forward(self, own):
        if self.enabled:
            if self.current is None or self.current.shape[-2] > own.shape[-2] or self.current.shape[-1] > own.shape[-1]:
                raise ValueError("Missing or invalid encoder-only observation")
            other = F.pad(self.current, (0, own.shape[-1] - self.current.shape[-1],
                                        0, own.shape[-2] - self.current.shape[-2]))
        else:
            other = torch.zeros_like(own)
        return self.base(own) + self.extra(other)


class ObservationJointVQ(JointAnchorVQ):
    def __init__(self, anchor, innovation, enabled, predictor_width=32):
        super().__init__(anchor, innovation, "closed-loop", predictor_width)
        self.encoder_context = enabled
        self.anchor.encoder.net[0] = ContextStem(self.anchor.encoder.net[0], enabled)
        self.innovation.encoder.net[0] = ContextStem(self.innovation.encoder.net[0], enabled)

    def forward(self, target):
        anchor_stem = self.anchor.encoder.net[0]
        innovation_stem = self.innovation.encoder.net[0]
        anchor_stem.current = target[:, :, 1::2]
        innovation_stem.current = target[:, :, ::2][:, :, :target.shape[2] // 2]
        try:
            return super().forward(target)
        finally:
            # No acquired samples are persisted/exported or available to decode_indices.
            anchor_stem.current = innovation_stem.current = None


def initialize(source, enabled, device):
    if source["config"].get("architecture") != "joint-anchor-innovation-vq-v1" or source["config"]["mode"] != "closed-loop":
        raise ValueError("Observation contrast requires the selected joint H2 closed-loop source")
    old = build_source(source["config"], source["model_state_dict"], device)
    model = ObservationJointVQ(old.anchor, old.innovation, enabled, old.predictor_width).to(device)
    model.predictor.load_state_dict(old.predictor.state_dict())
    config = {"architecture": VERSION, "base_config": source["config"]["base_config"],
              "mode": "closed-loop", "predictor_width": old.predictor_width, "encoder_context": enabled}
    return model, config


def build(config, state, device):
    if config.get("architecture") != VERSION or config.get("mode") != "closed-loop":
        raise ValueError("Unexpected encoder-observation architecture")
    model = ObservationJointVQ(new_model(config["base_config"], device),
                               new_model(config["base_config"], device),
                               config["encoder_context"], config["predictor_width"]).to(device)
    model.load_state_dict(state)
    return model.eval()
