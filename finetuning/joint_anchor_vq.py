"""Jointly trainable H2: discrete neural anchors, prediction and neural innovation."""
import torch
from torch import nn
import torch.nn.functional as F

from finetuning.model_factory import new_model

VERSION = "joint-anchor-innovation-vq-v1"


class AnchorPredictor(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, (3, 5), padding=(1, 2)), nn.SiLU(),
            nn.Conv2d(width, width, (3, 5), padding=(1, 2)), nn.SiLU(),
            nn.Conv2d(width, 1, (3, 5), padding=(1, 2)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, anchors, count):
        if not count:
            return anchors[:, :, :0]
        left = anchors[:, :, :count]
        indices = torch.arange(count, device=anchors.device).add(1).clamp_max(anchors.shape[2] - 1)
        right = anchors.index_select(2, indices)
        return .5 * (left + right) + self.net(torch.cat([left, right, left - right], dim=1))


def exact_branch_forward(model, target):
    """Hard forward is EXACT code vectors; only the zero-valued bridge has z gradient."""
    h, w = target.shape[-2:]
    z = model.encoder(F.pad(target, (0, (-w) % 16, 0, (-h) % 16)))
    _, loss, indices, perplexity = model.quantizer(z)
    b, d, lh, lw = z.shape
    exact = model.quantizer.embedding(indices.reshape(-1)).view(b, lh, lw, d).permute(0, 3, 1, 2).contiguous()
    q = exact.detach() + (z - z.detach()) if model.training else exact
    if not torch.equal(q.detach(), exact.detach()):
        raise AssertionError("Uncoded straight-through rounding residue")
    prediction = model.decoder(q)[..., :h, :w]
    return prediction, loss, indices.reshape(-1), (lh, lw), perplexity


class JointAnchorVQ(nn.Module):
    def __init__(self, anchor, innovation, mode="closed-loop", predictor_width=32):
        super().__init__()
        if mode not in ("independent", "closed-loop"):
            raise ValueError("Unsupported H2 mode")
        self.anchor = anchor
        self.innovation = innovation
        self.predictor = AnchorPredictor(predictor_width)
        self.mode = mode
        self.predictor_width = predictor_width
        if mode == "independent":
            self.predictor.requires_grad_(False)

    def predict(self, anchors, count):
        if self.mode == "independent":
            return torch.zeros(anchors.shape[0], 1, count, anchors.shape[-1],
                               dtype=anchors.dtype, device=anchors.device)
        return self.predictor(anchors, count)

    def forward(self, target):
        anchors, anchor_loss, anchor_ids, anchor_latent, anchor_perplexity = exact_branch_forward(
            self.anchor, target[:, :, ::2])
        count = target.shape[2] // 2
        prediction = self.predict(anchors, count)
        output = torch.empty_like(target)
        output[:, :, ::2] = anchors
        information = {"anchor_ids": anchor_ids, "anchor_latent": anchor_latent,
                       "anchor_perplexity": anchor_perplexity,
                       "innovation_perplexity": None, "innovation_ids": None, "innovation_latent": None}
        if count:
            residual, residual_loss, ids, latent, perplexity = exact_branch_forward(
                self.innovation, target[:, :, 1::2] - prediction)
            output[:, :, 1::2] = prediction + residual
            information.update(innovation_ids=ids, innovation_latent=latent,
                               innovation_perplexity=perplexity)
            quantizer_loss = .5 * (anchor_loss + residual_loss)
        else:
            quantizer_loss = anchor_loss
        return output, quantizer_loss, information

    @torch.no_grad()
    def decode_indices(self, anchor_ids, anchor_latent, innovation_ids, innovation_latent, shape):
        def branch(model, ids, latent, branch_shape):
            weight = model.quantizer.embedding.weight
            q = F.embedding(ids.reshape(-1), weight).view(1, *latent, weight.shape[1]).permute(0, 3, 1, 2).contiguous()
            return model.decoder(q)[..., :branch_shape[0], :branch_shape[1]]
        anchors = branch(self.anchor, anchor_ids, anchor_latent, ((shape[0] + 1) // 2, shape[1]))
        result = torch.empty(1, 1, *shape, device=anchors.device, dtype=anchors.dtype)
        result[:, :, ::2] = anchors
        if shape[0] > 1:
            residual = branch(self.innovation, innovation_ids, innovation_latent, (shape[0] // 2, shape[1]))
            result[:, :, 1::2] = self.predict(anchors, shape[0] // 2) + residual
        return result


def initialize(source, mode, device):
    config = {**source["config"], "exact_eval_quantization": True, "activation_checkpointing": True}
    anchor, innovation = new_model(config, device), new_model(config, device)
    anchor.load_state_dict(source["model_state_dict"])
    innovation.load_state_dict(source["model_state_dict"])
    if anchor.quantizer.embedding.weight.data_ptr() == innovation.quantizer.embedding.weight.data_ptr():
        raise AssertionError("Anchor and innovation weights must not alias")
    model = JointAnchorVQ(anchor, innovation, mode).to(device)
    return model, {"architecture": VERSION, "base_config": config, "mode": mode, "predictor_width": 32}


def build(config, state, device):
    if config.get("architecture") == "joint-encoder-observation-vq-v1":
        from finetuning.encoder_observation_vq import build as build_observation
        return build_observation(config, state, device)
    if config.get("architecture") != VERSION:
        raise ValueError("Unexpected joint anchor VQ architecture")
    anchor, innovation = new_model(config["base_config"], device), new_model(config["base_config"], device)
    model = JointAnchorVQ(anchor, innovation, config["mode"], config["predictor_width"]).to(device)
    model.load_state_dict(state)
    return model.eval()
