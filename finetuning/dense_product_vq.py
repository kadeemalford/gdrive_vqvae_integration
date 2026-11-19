"""Versioned dense full-tile flat/product codec; historical compact study unchanged."""
import io

import torch

from finetuning.factorized_vq import ProductEmbedding
from finetuning.model_factory import new_model

VERSION = "dense-full-tile-product-v1"


def initialize(source, variant, assets, device):
    if (variant not in ("flat", "product")
            or source["config"].get("latent_geometry") != "spatial4-temporal4"
            or source["config"].get("embedding_dim") != 128
            or source["config"].get("num_embeddings") != 2048):
        raise ValueError("Expected dense K2048x128 source and known variant")
    base_config = {**source["config"], "exact_eval_quantization": True, "activation_checkpointing": True}
    model = new_model(base_config, device)
    model.load_state_dict(source["model_state_dict"])
    if variant == "product":
        model.quantizer.embedding = ProductEmbedding(**{key: assets[key].to(device) for key in
                                                        ("left", "right", "basis", "center")})
    return model, {"architecture": VERSION, "variant": variant, "base_config": base_config}


def build(config, state, device):
    if config.get("architecture") == "local-internal-normalization-v1":
        from finetuning.local_internal_norm import build as build_local
        return build_local(config, state, device)
    if config.get("architecture") != VERSION or config.get("variant") not in ("flat", "product"):
        raise ValueError("Unknown dense product schema")
    model = new_model(config["base_config"], device)
    if config["variant"] == "product":
        model.quantizer.embedding = ProductEmbedding(torch.zeros(32, 64, device=device),
            torch.zeros(64, 64, device=device), torch.eye(128, device=device), torch.zeros(128, device=device))
    model.load_state_dict(state, strict=True)
    return model.eval()


def export(model, config):
    stream = io.BytesIO()
    torch.save({"config": config, "model_state_dict": {
        key: value.detach().cpu() for key, value in model.state_dict().items()}}, stream)
    return stream.getvalue()
