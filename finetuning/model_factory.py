"""Local, versioned architecture factory; never infer architecture from weight shapes."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as checkpoint_activations

try:
    from .latent_geometry import apply_geometry, checkpoint_geometry
except ImportError:
    from latent_geometry import apply_geometry, checkpoint_geometry

ARCHITECTURES = ("native-v1", "temporal-dilated-v1", "anisotropic-multiscale-v1")


def exact_eval_quantization(model, config):
    """Opt-in exact code-vector replay in evaluation; preserve training gradients.

    The straight-through z + (q-z).detach() can differ numerically from q. These
    rounding residuals must not become uncoded information in codec evaluation.
    Historical checkpoints keep their original behavior unless opted in.
    """
    if config.get("activation_checkpointing", False) and architecture(config) != "anisotropic-multiscale-v1":
        from vqvae.train_vqvae import ResidualBlock
        for module in model.modules():
            if isinstance(module, ResidualBlock) and not hasattr(module, "_activation_checkpointed"):
                original = module.forward
                def forward(x, _original=original, _module=module):
                    if _module.training and torch.is_grad_enabled():
                        return checkpoint_activations(_original, x, use_reentrant=False)
                    return _original(x)
                module.forward = forward
                module._activation_checkpointed = True
    if config.get("exact_eval_quantization", False) and not hasattr(model, "_exact_eval_hook"):
        def replace_eval_vectors(module, inputs, output):
            if module.training:
                return None
            quantized, loss, indices, perplexity = output
            batch, channels, height, width = quantized.shape
            exact = module.embedding(indices).view(batch, height, width, channels).permute(0,3,1,2).contiguous()
            return exact, loss, indices, perplexity
        model._exact_eval_hook = model.quantizer.register_forward_hook(replace_eval_vectors)
    return model


def architecture(config):
    name = config.get("architecture", "native-v1")
    if name not in ARCHITECTURES:
        raise ValueError("Unknown checkpoint architecture")
    return name


def temporal_residuals(model):
    """Replace second residual 3x3 by 1x9, preserving parameter count and grid.

    Dilations 1/2/4 repeat within encoder and decoder separately. No skip bypasses
    quantization; spatial mixing remains in the first 3x3 of every residual block.
    """
    from vqvae.train_vqvae import ResidualBlock
    for module in (model.encoder, model.decoder):
        blocks = [block for block in module.modules() if isinstance(block, ResidualBlock)]
        if not blocks:
            raise ValueError("Unsupported residual topology")
        for index, block in enumerate(blocks):
            old = block.net[-1]
            if not isinstance(old, nn.Conv2d) or old.kernel_size != (3, 3):
                raise ValueError("Unsupported residual convolution")
            dilation = (1, 2, 4)[index % 3]
            block.net[-1] = nn.Conv2d(old.in_channels, old.out_channels, (1, 9),
                                     padding=(0, 4 * dilation), dilation=(1, dilation),
                                     bias=old.bias is not None)
    return model


def new_model(config, device):
    if architecture(config) == "anisotropic-multiscale-v1":
        try:
            from .anisotropic_vqvae import AnisotropicVQVAE
        except ImportError:
            from anisotropic_vqvae import AnisotropicVQVAE
        model = AnisotropicVQVAE(**{key: config[key] for key in
            ("base_channels", "embedding_dim", "num_embeddings", "downsample_levels", "commitment_cost")})
        apply_geometry(model, checkpoint_geometry({"config": config}))
        return exact_eval_quantization(model.to(device), config)
    from vqvae.train_vqvae import VQVAE
    model = VQVAE(in_channels=1, base_channels=int(config["base_channels"]),
                  embedding_dim=int(config["embedding_dim"]), num_embeddings=int(config["num_embeddings"]),
                  downsample_levels=int(config["downsample_levels"]),
                  commitment_cost=float(config.get("commitment_cost", .25)))
    if architecture(config) == "temporal-dilated-v1":
        temporal_residuals(model)
    apply_geometry(model, checkpoint_geometry({"config": config}))
    return exact_eval_quantization(model.to(device), config)


def build_model(config, checkpoint, device):
    if architecture(config) != architecture(checkpoint.get("config", {})):
        raise ValueError("Conflicting checkpoint architecture")
    if architecture(config) == "native-v1":
        from vqvae.evaluate_vqvae import build_model as native_model
        model = native_model(config, checkpoint, device)
    else:
        model = new_model(config, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    apply_geometry(model, checkpoint_geometry(checkpoint))
    return exact_eval_quantization(model, config).eval()
